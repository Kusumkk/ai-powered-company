"""Small, dependency-free RAG server for the Company Policy Assistant.

Retrieval uses BM25 over individual policy statements. If OPENAI_API_KEY is set,
the retrieved passages are sent to the configured OpenAI model for grounded
generation; without a key, the app returns a clearly labelled extractive answer.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "policies"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
STOP = set("a an the is are was were be been being do does did done i me my we our you your can could should would will may might must to for of in on at by from with about and or it this that what when how who where why please tell give get per each any there they their them into after before above below within".split())
SYNONYMS = {
    "vacation": "leave", "pto": "leave", "holiday": "leave", "timeoff": "leave",
    "rollover": "carried", "carryforward": "carried", "carry": "carried",
    "claim": "claims", "reimburse": "reimbursement", "reimbursed": "reimbursement",
    "refund": "reimbursement", "invoice": "invoices", "receipt": "receipts",
    "deadline": "submitted", "file": "submitted", "filing": "submitted",
    "notice": "advance", "request": "apply", "permission": "approval",
    "permissions": "approval", "authorized": "approval", "authorize": "approval",
    "database": "production", "db": "production", "internship": "intern",
    "computer": "laptop", "proof": "receipts", "yearly": "calendar",
}

SECTION_NAMES = {
    "leave_policy.txt": ["Annual leave", "Applying for leave", "Carry forward", "Intern leave"],
    "expense_policy.txt": ["Eligible expenses", "Claim deadline", "Required receipts", "Manager approval"],
    "onboarding_policy.txt": ["Identity and documents", "Equipment and applications", "Production database access", "Intern access"],
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    result = []
    for word in words:
        if word in STOP or len(word) < 2:
            continue
        word = SYNONYMS.get(word, word)
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        result.append(word)
    return result


def load_chunks() -> list[dict]:
    chunks = []
    for filename, section_names in SECTION_NAMES.items():
        path = DOCS / filename
        lines = path.read_text(encoding="utf-8").splitlines()
        facts = [line.strip() for line in lines[1:] if line.strip()]
        for index, text in enumerate(facts):
            chunks.append({
                "id": f"{filename}:{index + 2}", "doc": filename,
                "title": lines[0].strip(), "section": section_names[index],
                "line": index + 2, "text": text, "tokens": tokenize(text),
            })
    return chunks


CHUNKS = load_chunks()


def retrieve(question: str, limit: int = 4) -> list[dict]:
    query = tokenize(question)
    if not query:
        return []
    counts = Counter(query)
    n_docs = len(CHUNKS)
    avg_len = sum(len(c["tokens"]) for c in CHUNKS) / max(n_docs, 1)
    doc_freq = Counter(term for chunk in CHUNKS for term in set(chunk["tokens"]))
    results = []
    for chunk in CHUNKS:
        tf = Counter(chunk["tokens"])
        score = 0.0
        matched = set()
        for term, qtf in counts.items():
            if term not in tf:
                continue
            matched.add(term)
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            length_norm = 1.2 * (1 - 0.75 + 0.75 * len(chunk["tokens"]) / max(avg_len, 1))
            score += idf * (tf[term] * 2.2) / (tf[term] + length_norm) * qtf
        coverage = len(matched) / len(set(query))
        # Require more than a stray shared word. Single-topic searches still work.
        if score > 0 and coverage >= 0.5 and (len(matched) >= 2 or len(set(query)) == 1):
            results.append({**chunk, "score": score, "coverage": coverage})
    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]


def generate(question: str, passages: list[dict]) -> tuple[str, str]:
    if not passages:
        return ("I couldn’t find enough information in the supplied policy documents to answer that.", "abstained")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (" ".join(p["text"] for p in passages[:2]), "extractive (no API key configured)")
    context = "\n".join(
        f"[{p['doc']}, line {p['line']} — {p['section']}] {p['text']}" for p in passages
    )
    prompt = (
        "Answer the employee's question using only the policy excerpts below. "
        "If the excerpts do not contain the answer, say that the policies do not specify it. "
        "Do not infer exceptions or combine facts into unsupported rules. Keep the answer concise. "
        "Cite each factual sentence with the exact bracketed source label shown in the excerpts.\n\n"
        f"POLICY EXCERPTS:\n{context}\n\nQUESTION: {question}"
    )
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful company policy assistant. Ground every claim in supplied passages and abstain when they do not answer the question."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }).encode()
    request = Request("https://api.openai.com/v1/chat/completions", data=payload, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"]["content"].strip(), f"{MODEL} + BM25"
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        # Keep the assistant usable when generation is unavailable; citations are
        # still derived only from passages that the retriever returned.
        return (" ".join(p["text"] for p in passages[:2]), f"extractive fallback (generation unavailable: {type(exc).__name__})")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "documents": 3, "passages": len(CHUNKS), "generation": "openai" if os.getenv("OPENAI_API_KEY") else "extractive"}).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send(404, b'{"error":"Not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            question = str(data.get("question", "")).strip()
            if not question or len(question) > 1000:
                raise ValueError("Enter a question between 1 and 1000 characters.")
            passages = retrieve(question)
            answer, mode = generate(question, passages)
            payload = {
                "answer": answer, "mode": mode,
                "sources": [{"doc": p["doc"], "title": p["title"], "section": p["section"], "line": p["line"], "text": p["text"]} for p in passages],
            }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    print(f"Policy Desk running at http://{HOST}:{PORT}")
    print(f"Indexed {len(CHUNKS)} policy passages. Generation: {'OpenAI' if os.getenv('OPENAI_API_KEY') else 'extractive fallback'}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
