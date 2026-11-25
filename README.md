# Policy Desk — RAG application

This is a small, working retrieval-augmented generation (RAG) app for the supplied company policies.

## Run it

Requires Python 3.10 or newer. From this folder:

```powershell
python server.py
```

Then open <http://127.0.0.1:8000>.

The app splits each policy into statement-sized passages, ranks passages with BM25, and passes the best matches as context to the answer generator. It always returns the retrieved source document, section, and line.

### Enable generated answers

Without configuration, the app uses a clearly labelled extractive answer so it works without an API key. To enable grounded generated answers, set `OPENAI_API_KEY` before starting the server. Optionally set `OPENAI_MODEL` (defaults to `gpt-4.1-mini`). The key stays in the server environment and is never sent to the browser.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "your-key"
python server.py
```

Policy source files are in `policies/`. Restart the server after editing them; the passages are indexed at startup.
