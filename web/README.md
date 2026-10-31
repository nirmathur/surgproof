# SurgProof web UI

FastAPI backend (`web/server.py`) + React/Vite/Tailwind frontend (`web/ui/`), built into `web/dist/` (committed, so Node is not needed to run it).

```bash
pip install -r web/requirements.txt
python -m uvicorn web.server:app --port 8600     # open http://localhost:8600
```

Rebuild the frontend after editing `web/ui/src/` (Node 20+):
```bash
cd web/ui && npm ci && npm run build            # writes web/dist
npm run dev                                       # hot reload on :5173, proxies /api to :8600
```

Reads only real pipeline files (`out/` first, then `coord/artifacts/`), see `DATA_CONTRACT.md`.
Missing files show as "waiting for ..."; no number is ever filled in. `app.py` (Streamlit) stays as the fallback.
