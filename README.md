# AI StudyMate RAG — Custom Theme

Streamlit starter app with an app-controlled Light/Dark toggle, chat history for the current session, PDF upload, semantic retrieval, and Gemini answers.

## Run on Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`, then run:

```powershell
python -m streamlit run app.py
```

The first run may take longer because the sentence-transformer model downloads.

## Security
Do not commit `.env` or a real API key. If a key was previously exposed, revoke it.

## Limitations
Chat history and indexed PDF data are kept in Streamlit session state and are not persisted to a database. Scanned PDFs may need OCR.
