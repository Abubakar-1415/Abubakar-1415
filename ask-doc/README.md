# Ask Doc

Ask Doc is a private document question-answering web app. It supports PDF,
Word, and Excel uploads and stores indexed vectors in Qdrant.

## Deploy on Render

The repository-level `render.yaml` defines Ask Doc as a separate Render web
service rooted at this folder. Connect the repository to Render as a Blueprint,
then provide the secret values requested during setup:

- `ADMIN_USERNAME` and `ADMIN_PASSWORD`
- `NVIDIA_API_KEY`
- `QDRANT_URL` and `QDRANT_API_KEY`

The NVIDIA key is read only from the environment; no API key is stored in this
folder. Rotate any key that was previously embedded in a local source file.

Optional email notifications use `OWNER_EMAIL`, `SMTP_HOST`, `SMTP_USERNAME`,
and `SMTP_PASSWORD`.

The service uses a persistent Render disk for the SQLite database. Qdrant
stores vectors in a separate collection.

## Run locally

Install `requirements.txt`, configure the same environment variables, and run:

```powershell
python rag_app.py
```

The app listens on `http://127.0.0.1:8000` by default.
