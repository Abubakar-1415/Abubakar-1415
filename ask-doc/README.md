# Ask Doc

Ask Doc lets approved Google accounts upload multiple PDF, DOCX, and XLSX files,
index them in Qdrant, and ask questions grounded in the uploaded content.

## Deploy on Streamlit Community Cloud

Create an app from this repository's `main` branch and select
`ask-doc/streamlit_app.py` as the entry point. In the app's **Settings → Secrets**,
configure the values below. Keep these values out of GitHub.

```toml
NVIDIA_API_KEY = "your-nvidia-api-key"
QDRANT_URL = "https://your-cluster.example.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"
NVIDIA_CHAT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
NVIDIA_EMBEDDING_DIMENSION = "2048"
NVIDIA_QDRANT_COLLECTION = "ask_doc_documents_nemotron_3_embed_1b_2048"
ALLOWED_EMAILS = ["you@example.com"]

[auth]
redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
cookie_secret = "generate-a-long-random-secret"
client_id = "your-google-oauth-client-id"
client_secret = "your-google-oauth-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Create a Google OAuth client and register the exact `redirect_uri` for the
Streamlit app. Set `ALLOWED_EMAILS` to the Google accounts allowed to use the
app. Each account receives a private Qdrant namespace. Active login sessions
automatically sign out after four hours.

If an existing deployment still sets `NVIDIA_EMBEDDING_MODEL` to either retired
embedding model, the app maps it to the currently available
`nvidia/nemotron-3-embed-1b` model and uses a separate Qdrant collection.

All API credentials are read from Streamlit secrets. Rotate any key that was
previously embedded in a local source file. Streamlit Community Cloud does not
provide a persistent local disk; document vectors persist in Qdrant.

## Run locally

Install `requirements.txt` and run:

```powershell
streamlit run streamlit_app.py
```

For local sign-in and provider access, configure the same values in
`.streamlit/secrets.toml`; never commit that file.
