# Ask Doc

Ask Doc lets visitors upload multiple PDF, DOCX, and XLSX files, index them in
Qdrant, and ask questions grounded in the uploaded content. The app is public:
anyone with its URL can view, search, upload, and delete documents in the shared
library.

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
CLOUD_USER_EMAIL = "you@example.com"
```

`CLOUD_USER_EMAIL` is used only to retain the existing Qdrant document namespace;
it does not enable or require Google sign-in. All visitors share that namespace.
Remove any old `[auth]` OAuth configuration from app secrets.

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

Configure the same values in `.streamlit/secrets.toml` for local development;
never commit that file.
