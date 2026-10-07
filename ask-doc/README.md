# Ask Doc

Ask Doc lets people create an account with a username and password, upload
multiple PDF, DOCX, and XLSX files, and ask questions grounded in their own
private document library. Search combines semantic retrieval with an exact
keyword scan across every indexed passage in that user's library. Exact matches
show the matching keyword with surrounding text from each source document.
Answers are shown in readable text with expandable citations; structured JSON
remains available on demand.

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
MAX_UPLOAD_MB = "100"
```

No separate database is needed: account records are stored in the
`ask_doc_users_v1` Qdrant collection, alongside but separately from document
vectors. Account records have unique case-insensitive usernames; passwords are
stored as salted scrypt hashes, never as plaintext. Remove the old Google OAuth
settings. Each account receives a separate Qdrant namespace, and sessions
expire after four hours.

The previous shared-library documents are preserved in their old Qdrant
namespace and are not visible to newly registered accounts. They have no
recorded owner and are not automatically assigned to the first registrant.

Each uploaded file may be up to 100 MB. The Streamlit app also sets
`server.maxUploadSize` to 100 MB.

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
never commit that file. Answers are returned with `answer`, `citations`,
`match_percent`, `not_found`, and `match_explanation` fields.
