"""Environment-based credentials for Ask Doc."""
import os


def get_nvidia_api_key() -> str:
    """Return the NVIDIA API key configured in the environment."""
    return os.environ.get("NVIDIA_API_KEY", "").strip()
