"""Configuration and settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LETTERS_DIR = DATA_DIR / "letters"
SIGNALS_DIR = DATA_DIR / "signals"
STOCK_DIR = DATA_DIR / "stock"

# Ensure directories exist
LETTERS_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
STOCK_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str | None:
    """Get Anthropic API key from environment or Streamlit secrets."""
    # First try environment variable
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key

    # Then try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    return None


# API
ANTHROPIC_API_KEY = get_api_key()

# Model settings
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# Stock settings
TICKER = "BRK-B"
RETURN_WINDOWS = [30, 60, 90]  # Days after letter release
