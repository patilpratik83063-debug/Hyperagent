"""Pytest bootstrap: make the backend package importable from anywhere and
force OFFLINE mode (in-memory store + mock providers) so the suite never
touches real Supabase/OpenAI/Serper — even when backend/.env has live
credentials."""
import os
import sys
from pathlib import Path

os.environ["MOCK_MODE"] = "1"

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
