import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_basic_chat():
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
