import pytest
from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)

def test_panorama_endpoints():
    # 1. Test status initially
    response = client.get("/panorama/status")
    assert response.status_code == 200
    assert response.json()["active"] is False

    # 2. Test start
    response = client.post("/panorama/start", json={"frames": 5, "drift_step": 10.0, "auto_align": True})
    assert response.status_code == 200
    assert response.json()["success"] is True

    # 3. Test status while active
    response = client.get("/panorama/status")
    assert response.status_code == 200
    assert response.json()["active"] is True

    # 4. Test stop
    response = client.post("/panorama/stop")
    assert response.status_code == 200
    assert response.json()["success"] is True

    # 5. Test status after stop
    # Small sleep to allow thread to handle stop
    time.sleep(0.5)
    response = client.get("/panorama/status")
    assert response.status_code == 200
    assert response.json()["active"] is False
    print("Panorama endpoints API test passed!")
