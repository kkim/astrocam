import os
import shutil
import tempfile
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app
from recorder import StreamRecorder

client = TestClient(app)

def test_stream_recorder_unit():
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = StreamRecorder(base_dir=temp_dir)
        status = recorder.get_status()
        assert status["is_recording"] is False

        # Start recording
        start_res = recorder.start_recording()
        assert start_res["success"] is True
        session_dir = os.path.join(temp_dir, start_res["directory"])
        assert os.path.isdir(session_dir)

        # Feed some frames
        for _ in range(5):
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            recorder.add_frame(dummy_frame)
            time.sleep(0.02)

        # Stop recording
        stop_res = recorder.stop_recording()
        assert stop_res["success"] is True
        assert stop_res["frames"] == 5
        assert stop_res["bytes_stored"] > 0
        assert stop_res["mb_stored"] >= 0.0

        # Verify files on disk
        files = os.listdir(session_dir)
        assert len(files) == 5
        assert "frame_000001.jpg" in files
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_recording_endpoints():
    # 1. Initial status
    res = client.get("/recording/status")
    assert res.status_code == 200
    assert "is_recording" in res.json()

    # 2. Start recording
    res = client.post("/recording/start")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    session_dir = data["directory"]
    assert session_dir.startswith("session_")

    # 3. Check status during recording
    res = client.get("/recording/status")
    assert res.status_code == 200
    assert res.json()["is_recording"] is True

    # 4. Check main status endpoint embeds recording
    res = client.get("/status")
    assert res.status_code == 200
    status_data = res.json()
    assert "recording" in status_data
    assert status_data["recording"]["is_recording"] is True

    # 5. Stop recording
    time.sleep(0.3)
    res = client.post("/recording/stop")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["directory"] == session_dir
    assert "frames" in data
    assert "mb_stored" in data

    # 6. Check status after stopping
    res = client.get("/recording/status")
    assert res.status_code == 200
    assert res.json()["is_recording"] is False
