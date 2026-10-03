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

def test_stream_recorder_raw_unit():
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = StreamRecorder(base_dir=temp_dir)
        status = recorder.get_status()
        assert status["is_recording"] is False

        # Start recording with raw format (0ms)
        start_res = recorder.start_recording(accumulation_ms=0, file_format="jpg")
        assert start_res["success"] is True
        session_dir = os.path.join(temp_dir, start_res["directory"])
        assert os.path.isdir(session_dir)

        # Feed 5 frames
        for _ in range(5):
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            recorder.add_frame(dummy_frame)
            time.sleep(0.01)

        stop_res = recorder.stop_recording()
        assert stop_res["success"] is True
        assert stop_res["frames"] == 5
        assert stop_res["raw_frames"] == 5

        files = os.listdir(session_dir)
        assert len(files) == 5
        for f in files:
            assert f.endswith(".jpg")
            assert not f.startswith("frame_")
            base = os.path.splitext(f)[0]
            assert len(base.split("_")[0]) == 8
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_stream_recorder_accumulation_unit():
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = StreamRecorder(base_dir=temp_dir)

        # Start recording with 100ms accumulation, TIFF format
        start_res = recorder.start_recording(accumulation_ms=100, file_format="tif")
        assert start_res["success"] is True
        session_dir = os.path.join(temp_dir, start_res["directory"])

        # Feed 6 frames spaced by 25ms (should form 1 or 2 accumulated frames)
        for _ in range(6):
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            recorder.add_frame(dummy_frame)
            time.sleep(0.025)

        stop_res = recorder.stop_recording()
        assert stop_res["success"] is True
        assert stop_res["raw_frames"] == 6
        assert stop_res["frames"] >= 1
        assert stop_res["accumulated_frames"] >= 1

        files = os.listdir(session_dir)
        assert len(files) >= 1
        for f in files:
            assert f.endswith(".tif")
            assert "_acc" in f
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_recording_endpoints():
    # 1. Initial status
    res = client.get("/recording/status")
    assert res.status_code == 200
    assert "is_recording" in res.json()

    # 2. Start recording with 200ms accumulation and TIFF format
    res = client.post("/recording/start", json={"accumulation_ms": 200, "format": "tif"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    session_dir = data["directory"]
    assert session_dir.startswith("session_")
    assert data["accumulation_ms"] == 200
    assert data["format"] == "tif"

    # 3. Check status during recording
    res = client.get("/recording/status")
    assert res.status_code == 200
    status = res.json()
    assert status["is_recording"] is True
    assert status["accumulation_ms"] == 200
    assert status["format"] == "tif"

    # 4. Check main status endpoint embeds recording
    res = client.get("/status")
    assert res.status_code == 200
    status_data = res.json()
    assert "recording" in status_data
    assert status_data["recording"]["is_recording"] is True

    # 5. Stop recording
    time.sleep(0.5)
    res = client.post("/recording/stop")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["directory"] == session_dir
    assert "frames" in data
    assert "accumulated_frames" in data
    assert "raw_frames" in data

    # 6. Check status after stopping
    res = client.get("/recording/status")
    assert res.status_code == 200
    assert res.json()["is_recording"] is False
