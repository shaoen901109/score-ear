# -*- coding: utf-8 -*-
"""app.server 的測試。

使用 fastapi.testclient.TestClient，並用依賴注入把 engine.pipeline.transcribe
與 notation.* 渲染函式換成假實作，讓測試不需要真正的辨識引擎/模型即可執行。
"""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402


class FakeScore:
    """模擬 notation.score_model.Score 的最小假物件。"""

    def __init__(self):
        self.title = "測試曲目"
        self.bpm = 120.0
        self.warnings = ["速度偵測信心低"]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "bpm": self.bpm,
            "time_signature": {"beats": 4, "beat_type": 4},
            "key": {"fifths": 0, "mode": "major", "tonic": "C"},
            "ticks_per_quarter": 480,
            "parts": [
                {
                    "id": "P1",
                    "name": "Piano",
                    "instrument": "piano",
                    "clef": "treble",
                    "midi_program": 0,
                    "measures": [
                        {
                            "number": 1,
                            "notes": [
                                {
                                    "start_tick": 0,
                                    "duration_tick": 480,
                                    "pitches": [60],
                                    "tie_start": False,
                                    "tie_stop": False,
                                    "voice": 1,
                                }
                            ],
                        }
                    ],
                }
            ],
            "source_file": "test.wav",
            "warnings": self.warnings,
        }


def fake_transcribe(path, options, progress):
    if progress:
        progress(0.5, "假辨識中……")
    return FakeScore()


def fake_musicxml(score) -> str:
    return "<?xml version='1.0'?><score-partwise/>"


def fake_midi(score) -> bytes:
    return b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"


def fake_jianpu_svg(score, part_index=None):
    suffix = f"-part{part_index}" if part_index is not None else "-all"
    return f"<svg xmlns='http://www.w3.org/2000/svg'><text>jianpu{suffix}</text></svg>"


def fake_staff_svg(score):
    return [
        "<svg xmlns='http://www.w3.org/2000/svg'><text>staff-page-1</text></svg>",
    ]


@pytest.fixture()
def client():
    app = create_app(
        transcribe_fn=fake_transcribe,
        renderers={
            "musicxml": fake_musicxml,
            "midi": fake_midi,
            "jianpu_svg": fake_jianpu_svg,
            "staff_svg": fake_staff_svg,
        },
    )
    return TestClient(app)


def _make_options() -> str:
    return json.dumps(
        {
            "mode": "solo",
            "instrument_hint": "auto",
            "bpm_override": None,
            "time_signature": "auto",
            "quantize": "1/16",
            "use_gpu": False,
        }
    )


def _upload_job(client: "TestClient", filename="test.wav", content=b"fake-audio-bytes"):
    files = {"file": (filename, io.BytesIO(content), "audio/wav")}
    data = {"options": _make_options()}
    return client.post("/api/jobs", files=files, data=data)


def _wait_for_done(client: "TestClient", job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in ("done", "error"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"任務未在 {timeout} 秒內完成，最後狀態：{last}")


def test_upload_and_full_flow(client):
    resp = _upload_job(client)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    status = _wait_for_done(client, job_id)
    assert status["status"] == "done"
    assert status["progress"] == 1.0

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    result = result_resp.json()
    assert result["score"]["title"] == "測試曲目"
    assert "jianpu" in result["jianpu_svg"]
    assert len(result["staff_svg"]) == 1


def test_result_with_part_query(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    result_resp = client.get(f"/api/jobs/{job_id}/result", params={"part": 0})
    assert result_resp.status_code == 200
    result = result_resp.json()
    assert "part0" in result["jianpu_svg"]


def test_download_musicxml(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    dl = client.get(f"/api/jobs/{job_id}/download/musicxml")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/vnd.recordare.musicxml+xml")
    assert b"score-partwise" in dl.content


def test_download_midi(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    dl = client.get(f"/api/jobs/{job_id}/download/midi")
    assert dl.status_code == 200
    assert dl.content.startswith(b"MThd")


def test_download_jianpu_svg(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    dl = client.get(f"/api/jobs/{job_id}/download/jianpu.svg")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in dl.content


def test_download_staff_svg_zip(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    dl = client.get(f"/api/jobs/{job_id}/download/staff.svg")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/zip")

    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert b"staff-page-1" in zf.read(names[0])


def test_download_json(client):
    resp = _upload_job(client)
    job_id = resp.json()["job_id"]
    _wait_for_done(client, job_id)

    dl = client.get(f"/api/jobs/{job_id}/download/json")
    assert dl.status_code == 200
    body = json.loads(dl.content)
    assert body["title"] == "測試曲目"


def test_invalid_extension_rejected(client):
    resp = _upload_job(client, filename="test.exe")
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert isinstance(body["error"], str)


def test_job_not_found(client):
    resp = client.get("/api/jobs/不存在的任務id")
    assert resp.status_code == 404


def test_result_before_done_returns_400(client):
    # 用一個會卡住的假 transcribe 模擬「還沒完成」的狀態
    import threading

    gate = threading.Event()

    def slow_transcribe(path, options, progress):
        gate.wait(timeout=5)
        return FakeScore()

    app = create_app(
        transcribe_fn=slow_transcribe,
        renderers={
            "musicxml": fake_musicxml,
            "midi": fake_midi,
            "jianpu_svg": fake_jianpu_svg,
            "staff_svg": fake_staff_svg,
        },
    )
    local_client = TestClient(app)
    try:
        resp = _upload_job(local_client)
        job_id = resp.json()["job_id"]
        result_resp = local_client.get(f"/api/jobs/{job_id}/result")
        assert result_resp.status_code == 400
        assert "error" in result_resp.json()
    finally:
        gate.set()
