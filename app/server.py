# -*- coding: utf-8 -*-
"""聽歌寫譜 - FastAPI 伺服器。

負責：接收上傳音檔、建立任務、以單一 worker 執行緒依序處理、快取結果、
提供下載端點（MusicXML / MIDI / 簡譜 SVG / 五線譜 SVG / JSON）。

依賴注入：`create_app(transcribe_fn=None, renderers=None)`
    - transcribe_fn(path, options, progress) -> Score
      預設為 None 時，延遲 import `engine.pipeline.transcribe`（避免在測試環境、
      或 engine 尚未就緒時，import 這個模組就失敗）。
    - renderers 是一個 dict，可覆寫以下鍵（皆為函式），預設延遲 import 對應的
      `notation.*` 實作：
        "musicxml": score_to_musicxml(score) -> str
        "midi": score_to_midi_bytes(score) -> bytes
        "jianpu_svg": score_to_jianpu_svg(score, part_index=None) -> str
        "staff_svg": score_to_staff_svg(score) -> list[str]

任務狀態只存在記憶體（dict），伺服器重啟即消失，符合單機工具定位。
"""

from __future__ import annotations

import io
import json
import queue
import shutil
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles

from notation.score_model import TranscribeOptions

# ---------------------------------------------------------------------------
# 路徑處理
# ---------------------------------------------------------------------------

def _app_root() -> Path:
    """回傳應用程式根目錄。

    - 一般開發模式：本檔案所在目錄的上一層（專案根目錄）。
    - PyInstaller frozen 模式：exe 所在目錄。

    若日後 `engine.paths.app_root()` 就緒，伺服器啟動時會優先使用該函式
    （見 `create_app` 內的 try/except），此函式僅作為尚未就緒時的備援邏輯，
    邏輯需與該模組保持一致。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _runtime_uploads_dir(root: Path) -> Path:
    d = root / "runtime" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".mp4", ".aac"}

DOWNLOAD_MEDIA_TYPES = {
    "musicxml": ("application/vnd.recordare.musicxml+xml", ".musicxml"),
    "midi": ("audio/midi", ".mid"),
    "jianpu.svg": ("image/svg+xml", ".svg"),
    "staff.svg": ("application/zip", ".zip"),  # 多頁時打包 zip；單頁也統一用 zip
    "json": ("application/json", ".json"),
}


def _content_disposition(ascii_name: str, pretty_name: str) -> str:
    """組出可安全放進 HTTP 標頭的 Content-Disposition 值。

    HTTP 標頭只能是 latin-1，中文檔名（例如「聽歌寫譜結果」）不能直接放進
    `filename="..."`——Starlette 組 header 時會丟
    `UnicodeEncodeError: 'latin-1' codec can't encode characters`。
    改用 RFC 6266 / RFC 5987：`filename=` 帶一個純 ASCII 備援名稱給不支援
    `filename*` 的用戶端，`filename*=UTF-8''...` 帶 percent-encode 過的中文
    檔名給支援的瀏覽器顯示。
    """
    quoted = urllib.parse.quote(pretty_name)
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quoted}'


# ---------------------------------------------------------------------------
# 任務資料結構
# ---------------------------------------------------------------------------

class Job:
    __slots__ = (
        "id",
        "status",
        "progress",
        "message",
        "error",
        "options",
        "audio_path",
        "result",
        "created_at",
    )

    def __init__(self, job_id: str, audio_path: Path, options: dict):
        self.id = job_id
        self.status = "queued"  # queued|running|done|error
        self.progress = 0.0
        self.message = "已加入佇列"
        self.error: Optional[str] = None
        self.options = options
        self.audio_path = audio_path
        self.result: Optional[dict] = None  # {"score":..., "staff_svg":[...], "jianpu_svg":...}
        self.created_at = time.time()

    def to_status_dict(self) -> dict:
        d = {
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
        }
        if self.error:
            d["error"] = self.error
        return d


def _score_to_dict(score: Any) -> dict:
    """把 Score（dataclass 或已是 dict）轉成可 JSON 序列化的 dict。"""
    if score is None:
        return {}
    if isinstance(score, dict):
        return score
    if hasattr(score, "to_dict"):
        return score.to_dict()
    if is_dataclass(score):
        return asdict(score)
    raise TypeError(f"無法將 {type(score)!r} 轉為 dict")


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

def create_app(
    transcribe_fn: Optional[Callable[..., Any]] = None,
    renderers: Optional[dict] = None,
) -> FastAPI:
    """建立 FastAPI app。

    transcribe_fn: 簽章同 engine.pipeline.transcribe(path, options, progress) -> Score
    renderers: dict，可覆寫 "musicxml"/"midi"/"jianpu_svg"/"staff_svg" 四個渲染函式。
    """

    renderers = dict(renderers or {})

    # -- 延遲 import 真實實作（只在真正需要、且未被注入時才 import） --------

    def _real_transcribe():
        from engine.pipeline import transcribe as real_fn  # noqa: WPS433

        return real_fn

    def _real_musicxml():
        from notation.musicxml import score_to_musicxml  # noqa: WPS433

        return score_to_musicxml

    def _real_midi():
        from notation.midi import score_to_midi_bytes  # noqa: WPS433

        return score_to_midi_bytes

    def _real_jianpu():
        from notation.jianpu import score_to_jianpu_svg  # noqa: WPS433

        return score_to_jianpu_svg

    def _real_staff():
        from notation.staff import score_to_staff_svg  # noqa: WPS433

        return score_to_staff_svg

    def get_transcribe_fn():
        return transcribe_fn or _real_transcribe()

    def get_renderer(name: str, fallback_loader: Callable[[], Callable]) -> Callable:
        fn = renderers.get(name)
        if fn is not None:
            return fn
        return fallback_loader()

    # -- app_root：優先使用 engine.paths.app_root()（若已就緒） --------------

    try:
        from engine.paths import app_root as _engine_app_root  # noqa: WPS433

        root = _engine_app_root()
    except Exception:
        root = _app_root()

    uploads_dir = _runtime_uploads_dir(root)
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="聽歌寫譜")

    # -- 任務狀態（記憶體）與 worker 執行緒 -----------------------------------

    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    task_queue: "queue.Queue[str]" = queue.Queue()

    def _worker_loop():
        while True:
            job_id = task_queue.get()
            if job_id is None:  # 用於測試時關閉 worker（一般不會用到）
                break
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                continue
            _run_job(job)

    def _run_job(job: Job):
        with jobs_lock:
            job.status = "running"
            job.progress = 0.0
            job.message = "開始處理"

        def progress_cb(fraction: float, message: str):
            with jobs_lock:
                job.progress = max(0.0, min(1.0, float(fraction)))
                job.message = message

        try:
            fn = get_transcribe_fn()
            # job.options 是從前端上傳的 JSON 解析出來的純 dict；
            # engine.pipeline.transcribe 用 getattr(options, "欄位", 預設值) 讀取選項，
            # 對 dict 做 getattr 永遠拿不到對應欄位（dict 沒有這些屬性），會靜默地
            # 全部退回預設值，等於忽略使用者所有選項。這裡統一轉成 TranscribeOptions
            # 物件再傳給 transcribe_fn，維持 docs/ARCHITECTURE.md 的對外函式契約。
            options_obj = (
                TranscribeOptions.from_dict(job.options)
                if isinstance(job.options, dict)
                else job.options
            )
            score = fn(str(job.audio_path), options_obj, progress_cb)

            score_dict = _score_to_dict(score)

            jianpu_fn = get_renderer("jianpu_svg", _real_jianpu)
            staff_fn = get_renderer("staff_svg", _real_staff)

            jianpu_svg = jianpu_fn(score)
            staff_svg_list = staff_fn(score)

            with jobs_lock:
                job.result = {
                    "score": score_dict,
                    "score_obj": score,  # 供下載端點重用（不進 JSON 回應）
                    "staff_svg": staff_svg_list,
                    "jianpu_svg": jianpu_svg,
                }
                job.status = "done"
                job.progress = 1.0
                job.message = "完成"
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            with jobs_lock:
                job.status = "error"
                job.error = f"處理失敗：{exc}"
                job.message = job.error

    worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    worker_thread.start()

    # -- 工具函式 -------------------------------------------------------------

    def _error(message: str, status_code: int = 400) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"error": message})

    def _get_job_or_404(job_id: str) -> Job:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到任務")
        return job

    # -- API 端點 --------------------------------------------------------------

    @app.post("/api/jobs")
    async def create_job(file: UploadFile = File(...), options: str = Form("{}")):
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            return _error(
                f"不支援的檔案格式：{ext or '（無副檔名）'}，"
                f"允許格式：{', '.join(sorted(ALLOWED_EXTS))}",
                400,
            )

        try:
            opts = json.loads(options) if options else {}
            if not isinstance(opts, dict):
                raise ValueError("options 必須是 JSON 物件")
        except Exception:
            return _error("options 參數不是合法的 JSON", 400)

        job_id = uuid.uuid4().hex
        job_dir = uploads_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest_path = job_dir / (Path(filename).name or "upload")

        size = 0
        try:
            with dest_path.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        out.close()
                        shutil.rmtree(job_dir, ignore_errors=True)
                        return _error("檔案超過 200MB 上限", 400)
                    out.write(chunk)
        finally:
            await file.close()

        if size == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            return _error("上傳檔案是空的", 400)

        job = Job(job_id, dest_path, opts)
        with jobs_lock:
            jobs[job_id] = job
        task_queue.put(job_id)

        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str):
        job = _get_job_or_404(job_id)
        return job.to_status_dict()

    @app.get("/api/jobs/{job_id}/result")
    async def get_job_result(job_id: str, part: Optional[int] = None):
        job = _get_job_or_404(job_id)
        if job.status != "done" or job.result is None:
            return _error("任務尚未完成", 400)

        result = job.result
        jianpu_svg = result["jianpu_svg"]
        if part is not None:
            jianpu_fn = get_renderer("jianpu_svg", _real_jianpu)
            try:
                jianpu_svg = jianpu_fn(result["score_obj"], part)
            except TypeError:
                # 舊版渲染器可能不支援 part_index 關鍵字，退回全體結果
                jianpu_svg = result["jianpu_svg"]

        return {
            "score": result["score"],
            "staff_svg": result["staff_svg"],
            "jianpu_svg": jianpu_svg,
        }

    @app.get("/api/jobs/{job_id}/download/{fmt:path}")
    async def download_job(job_id: str, fmt: str):
        job = _get_job_or_404(job_id)
        if job.status != "done" or job.result is None:
            return _error("任務尚未完成", 400)

        if fmt not in DOWNLOAD_MEDIA_TYPES:
            return _error(f"不支援的下載格式：{fmt}", 400)

        media_type, out_ext = DOWNLOAD_MEDIA_TYPES[fmt]
        result = job.result
        score_obj = result["score_obj"]
        filename_base = "聽歌寫譜結果"

        if fmt == "json":
            content = json.dumps(result["score"], ensure_ascii=False, indent=2)
            return Response(
                content=content.encode("utf-8"),
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition(
                        f"score{out_ext}", f"{filename_base}{out_ext}"
                    )
                },
            )

        if fmt == "musicxml":
            musicxml_fn = get_renderer("musicxml", _real_musicxml)
            xml_str = musicxml_fn(score_obj)
            return Response(
                content=xml_str.encode("utf-8"),
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition(
                        f"score{out_ext}", f"{filename_base}{out_ext}"
                    )
                },
            )

        if fmt == "midi":
            midi_fn = get_renderer("midi", _real_midi)
            midi_bytes = midi_fn(score_obj)
            return Response(
                content=midi_bytes,
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition(
                        f"score{out_ext}", f"{filename_base}{out_ext}"
                    )
                },
            )

        if fmt == "jianpu.svg":
            svg_str = result["jianpu_svg"]
            return Response(
                content=svg_str.encode("utf-8"),
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition(
                        f"score_jianpu{out_ext}", f"{filename_base}_簡譜{out_ext}"
                    )
                },
            )

        if fmt == "staff.svg":
            svg_list = result["staff_svg"]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, svg in enumerate(svg_list, start=1):
                    zf.writestr(f"staff_{i:02d}.svg", svg.encode("utf-8"))
            buf.seek(0)
            return Response(
                content=buf.getvalue(),
                media_type=media_type,
                headers={
                    "Content-Disposition": _content_disposition(
                        "score_staff.zip", f"{filename_base}_五線譜.zip"
                    )
                },
            )

        return _error(f"不支援的下載格式：{fmt}", 400)

    # -- 靜態頁面 --------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = static_dir / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>找不到 static/index.html</h1>", status_code=500)
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -- 統一例外處理（確保所有錯誤都回傳 {"error": 中文訊息} 格式） -------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):  # noqa: ANN001
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc):  # noqa: ANN001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"伺服器內部錯誤：{exc}"})

    return app
