# -*- coding: utf-8 -*-
"""聽歌寫譜 - 程式進入點。

啟動本地 FastAPI 伺服器（於背景執行緒跑 uvicorn），自動開啟瀏覽器，
主執行緒等待 Ctrl+C 結束。這是 PyInstaller 打包的目標腳本。
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from app.server import create_app

DEFAULT_PORT = 8765
PORT_SCAN_RANGE = 50  # 從預設埠開始往上找，最多嘗試這麼多個埠


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def _find_free_port(start_port: int) -> int:
    for port in range(start_port, start_port + PORT_SCAN_RANGE):
        if _port_is_free(port):
            return port
    raise RuntimeError(f"找不到可用的埠（已嘗試 {start_port}-{start_port + PORT_SCAN_RANGE - 1}）")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聽歌寫譜 - 本地伺服器")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"指定埠號（預設從 {DEFAULT_PORT} 開始自動尋找可用埠）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="啟動後不要自動開啟瀏覽器",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    port = args.port if args.port is not None else _find_free_port(DEFAULT_PORT)
    if args.port is not None and not _port_is_free(port):
        print(f"埠 {port} 已被占用，改為自動尋找可用埠……")
        port = _find_free_port(DEFAULT_PORT)

    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # 等待伺服器真正就緒再開瀏覽器
    url = f"http://127.0.0.1:{port}/"
    for _ in range(100):  # 最多等 10 秒
        if server.started:
            break
        time.sleep(0.1)

    print("=" * 60)
    print("聽歌寫譜 已啟動")
    print(f"網址：{url}")
    print("請勿關閉此視窗，關閉視窗會中止伺服器。")
    print("要結束程式請按 Ctrl+C。")
    print("=" * 60)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n收到中止指令，正在關閉伺服器……")
        server.should_exit = True
        server_thread.join(timeout=5)
        print("已結束。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
