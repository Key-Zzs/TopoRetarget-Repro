"""Minimal Chrome DevTools driver for OakInk2 real-input certification.

This intentionally uses only the Python standard library. Mouse commands go
through Chrome's ``Input.dispatchMouseEvent`` domain, so the viewer receives
real browser pointer callbacks rather than direct calls to camera functions.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class _WebSocket:
    def __init__(self, url: str) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "ws" or parsed.hostname is None or parsed.port is None:
            raise RuntimeError(f"CDP_WEBSOCKET_URL_INVALID:{url}")
        self.socket = socket.create_connection((parsed.hostname, parsed.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            f"Origin: http://{parsed.hostname}:{parsed.port}\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.socket.recv(4096)
        if not response.startswith(b"HTTP/1.1 101"):
            raise RuntimeError(f"CDP_WEBSOCKET_HANDSHAKE_FAILED:{response[:200]!r}")

    def close(self) -> None:
        self.socket.close()

    def send_json(self, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = bytes((0x81, 0x80 | length))
        elif length < 65536:
            header = bytes((0x81, 0xFE)) + struct.pack("!H", length)
        else:
            header = bytes((0x81, 0xFF)) + struct.pack("!Q", length)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(header + mask + masked)

    def recv_json(self) -> dict[str, Any]:
        fragments = bytearray()
        while True:
            first, second = self._read_exact(2)
            opcode = first & 0x0F
            final = bool(first & 0x80)
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if masked:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 0x8:
                raise RuntimeError("CDP_WEBSOCKET_CLOSED")
            if opcode == 0x9:
                self._send_control(0xA, payload)
                continue
            if opcode in (0x0, 0x1):
                fragments.extend(payload)
                if final:
                    return json.loads(fragments.decode("utf-8"))

    def _send_control(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(bytes((0x80 | opcode, 0x80 | len(payload))) + mask + masked)

    def _read_exact(self, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            block = self.socket.recv(count - len(data))
            if not block:
                raise RuntimeError("CDP_WEBSOCKET_EOF")
            data.extend(block)
        return bytes(data)


class ChromeCDP:
    """One isolated headless Chrome page controlled through CDP."""

    def __init__(self, executable: str, *, width: int = 640, height: int = 640) -> None:
        self.width = width
        self.height = height
        self._profile = tempfile.TemporaryDirectory(prefix="oakink2_o1r2d_chrome_")
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
        probe.close()
        self.process = subprocess.Popen(
            [
                executable,
                "--headless=new",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--metrics-recording-only",
                "--enable-unsafe-swiftshader",
                "--use-angle=swiftshader",
                "--use-gl=angle",
                "--hide-scrollbars",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={self._profile.name}",
                f"--window-size={width},{height}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        page: dict[str, Any] | None = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"CHROME_EXITED_EARLY:{self.process.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=1
                ) as response:
                    pages = json.load(response)
                page = next(item for item in pages if item.get("type") == "page")
                break
            except (OSError, StopIteration, json.JSONDecodeError):
                time.sleep(0.05)
        if page is None:
            self.close()
            raise RuntimeError("CHROME_CDP_START_TIMEOUT")
        try:
            self.websocket = _WebSocket(str(page["webSocketDebuggerUrl"]))
        except Exception:
            self.close()
            raise
        self._next_id = 1
        self.command("Page.enable")
        self.command("Runtime.enable")
        self.command(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
        )

    def __enter__(self) -> ChromeCDP:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        websocket = getattr(self, "websocket", None)
        if websocket is not None:
            websocket.close()
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        profile = getattr(self, "_profile", None)
        if profile is not None:
            profile.cleanup()

    def command(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        identifier = self._next_id
        self._next_id += 1
        self.websocket.send_json({"id": identifier, "method": method, "params": params or {}})
        while True:
            message = self.websocket.recv_json()
            if message.get("id") != identifier:
                continue
            if "error" in message:
                raise RuntimeError(f"CDP_COMMAND_FAILED:{method}:{message['error']}")
            return dict(message.get("result", {}))

    def navigate(self, url: str) -> None:
        self.command("Page.navigate", {"url": url})
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if self.evaluate("document.readyState") == "complete" and self.evaluate(
                    "Boolean(window.__OAKINK2_BROWSER_CERTIFICATE__)"
                ):
                    return
            except RuntimeError:
                pass
            time.sleep(0.05)
        raise RuntimeError("CHROME_PAGE_READY_TIMEOUT")

    def evaluate(self, expression: str) -> Any:
        result = self.command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        ).get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(f"CDP_EVALUATE_FAILED:{result['exceptionDetails']}")
        if result.get("subtype") == "error":
            raise RuntimeError(f"CDP_EVALUATE_FAILED:{result.get('description')}")
        return result.get("value")

    def certificate(self) -> dict[str, Any]:
        value = self.evaluate("window.__OAKINK2_BROWSER_CERTIFICATE__")
        if not isinstance(value, dict):
            raise RuntimeError("OAKINK2_BROWSER_CERTIFICATE_MISSING")
        return value

    def mouse_drag(
        self, start: tuple[float, float], end: tuple[float, float], *, steps: int = 2
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        self.command("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x0, "y": y0})
        self.command(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed",
                "x": x0,
                "y": y0,
                "button": "left",
                "buttons": 1,
                "clickCount": 1,
            },
        )
        for step in range(1, steps + 1):
            alpha = step / steps
            self.command(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": x0 + (x1 - x0) * alpha,
                    "y": y0 + (y1 - y0) * alpha,
                    "button": "left",
                    "buttons": 1,
                },
            )
        self.command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseReleased",
                "x": x1,
                "y": y1,
                "button": "left",
                "buttons": 0,
                "clickCount": 1,
            },
        )

    def wheel(self, delta_y: float) -> None:
        self.command(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseWheel",
                "x": self.width / 2,
                "y": self.height / 2,
                "deltaX": 0,
                "deltaY": delta_y,
            },
        )

    def screenshot(self, destination: Path) -> None:
        result = self.command(
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
                "clip": {"x": 0, "y": 0, "width": self.width, "height": self.height, "scale": 1},
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(str(result["data"])))
