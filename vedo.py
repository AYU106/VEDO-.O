"""A simple local agent that can check for updates and ask for user permission before upgrading itself.

This is NOT a fully autonomous self-updating agent. By default it only updates when you:
  1) run the `check_update` command
  2) grant permission when prompted

It can be run as a small HTTP API server so you can control it from another device on the same network.

Usage (CLI):
  python vedo.py

Usage (HTTP server):
  python vedo.py --serve

Commands (CLI):
  help           - show help
  version        - show current version
  check_update   - check remote version and ask to upgrade
  exit           - quit

Config (optional):
  Create a `config.json` next to this script to customize behavior.

  Example config.json:
  {
    "update_version_url": "https://raw.githubusercontent.com/<you>/<repo>/main/vedo_version.txt",
    "update_script_url": "https://raw.githubusercontent.com/<you>/<repo>/main/vedo.py",
    "auto_update": false,
    "serve": {
      "enabled": true,
      "host": "0.0.0.0",
      "port": 8000,
      "require_token": false,
      "token": "your-secret-token"
    }
  }

Important: Running the server and allowing auto-update makes the script able to modify itself remotely.
Only do this with trusted hosts and sources.
"""

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# --- Defaults ---------------------------------------------------------------
DEFAULT_CONFIG = {
    "update_version_url": "https://example.com/vedo_version.txt",
    "update_script_url": "https://example.com/vedo.py",
    "auto_update": False,
    "gemini_api_key": "",
    "gemini_model": "text-bison-001",
    "serve": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8000,
        "require_token": False,
        "token": "",
    },
}

# A simple semantic version string.
__version__ = "0.0.1"

# Runtime state -------------------------------------------------------------
_last_update_check: Optional[str] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json if present."""
    config_path = Path(__file__).resolve().parent / "config.json"
    config: Dict[str, Any] = DEFAULT_CONFIG.copy()

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # Merge top-level keys
            config.update(user_cfg)
            # Merge 'serve' subkeys if provided
            if isinstance(user_cfg.get("serve"), dict):
                config["serve"].update(user_cfg["serve"])
        except Exception as e:
            print(f"[config] failed to load config.json: {e}")

    return config


def prompt_yes_no(prompt: str) -> bool:
    """Prompt the user with a yes/no question. Returns True for yes, False for no."""
    while True:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no", ""}:
            return False
        print("Please answer 'y' or 'n'.")


def fetch_text(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch text content from a URL. Returns None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8").strip()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"[update] failed to fetch {url}: {e}")
        return None


def call_gemini_api(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Ask Gemini (Generative Language API) for a completion."""

    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate?key={api_key}"
    payload = {"prompt": {"text": prompt}}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except Exception as e:
        print(f"[gemini] request failed: {e}")
        return None

    candidates = body.get("candidates") or []
    if candidates and isinstance(candidates, list):
        first = candidates[0]
        content = first.get("content") if isinstance(first, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]

    return body.get("output") or body.get("text")


def check_for_update(update_version_url: str, current_version: str) -> Tuple[bool, Optional[str]]:
    """Check the remote version and return (is_update_available, remote_version)."""
    remote_version = fetch_text(update_version_url)
    if remote_version is None:
        return False, None

    if remote_version.strip() == current_version.strip():
        return False, remote_version

    # Simple semantic compare: assumes versions are dot-separated ints
    try:
        local_parts = [int(p) for p in current_version.split(".") if p.isdigit()]
        remote_parts = [int(p) for p in remote_version.split(".") if p.isdigit()]
        if remote_parts > local_parts:
            return True, remote_version
    except Exception:
        # Fallback: treat any different version as an update
        return True, remote_version

    return False, remote_version


def download_update(update_script_url: str) -> bool:
    """Download the updated script and replace this file. Returns True on success."""
    print("[update] downloading updated script...")
    content = fetch_text(update_script_url)
    if content is None:
        return False

    current_path = os.path.abspath(__file__)
    backup_path = current_path + ".backup"

    try:
        # Backup current script first
        with open(current_path, "r", encoding="utf-8") as f:
            old = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(old)

        # Write new content
        with open(current_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[update] updated successfully. Backup saved to {backup_path}")
        return True
    except Exception as e:
        print(f"[update] failed to write update: {e}")
        # Attempt to restore from backup
        if os.path.exists(backup_path):
            try:
                with open(backup_path, "r", encoding="utf-8") as f:
                    restored = f.read()
                with open(current_path, "w", encoding="utf-8") as f:
                    f.write(restored)
                print("[update] restored the original script from backup.")
            except Exception:
                print("[update] could not restore the original script. Manual fix required.")
        return False


def run_update_flow(update_version_url: str, update_script_url: str, auto_confirm: bool = False) -> bool:
    """Check for updates and apply them. Returns True if an update was applied."""
    global _last_update_check

    print("Checking for updates...")
    available, remote_version = check_for_update(update_version_url, __version__)
    _last_update_check = remote_version

    if remote_version is None:
        print("Could not determine remote version. Skipping update.")
        return False

    if not available:
        print(f"You are already on the latest version ({__version__}).")
        return False

    print(f"Update available: {remote_version} (current: {__version__})")

    if not auto_confirm and not prompt_yes_no("Download and install update?"):
        print("Update cancelled.")
        return False

    if download_update(update_script_url):
        print("Update applied. Restart the program to run the new version.")
        return True

    print("Update failed.")
    return False


def respond_to(message: str) -> str:
    """A tiny "chat" responder so you can talk to the agent."""
    msg = (message or "").strip().lower()
    if not msg:
        return "Say something and I'll try to respond."
    if msg in {"hi", "hello", "hey"}:
        return "Hello! I'm VEDO. You can ask me to check for updates, show version, or start the server."
    if "update" in msg:
        return "To check for updates, use `check_update`. If you're using the HTTP server, call `/check_update`."
    if msg in {"who are you", "what are you"}:
        return "I'm VEDO, a local agent that can update itself with your permission."
    return f"I heard: {message}. Try `help` to see what I can do."


def chat_with_llm(message: str, config: Dict[str, Any]) -> str:
    """Send the message to Gemini (if configured) or fallback to a simple response."""
    api_key = config.get("gemini_api_key", "")
    model = config.get("gemini_model", "text-bison-001")

    if api_key:
        reply = call_gemini_api(message, api_key, model)
        if reply:
            return reply

    return respond_to(message)


def get_web_ui_html() -> str:
    """Return the embedded web UI HTML for the chat interface."""
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>VEDO Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }
    main { flex: 1; display: flex; flex-direction: column; }
    header { padding: 12px; background: #1c1c2d; color: #fff; }
    #log { flex: 1; padding: 12px; overflow-y: auto; background: #0b0b15; color: #eee; }
    #log div { margin-bottom: 10px; }
    #controls { padding: 12px; display: flex; gap: 8px; background: #111123; }
    input[type=text] { flex: 1; padding: 8px; border-radius: 4px; border: 1px solid #333; background: #121229; color: #eee; }
    button { padding: 8px 12px; border-radius: 4px; border: none; cursor: pointer; background: #3c3c9d; color: #fff; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    small { color: #bbb; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>VEDO Agent</h1>
      <p>Type a message and click Send. Voice uses your browser's speech recognition / speech synthesis.</p>
    </header>
    <div id=\"log\"></div>
    <div id=\"controls\">
      <input id=\"input\" type=\"text\" placeholder=\"Ask VEDO...\" autocomplete=\"off\" />
      <button id=\"send\">Send</button>
      <button id=\"mic\">🎤</button>
    </div>
  </main>
  <script>
    const log = document.getElementById('log');
    const input = document.getElementById('input');
    const send = document.getElementById('send');
    const mic = document.getElementById('mic');

    function appendMessage(who, text) {
      const el = document.createElement('div');
      el.innerHTML = `<strong>${who}:</strong> ${text}`;
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
    }

    async function sendMessage(msg) {
      appendMessage('You', msg);
      input.value = '';
      input.disabled = true;
      send.disabled = true;

      try {
        const url = new URL(window.location.origin + '/chat');
        url.searchParams.set('msg', msg);
        const res = await fetch(url.toString());
        const json = await res.json();
        appendMessage('VEDO', json.reply || '(no response)');
        speak(json.reply || '');
      } catch (err) {
        appendMessage('VEDO', 'Error: ' + err);
      }

      input.disabled = false;
      send.disabled = false;
      input.focus();
    }

    function speak(text) {
      if (!('speechSynthesis' in window) || !text) return;
      const ut = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(ut);
    }

    send.addEventListener('click', () => {
      const msg = input.value.trim();
      if (msg) sendMessage(msg);
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') send.click();
    });

    mic.addEventListener('click', () => {
      if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        appendMessage('VEDO', 'Speech recognition is not supported in this browser.');
        return;
      }

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognizer = new SpeechRecognition();
      recognizer.lang = 'en-US';
      recognizer.interimResults = false;
      recognizer.maxAlternatives = 1;

      recognizer.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
        send.click();
      };
      recognizer.onerror = (ev) => {
        appendMessage('VEDO', 'Speech recognition error: ' + ev.error);
      };
      recognizer.start();
    });

    // Welcome message
    appendMessage('VEDO', 'Hello! Type a message or press the microphone button to speak.');
  </script>
</body>
</html>
"""


# --- HTTP API ---------------------------------------------------------------

class ApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _assert_token(self, token: str, require_token: bool) -> bool:
        if not require_token:
            return True
        query = parse_qs(urlparse(self.path).query)
        return token and query.get("token", [""])[0] == token

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        config = self.server.config

        if not self._assert_token(config["serve"]["token"], config["serve"]["require_token"]):
            self._send_json({"error": "unauthorized"}, status=401)
            return

        if path == "" or path == "/" or path == "/info":
            self._send_json(
                {
                    "version": __version__,
                    "auto_update": config.get("auto_update", False),
                    "last_update_check": _last_update_check,
                    "update_version_url": config.get("update_version_url"),
                    "update_script_url": config.get("update_script_url"),
                }
            )
            return

        if path == "/check_update":
            auto_confirm = config.get("auto_update", False)
            updated = run_update_flow(
                config.get("update_version_url"),
                config.get("update_script_url"),
                auto_confirm=auto_confirm,
            )
            self._send_json({"updated": updated, "last_update_version": _last_update_check})
            return

        if path == "/chat" or path == "/talk":
            query = parse_qs(parsed.query)
            message = query.get("msg", [""])[0]
            self._send_json({"reply": chat_with_llm(message, config)})
            return

        if path == "/ui":
            self._send_html(get_web_ui_html())
            return

        self._send_json({"error": "not found"}, status=404)


def run_http_server(config: Dict[str, Any]) -> None:
    serve_cfg = config.get("serve", {})
    host = serve_cfg.get("host", "0.0.0.0")
    port = serve_cfg.get("port", 8000)

    class Server(HTTPServer):
        pass

    server = Server((host, port), ApiHandler)
    server.config = config

    print(f"[server] running on http://{host}:{port} (CTRL+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopped.")


# --- Agent interaction -------------------------------------------------------

def show_help() -> None:
    print(
        """
Available commands:
  help         - show this help message
  version      - show current version
  check_update - check remote version and ask to upgrade
  talk         - have a quick conversation with the agent
  ui           - print the URL for the web chat interface
  serve        - start HTTP server (for remote control)
  exit         - quit
"""
    )


def main() -> None:
    config = load_config()

    if config.get("auto_update"):
        # Auto-check/update on startup (use with caution).
        run_update_flow(
            config.get("update_version_url"),
            config.get("update_script_url"),
            auto_confirm=True,
        )

    print("VEDO agent (local). Type 'help' for commands.")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if cmd in {"", "help"}:
            show_help()
        elif cmd == "version":
            print(f"VEDO version: {__version__}")
        elif cmd == "check_update":
            run_update_flow(
                config.get("update_version_url"),
                config.get("update_script_url"),
            )
        elif cmd in {"talk", "chat"}:
            print("Enter a message (type 'exit' to return to the prompt).")
            while True:
                try:
                    message = input("you: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if message.lower() in {"exit", "quit"}:
                    break
                print("bot:", chat_with_llm(message, config))
        elif cmd == "ui":
            serve_cfg = config.get("serve", {})
            host = serve_cfg.get("host", "0.0.0.0")
            port = serve_cfg.get("port", 8000)
            print(f"Open this URL in your browser: http://{host}:{port}/ui")
        elif cmd == "serve":
            run_http_server(config)
        elif cmd in {"exit", "quit"}:
            print("Goodbye.")
            return
        else:
            print(f"Unknown command: {cmd}. Type 'help' for options.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VEDO agent")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start HTTP server for remote control",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.serve or cfg.get("serve", {}).get("enabled"):
        run_http_server(cfg)
    else:
        main()
