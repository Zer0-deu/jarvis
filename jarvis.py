#!/usr/bin/env python3
"""
Desktop clap listener: reads the default microphone and logs when two loud transients
(a double clap) are detected within a short time window.

Run:
  python -m pip install -r requirements.txt
  python clap_listen.py

Tuning (constants below):
  SAMPLE_RATE   — usually 44100 or 48000; match your device if needed.
  BLOCK_MS      — analysis window size; smaller = snappier, noisier.
  SPIKE_RATIO   — how many times louder than the noise floor counts as a clap;
                    raise if false triggers; lower if claps are missed.
  COOLDOWN_S    — minimum seconds between double-clap logs (debounce).
  MIN_DOUBLE_GAP_S / MAX_DOUBLE_GAP_S — allowed time between the two claps.
  RETRIGGER_RATIO — audio must fall below threshold * this before another hit counts.
  NOISE_FLOOR_ALPHA — closer to 1 = slower baseline adaptation to room noise.
  MIN_RMS       — ignore spikes below this absolute level (float audio ~ [-1, 1]).
  SONG_URI      — Spotify or YouTube URL/URI to open on each double clap (empty = log only).
  FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP — if True, launch Cursor without -n (reuse / focus existing instance).
  OPEN_NEW_CURSOR_ON_DOUBLE_CLAP — if True, also launch Cursor with -n (extra new window; runs after focus launch if both).
  CURSOR_OPEN_FULLSCREEN — Windows: after focus/launch, send F11 to enter Cursor/VS Code-style fullscreen (toggle off with F11).
  OPEN_CLAUDE_CODE_IN_CHROME — Claude in Chrome after Spotify (CLAUDE_CODE_URL).
  OPEN_BINANCE_BTC_IN_CHROME — Binance BTC trade page in Chrome (BINANCE_BTC_URL).
  CLAUDE_CHROME_MONITOR / BINANCE_CHROME_MONITOR — 1-based display index (Windows: sorted left-to-top).
  CHROME_SEPARATE_SITE_PROFILES — Windows: if True, uses temp --user-data-dir per site (not your normal profile).
    Default False so Claude/Binance use your usual Chrome profile and logins; enable only if both windows keep
    opening on the same monitor and you accept a separate profile for automation.
  OPEN_CHROME_FULLSCREEN — Fullscreen on the chosen monitor (Windows: new window is detected and snapped with SetWindowPos).
  JARVIS_WELCOME_* — TTS after the song (ElevenLabs). Configure via environment or a `.env`
    file next to this script (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, etc.).
    With JARVIS_WELCOME_CACHE_ENABLED, audio is saved under `.cache/jarvis_welcome/` (WAV) and
    replayed when phrase + voice + model + format match—no repeat API call. Delete that folder
    or set JARVIS_WELCOME_CACHE_ENABLED=False to force a fresh fetch.
  The welcome sequence runs only once per process. The assistant speaks in the background so Cursor
    opens without waiting for playback to finish (restart the script to run again).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import wave
import webbrowser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import requests
import random
import speech_recognition as sr
import http.server

# --- tuning knobs -----------------------------------------------------------
SAMPLE_RATE = 44100
BLOCK_MS = 40
CHANNELS = 1

SPIKE_RATIO = 7.0
COOLDOWN_S = 0.45
MIN_DOUBLE_GAP_S = 0.05
MAX_DOUBLE_GAP_S = 0.35
RETRIGGER_RATIO = 0.55
NOISE_FLOOR_ALPHA = 0.992
MIN_RMS = 0.012
QUIET_GATE_MULT = 2.2  # update noise floor only when below floor * this
# Startup mic probe: if default input RMS stays below this, scan for a louder device.
INPUT_PROBE_S = 0.5
INPUT_SILENT_RMS = 0.001

# Spotify: "spotify:track:TRACK_ID" or https://open.spotify.com/track/...
# YouTube: https://www.youtube.com/watch?v=...
SONG_URI = "https://open.spotify.com/track/39shmbIHICJ2Wxnk1fPSdz?si=2900c75c2e2d4b82"

# Cursor: focus existing instance (no -n). Set OPEN_NEW_CURSOR_ON_DOUBLE_CLAP for a new window as well.
FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP = False
OPEN_NEW_CURSOR_ON_DOUBLE_CLAP = False
CURSOR_OPEN_FULLSCREEN = True

# Google Chrome (fallback: default browser). URLs overridable in .env.
OPEN_CLAUDE_CODE_IN_CHROME = False
OPEN_BINANCE_BTC_IN_CHROME = False
OPEN_CHROME_FULLSCREEN = False
# False = default Chrome profile (your normal user, extensions, cookies). True = temp dirs under %TEMP% per site.
CHROME_SEPARATE_SITE_PROFILES = False
# Which physical screen (1 = leftmost/top-first after sorting). Windows only; ignored elsewhere.
CLAUDE_CHROME_MONITOR = 1
BINANCE_CHROME_MONITOR = 1

JARVIS_WELCOME_ENABLED = True
JARVIS_WELCOME_PHRASE = "Greetings sir,welcome back..how have you been"
# Seconds after launching SONG_URI before speaking (gives Spotify/browser time to prepare).
JARVIS_AFTER_SONG_DELAY_S = 3.0
JARVIS_SONG_PLAYBACK_DELAY_S = 0.5
# Voice command recording settings.
JARVIS_VOICE_COMMAND_MAX_S = 8.0
JARVIS_VOICE_COMMAND_BLOCK_MS = 120
JARVIS_VOICE_COMMAND_SILENCE_THRESHOLD = 0.02
JARVIS_VOICE_COMMAND_END_S = 1.0
JARVIS_WAKEWORD_ENABLED = False
JARVIS_WAKEWORD = "jarvis"
# Save ElevenLabs PCM as WAV under .cache/jarvis_welcome/; replay skips the API when the key matches.
JARVIS_WELCOME_CACHE_ENABLED = True
JARVIS_AGENT_ENABLED = True
JARVIS_VOICE_AGENT_ENABLED = True

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("clap_listen")


def _post_with_retries(url: str, *, json=None, headers=None, timeout: int = 10, max_retries: int = 4, backoff_factor: float = 1.0):
    """POST with retries for 429 / 5xx and transient network errors.
    Honors `Retry-After` header when present. Raises the final exception on failure.
    Returns a `requests.Response` on success.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=json, headers=headers, timeout=timeout)
            # If rate limited or server error, consider retrying
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                ra = resp.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra is not None else backoff_factor * (2 ** (attempt - 1))
                except Exception:
                    wait = backoff_factor * (2 ** (attempt - 1))
                wait = max(0.1, wait) + random.uniform(0, 0.5)
                log.warning(
                    "POST %s returned %s; Retry-After=%s; retrying after %.1fs (attempt %d/%d)",
                    url,
                    resp.status_code,
                    ra,
                    wait,
                    attempt,
                    max_retries,
                )
                if attempt == max_retries:
                    # Final attempt: raise a clear HTTPError including Retry-After when present
                    try:
                        reason = resp.reason
                    except Exception:
                        reason = ""
                    msg = f"{resp.status_code} {reason}: {resp.text}"
                    if ra:
                        msg += f" (Retry-After: {ra})"
                    raise requests.HTTPError(msg)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            # Detect DNS / name resolution errors and fail fast with a clear message
            estr = str(e)
            if "NameResolutionError" in estr or "Failed to resolve" in estr or "getaddrinfo failed" in estr:
                log.error("DNS resolution failed when contacting %s: %s", url, estr)
                raise requests.ConnectionError(f"DNS resolution failed contacting {url}: {estr}")

            # Attempt to surface any Retry-After or status info from the response if available
            resp = getattr(e, "response", None)
            ra = None
            status = None
            if resp is not None:
                try:
                    ra = resp.headers.get("Retry-After")
                except Exception:
                    ra = None
                status = getattr(resp, "status_code", None)
            if attempt == max_retries:
                # Include Retry-After in final error when possible
                if status or ra:
                    parts = [str(status) if status else "", str(e)]
                    if ra:
                        parts.append(f"Retry-After: {ra}")
                    raise requests.HTTPError("; ".join([p for p in parts if p]))
                raise
            wait = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning("Request error to %s: %s; retrying in %.1fs (attempt %d/%d)", url, e, wait, attempt, max_retries)
            time.sleep(wait)
    raise RuntimeError("Retries exhausted for POST %s" % url)

# --- JARVIS core: persistent memory, task queue, scheduling, news, sentiment, actions
import json
import urllib.request
import xml.etree.ElementTree as ET


class JarvisCore:
    def __init__(self, base_dir: Path | None = None):
        self.base = (Path(base_dir) if base_dir else Path(__file__).resolve().parent) / ".jarvis_data"
        self.base.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.base / "memory.json"
        self.tasks_path = self.base / "tasks.json"
        self.calendar_path = self.base / "calendar.json"
        self._memory = self._load_json(self.memory_path) or {}
        self._tasks = self._load_json(self.tasks_path) or []
        self._calendar = self._load_json(self.calendar_path) or []
        self._task_lock = threading.Lock()
        self._pending_actions: dict[int, dict] = {}
        self._pending_lock = threading.Lock()
        self._start_scheduler_thread()

    # Pending action confirmation (for risky actions)
    def add_pending_action(self, action: dict) -> int:
        with self._pending_lock:
            aid = int(time.time() * 1000)
            self._pending_actions[aid] = {"action": action, "ts": time.time()}
            return aid

    def confirm_pending_action(self, aid: int) -> dict | None:
        with self._pending_lock:
            act = self._pending_actions.pop(aid, None)
        if not act:
            return None
        a = act.get("action")
        if not a:
            return None
        if a.get("type") == "run_command":
            return self.run_command(a.get("cmd"))
        return None

    def list_pending_actions(self) -> dict:
        return dict(self._pending_actions)

    def calendar_oauth_instructions(self) -> str:
        return (
            "To integrate Google Calendar: create OAuth credentials at https://console.developers.google.com, "
            "set a redirect URI to http://localhost:PORT/callback, and set environment variables: "
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET. Then implement an OAuth exchange to obtain tokens. "
            "This helper only provides instructions; provide credentials and I can scaffold the OAuth flow."
        )

    # Google Calendar OAuth and API helpers
    def _google_tokens_path(self) -> Path:
        return self.base / "google_tokens.json"

    def _load_google_tokens(self) -> dict | None:
        return self._load_json(self._google_tokens_path())

    def _save_google_tokens(self, obj: dict) -> None:
        self._save_json(self._google_tokens_path(), obj)

    def google_disconnect(self) -> bool:
        p = self._google_tokens_path()
        try:
            if p.is_file():
                p.unlink()
            return True
        except Exception:
            return False

    def _refresh_google_token(self, client_id: str, client_secret: str, tokens: dict) -> dict | None:
        if not tokens or not tokens.get("refresh_token"):
            return None
        try:
            data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
                "refresh_token": tokens.get("refresh_token"),
            }
            r = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=10)
            r.raise_for_status()
            new = r.json()
            # merge
            tokens.update({k: v for k, v in new.items() if k in ("access_token", "expires_in", "scope", "token_type")})
            tokens["expires_at"] = time.time() + int(new.get("expires_in", 0))
            self._save_google_tokens(tokens)
            return tokens
        except Exception:
            log.exception("Failed to refresh Google token")
            return None

    def _ensure_google_tokens(self) -> dict | None:
        tokens = self._load_google_tokens()
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not tokens:
            return None
        if tokens.get("expires_at") and tokens.get("expires_at") > time.time() + 30:
            return tokens
        # try refresh
        if client_id and client_secret:
            return self._refresh_google_token(client_id, client_secret, tokens)
        return tokens

    def google_start_oauth(self, host: str = "localhost", port: int = 8765, scope: str | None = None, timeout: int = 300) -> str:
        """Start an OAuth flow to obtain Google Calendar tokens.
        Returns a short status message; will open the browser for the user to consent.
        """
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            return "Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in environment."

        scope = scope or "https://www.googleapis.com/auth/calendar.readonly"
        redirect_path = "/oauth2callback"
        redirect_uri = f"http://{host}:{port}{redirect_path}"
        state = hashlib.sha256(f"{time.time()}-{client_id}".encode()).hexdigest()

        auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth?"
            + urllib.parse.urlencode(
                {
                    "client_id": client_id,
                    "response_type": "code",
                    "scope": scope,
                    "redirect_uri": redirect_uri,
                    "access_type": "offline",
                    "prompt": "consent",
                    "state": state,
                }
            )
        )

        # start a local HTTP server to receive the code
        code_box: dict = {"code": None}
        received = threading.Event()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                qs = urllib.parse.urlparse(self.path)
                if qs.path != redirect_path:
                    self.send_response(404)
                    self.end_headers()
                    return
                params = urllib.parse.parse_qs(qs.query)
                code = params.get("code", [None])[0]
                # respond
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization received. You can close this tab.</h1></body></html>")
                code_box["code"] = code
                received.set()

            def log_message(self, format, *args):
                return

        server = http.server.ThreadingHTTPServer((host, port), _Handler)

        def _serve():
            try:
                server.serve_forever()
            except Exception:
                pass

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        try:
            webbrowser.open(auth_url)
        except Exception:
            log.info("Open the following URL in your browser: %s", auth_url)

        waited = received.wait(timeout)
        server.shutdown()
        if not waited:
            return "Timed out waiting for authorization code."
        code = code_box.get("code")
        if not code:
            return "No code received."

        # exchange code
        try:
            data = {
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
            r = requests.post("https://oauth2.googleapis.com/token", data=data, timeout=10)
            r.raise_for_status()
            tokens = r.json()
            tokens["expires_at"] = time.time() + int(tokens.get("expires_in", 0))
            self._save_google_tokens(tokens)
            return "Google Calendar connected and tokens saved."
        except Exception as e:
            log.exception("Failed to exchange code for tokens")
            return f"Token exchange failed: {e}"

    def calendar_list_google_events(self, max_results: int = 10) -> list[str]:
        tokens = self._ensure_google_tokens()
        if not tokens or not tokens.get("access_token"):
            return []
        headers = {"Authorization": f"Bearer {tokens.get('access_token')}"}
        now = datetime.utcnow().isoformat() + "Z"
        params = {"timeMin": now, "maxResults": max_results, "orderBy": "startTime", "singleEvents": "true"}
        try:
            r = requests.get("https://www.googleapis.com/calendar/v3/calendars/primary/events", params=params, headers=headers, timeout=10)
            r.raise_for_status()
            j = r.json()
            items = j.get("items") or []
            out = []
            for it in items:
                start = it.get("start", {}).get("dateTime") or it.get("start", {}).get("date")
                summary = it.get("summary") or "(no title)"
                out.append(f"{start} {summary}")
            return out
        except Exception:
            log.exception("Failed to fetch Google Calendar events")
            return []

    def _load_json(self, p: Path):
        try:
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            log.warning("Failed to read %s", p)
        return None

    def _save_json(self, p: Path, obj) -> None:
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception:
            log.exception("Failed to write %s", p)

    # Memory API
    def remember(self, key: str, value: str) -> None:
        self._memory[key] = {"value": value, "ts": time.time()}
        self._save_json(self.memory_path, self._memory)

    def recall(self, key: str) -> str | None:
        v = self._memory.get(key)
        return v["value"] if v else None

    def forget(self, key: str) -> bool:
        if key in self._memory:
            del self._memory[key]
            self._save_json(self.memory_path, self._memory)
            return True
        return False

    def list_memory(self) -> dict:
        return {k: v["value"] for k, v in self._memory.items()}

    # Tasks
    def add_task(self, title: str, metadata: dict | None = None) -> int:
        with self._task_lock:
            tid = int(time.time() * 1000)
            item = {"id": tid, "title": title, "meta": metadata or {}, "created": time.time(), "done": False}
            self._tasks.append(item)
            self._save_json(self.tasks_path, self._tasks)
            return tid

    def list_tasks(self) -> list:
        return list(self._tasks)

    def complete_task(self, tid: int) -> bool:
        with self._task_lock:
            for t in self._tasks:
                if t.get("id") == tid:
                    t["done"] = True
                    self._save_json(self.tasks_path, self._tasks)
                    return True
        return False

    # Calendar/scheduling
    def schedule_event(self, ts_iso: str, title: str) -> bool:
        try:
            dt = datetime.fromisoformat(ts_iso)
        except Exception:
            return False
        self._calendar.append({"when": dt.isoformat(), "title": title})
        self._save_json(self.calendar_path, self._calendar)
        return True

    def list_events(self) -> list:
        return list(self._calendar)

    def _start_scheduler_thread(self) -> None:
        def _runner():
            while True:
                try:
                    now = datetime.now()
                    to_run = []
                    remaining = []
                    for e in self._calendar:
                        try:
                            when = datetime.fromisoformat(e["when"])
                            if when <= now:
                                to_run.append(e)
                            else:
                                remaining.append(e)
                        except Exception:
                            remaining.append(e)
                    if to_run:
                        self._calendar = remaining
                        self._save_json(self.calendar_path, self._calendar)
                        for e in to_run:
                            log.info("Scheduled event triggered: %s", e.get("title"))
                    time.sleep(30)
                except Exception:
                    log.exception("Scheduler thread error")
                    time.sleep(5)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    # News fetcher
    def fetch_news(self, source: str | None = None, limit: int = 5) -> list[str]:
        # default to BBC RSS for public headlines
        url = source or "http://feeds.bbci.co.uk/news/rss.xml"
        try:
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            items = root.findall('.//item')
            headlines = []
            for it in items[:limit]:
                t = it.find('title')
                if t is not None and t.text:
                    headlines.append(t.text.strip())
            return headlines
        except Exception:
            log.exception("News fetch failed")
            return []
    # Simple sentiment analysis (rule-based)
    def sentiment(self, text: str) -> dict:
        if not text:
            return {"label": "neutral", "score": 0.0}
        pos_words = {"good", "great", "happy", "love", "excellent", "awesome", "fantastic", "positive", "nice", "best"}
        neg_words = {"bad", "sad", "hate", "terrible", "awful", "poor", "negative", "angry", "worse", "worst"}
        t = text.lower()
        score = 0
        for w in pos_words:
            if w in t:
                score += 1
        for w in neg_words:
            if w in t:
                score -= 1
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        return {"label": label, "score": float(score)}

    def run_command(self, cmd: str, timeout: int = 10) -> dict:
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except Exception as e:
            return {"rc": -1, "stdout": "", "stderr": str(e)}
        # Note: LLM call logic was removed from here; `run_command` only executes shell commands.


def open_path(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Path does not exist: {path}"
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Opened {path}."
    except Exception as e:
        return f"Unable to open path: {e}"


def take_screenshot() -> str:
    try:
        from PIL import ImageGrab
    except ImportError:
        return "Screenshot unavailable: install Pillow to enable screenshots."

    screenshot_dir = Path(Path(__file__).resolve().parent / ".cache" / "jarvis_screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = screenshot_dir / f"screenshot-{int(time.time())}.png"
    try:
        ImageGrab.grab().save(path)
        return f"Saved screenshot to {path}"
    except Exception as e:
        return f"Screenshot capture failed: {e}"


def llm_reason(prompt: str, timeout: int = 20) -> str:
    """Ask a configured LLM to reason about `prompt`.

    Provider selection (environment variables):
      - `LLM_PROVIDER=OLLAMA` to use a local Ollama installation via the `ollama` CLI.
      - `LLM_PROVIDER=OLLAMA_HTTP` to use a local Ollama HTTP server (set `OLLAMA_HTTP_URL`).
      - default: GROQ via `GROQ_API_KEY` / `XAI_API_KEY`.
    """
    provider = (os.getenv("LLM_PROVIDER") or "").upper()

    # Ollama CLI provider
    if provider == "OLLAMA":
        model = os.getenv("OLLAMA_MODEL") or "llama2"
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if proc.returncode != 0:
                serr = proc.stderr.decode(errors="replace")
                return f"Ollama CLI error: {serr.strip() or proc.returncode}"
            return proc.stdout.decode(errors="replace").strip()
        except FileNotFoundError:
            return "Ollama CLI not found: install Ollama and ensure `ollama` is on PATH."
        except subprocess.TimeoutExpired:
            return "Ollama CLI timed out while generating a response."
        except Exception as e:
            return f"Ollama CLI error: {e}"

    # Ollama HTTP provider (local server)
    if provider == "OLLAMA_HTTP":
        ollama_url = os.getenv("OLLAMA_HTTP_URL") or "http://127.0.0.1:11434/v1/generate"
        model = os.getenv("OLLAMA_MODEL") or "llama2"
        try:
            payload = {"model": model, "prompt": prompt}
            r = requests.post(ollama_url, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j.get("text") or j.get("result") or str(j)
        except requests.RequestException as e:
            estr = str(e)
            if "Failed to establish a new connection" in estr or "Connection refused" in estr:
                return (
                    f"Ollama HTTP server unreachable at {ollama_url}. Start the Ollama server or check the URL."
                )
            return f"Ollama HTTP request failed: {e}"

    # Prefer GROQ/XAI if configured (do not use OpenAI)
    # This branch intentionally avoids contacting OpenAI even if OPENAI_API_KEY is present.

    # Default: GROQ-compatible endpoint
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if not groq_key:
        return "No LLM configured: set GROQ_API_KEY, XAI_API_KEY, OPENAI_API_KEY, or set LLM_PROVIDER=OLLAMA for local Ollama."
    groq_url = "https://api.groq.x.ai/v1"
    try:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        resp = _post_with_retries(groq_url, json={"prompt": prompt}, headers=headers, timeout=timeout)
        j = resp.json()
        return j.get("text") or j.get("result") or j.get("output") or str(j)
    except Exception as e:
        estr = str(e)
        if "NameResolutionError" in estr or "Failed to resolve" in estr or "getaddrinfo failed" in estr:
            return (
                "Groq request failed: DNS resolution error contacting api.groq.x.ai. "
                "Check your network, DNS settings, VPN, or hosts file, and try again."
            )
        return f"Groq request failed: {e}"


def llm_test_key(timeout: int = 20) -> str:
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if not groq_key:
        return "No LLM API key configured. Set GROQ_API_KEY (or XAI_API_KEY)."
    groq_url = "https://api.groq.x.ai/v1"
    try:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        resp = _post_with_retries(groq_url, json={"prompt": "Say hello"}, headers=headers, timeout=timeout)
        return "Groq key is valid and the endpoint responded successfully."
    except Exception as e:
        estr = str(e)
        if "NameResolutionError" in estr or "Failed to resolve" in estr or "getaddrinfo failed" in estr:
            return (
                "Groq key test failed: DNS resolution error contacting api.groq.x.ai. "
                "Check your network, DNS, VPN, or hosts file."
            )
        return f"Groq key test failed: {e}"


# Global Jarvis core instance
JARVIS_CORE = JarvisCore()


def block_samples() -> int:
    n = int(SAMPLE_RATE * BLOCK_MS / 1000)
    return max(n, 1)


def rms_mono(block: np.ndarray) -> float:
    if block.ndim > 1:
        block = np.mean(block.astype(np.float64), axis=1)
    else:
        block = block.astype(np.float64)
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))


def _input_devices() -> list[tuple[int, dict]]:
    return [
        (i, dev)
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] >= 1
    ]


def format_input_devices() -> str:
    """Return a formatted string listing available input audio devices."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        return f"Could not query audio devices: {e}"
    lines = []
    default_idx = sd.default.device[0] if sd.default and sd.default.device else None
    for i, dev in enumerate(devices):
        mark = "<default>" if default_idx is not None and i == default_idx else ""
        lines.append(f"{i:3d}: {dev.get('name')} (inputs={dev.get('max_input_channels')}) {mark}")
    return "\n".join(lines)


def _resolve_input_device_index(spec: str) -> int:
    spec = spec.strip()
    if spec.isdigit():
        idx = int(spec)
        sd.query_devices(idx)
        return idx
    needle = spec.lower()
    for idx, dev in _input_devices():
        if needle in dev["name"].lower():
            return idx
    raise ValueError(f"No input device matches {spec!r}")


def _probe_input_max_rms(device: int, blocksize: int) -> float | None:
    try:
        with sd.InputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            peak = 0.0
            deadline = time.monotonic() + INPUT_PROBE_S
            while time.monotonic() < deadline:
                data, _ = stream.read(blocksize)
                peak = max(peak, rms_mono(data))
            return peak
    except sd.PortAudioError:
        return None


def _choose_input_device(blocksize: int) -> int:
    log.info("Audio devices:\n%s", sd.query_devices())

    override = (os.environ.get("JARVIS_INPUT_DEVICE") or "").strip()
    if override:
        try:
            idx = _resolve_input_device_index(override)
        except ValueError as e:
            log.error("%s", e)
            log.error("Set JARVIS_INPUT_DEVICE to a device index or name substring.")
            raise SystemExit(1) from e
        name = sd.query_devices(idx)["name"]
        peak = _probe_input_max_rms(idx, blocksize)
        log.info("Using JARVIS_INPUT_DEVICE [%d]: %s", idx, name)
        if peak is None:
            log.warning("Could not open configured mic; trying anyway.")
        elif peak < INPUT_SILENT_RMS:
            log.warning(
                "Configured mic looks silent (probe rms=%.5f). "
                "Check Windows input level or try another JARVIS_INPUT_DEVICE.",
                peak,
            )
        else:
            log.info("Mic probe OK (rms=%.5f).", peak)
        return idx

    default = sd.default.device[0]
    if default is not None and default >= 0:
        default_name = sd.query_devices(default)["name"]
        peak = _probe_input_max_rms(default, blocksize)
        if peak is not None and peak >= INPUT_SILENT_RMS:
            log.info(
                "Using default microphone [%d]: %s (probe rms=%.5f)",
                default,
                default_name,
                peak,
            )
            return default
        log.warning(
            "Default mic [%d] %s is silent or unavailable (probe rms=%s); "
            "scanning other inputs...",
            default,
            default_name,
            f"{peak:.5f}" if peak is not None else "unopenable",
        )

    best_idx: int | None = None
    best_peak = -1.0
    for idx, dev in _input_devices():
        if default is not None and idx == default:
            continue
        peak = _probe_input_max_rms(idx, blocksize)
        if peak is not None and peak > best_peak:
            best_peak = peak
            best_idx = idx

    if best_idx is not None and best_peak >= INPUT_SILENT_RMS:
        log.info(
            "Auto-selected microphone [%d]: %s (probe rms=%.5f)",
            best_idx,
            sd.query_devices(best_idx)["name"],
            best_peak,
        )
        return best_idx

    if default is not None and default >= 0:
        log.warning("No active mic found; falling back to default [%d].", default)
        return default
    inputs = _input_devices()
    if not inputs:
        log.error("No input devices found.")
        raise SystemExit(1)
    idx, dev = inputs[0]
    log.warning("No active mic found; falling back to [%d] %s.", idx, dev["name"])
    return idx


def _elevenlabs_pcm_sample_rate(output_format: str) -> int:
    override = (os.environ.get("ELEVENLABS_PCM_SAMPLE_RATE") or "").strip()
    if override.isdigit():
        return int(override)
    if output_format.startswith("pcm_"):
        try:
            return int(output_format.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            pass
    return 24000


def elevenlabs_env_config() -> tuple[str, str, str, int]:
    """voice_id, model_id, output_format, pcm_sample_rate."""
    voice = (os.environ.get("ELEVENLABS_VOICE_ID") or "").strip()
    model = (os.environ.get("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2").strip()
    fmt = (os.environ.get("ELEVENLABS_OUTPUT_FORMAT") or "pcm_24000").strip()
    rate = _elevenlabs_pcm_sample_rate(fmt)
    return voice, model, fmt, rate


def _jarvis_welcome_cache_dir() -> Path:
    base = Path(__file__).resolve().parent
    override = (os.environ.get("JARVIS_WELCOME_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return base / ".cache" / "jarvis_welcome"


def _jarvis_welcome_cache_path(
    text: str, voice_id: str, model_id: str, output_format: str
) -> Path:
    key = f"{text}|{voice_id}|{model_id}|{output_format}".encode()
    digest = hashlib.sha256(key).hexdigest()[:24]
    return _jarvis_welcome_cache_dir() / f"{digest}.wav"


def _play_pcm_wav_file(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            if ch != 1 or sw != 2:
                log.warning("Unsupported cached WAV (channels=%s, width=%s).", ch, sw)
                return False
            raw = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as e:
        log.warning("Could not read cached welcome audio: %s", e)
        return False
    if not raw:
        return False
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f = pcm_i16.astype(np.float32) / 32768.0
    try:
        sd.play(pcm_f, rate)
        sd.wait()
    except Exception as e:
        log.warning("Could not play cached welcome audio: %s", e)
        return False
    return True


def _save_pcm_wav_file(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        tmp.replace(path)
    except OSError:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def say_text(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    vid, model_id, output_format, pcm_rate = elevenlabs_env_config()
    if not vid:
        log.warning("Set ELEVENLABS_VOICE_ID in the environment for ElevenLabs TTS.")
        return

    cache_path = _jarvis_welcome_cache_path(text, vid, model_id, output_format)
    if JARVIS_WELCOME_CACHE_ENABLED and cache_path.is_file():
        log.info("Playing cached audio for text: %s", text)
        if _play_pcm_wav_file(cache_path):
            return
        log.warning("Cache miss after read failure; fetching from ElevenLabs.")

    api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        log.warning("Set ELEVENLABS_API_KEY in the environment for ElevenLabs TTS.")
        return
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        log.warning("Install dependencies: pip install -r requirements.txt")
        return
    try:
        client = ElevenLabs(api_key=api_key)
        chunks = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=model_id,
            output_format=output_format,
        )
        raw = b"".join(chunks)
    except Exception as e:
        log.warning("ElevenLabs TTS failed: %s", e)
        return
    if not raw:
        log.warning("ElevenLabs returned empty audio.")
        return
    if JARVIS_WELCOME_CACHE_ENABLED:
        try:
            _save_pcm_wav_file(cache_path, raw, pcm_rate)
            log.info("Saved generated audio to cache: %s", cache_path)
        except OSError as e:
            log.warning("Could not save welcome cache: %s", e)
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f = pcm_i16.astype(np.float32) / 32768.0
    try:
        sd.play(pcm_f, pcm_rate)
        sd.wait()
    except Exception as e:
        log.warning("Could not play ElevenLabs audio: %s", e)


def say_jarvis_welcome() -> None:
    if not JARVIS_WELCOME_ENABLED or not JARVIS_WELCOME_PHRASE.strip():
        return
    say_text(JARVIS_WELCOME_PHRASE.strip())


def open_url_default_browser(url: str) -> None:
    u = url.strip()
    if not u:
        return
    try:
        webbrowser.open_new_tab(u)
    except OSError as e:
        log.warning("Could not open URL in default browser: %s", e)


def play_song(uri: str) -> None:
    u = uri.strip()
    if not u:
        return
    try:
        if u.lower().startswith("spotify:"):
            if sys.platform == "win32":
                os.startfile(u)
            else:
                open_url_default_browser(u)
        elif u.lower().startswith(("http://", "https://")):
            open_url_default_browser(u)
        else:
            if sys.platform == "win32":
                os.startfile(u)
            else:
                open_url_default_browser(u)
    except OSError as e:
        log.warning("Could not open SONG_URI: %s", e)


def open_search_in_default_browser(query: str) -> None:
    q = (query or "").strip()
    if not q:
        return
    url = f"https://www.google.com/search?q={urllib.parse.quote(q)}"
    log.info("Opening search in default browser: %s", url)
    open_url_default_browser(url)


def open_claude_in_default_browser() -> None:
    query = (os.environ.get("CLAUDE_CODE_SEARCH") or "claude code").strip()
    if not query:
        return
    open_search_in_default_browser(query)


def _chrome_executable() -> str | None:
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not base:
                continue
            p = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
            if os.path.isfile(p):
                return p
    return shutil.which("google-chrome") or shutil.which("chrome")


def _win32_sorted_monitor_rects() -> list[tuple[int, int, int, int]]:
    """Each monitor as (left, top, right, bottom), sorted left-to-right then top-to-bottom."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    collected: list[tuple[int, int, int, int]] = []

    @ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )
    def _cb(_hm, _hdc, lprc, _lp):
        r = lprc.contents
        collected.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, _cb, 0)
    collected.sort(key=lambda t: (t[0], t[1]))
    return collected


def _chrome_monitor_top_left(one_based_index: int) -> tuple[int, int]:
    """Top-left corner on virtual desktop for monitor N (1-based)."""
    l, t, _, _ = _chrome_monitor_bounds(one_based_index)
    return (l, t)


def _chrome_monitor_bounds(one_based_index: int) -> tuple[int, int, int, int]:
    """Monitor N as (left, top, right, bottom), 1-based index (sorted like other Chrome helpers)."""
    rects = _win32_sorted_monitor_rects()
    if not rects:
        return (0, 0, 1920, 1080)
    idx = one_based_index - 1
    if idx < 0:
        idx = 0
    if idx >= len(rects):
        log.warning(
            "Monitor %d requested but only %d found; using last monitor.",
            one_based_index,
            len(rects),
        )
        idx = len(rects) - 1
    return rects[idx]


def _chrome_monitor_pixel_size(one_based_index: int) -> tuple[int, int]:
    l, t, r, b = _chrome_monitor_bounds(one_based_index)
    return (max(320, r - l), max(240, b - t))


def _chrome_window_size() -> tuple[int, int]:
    w = (os.environ.get("CHROME_WINDOW_WIDTH") or "1400").strip()
    h = (os.environ.get("CHROME_WINDOW_HEIGHT") or "900").strip()
    try:
        return (max(400, int(w)), max(300, int(h)))
    except ValueError:
        return (1400, 900)


def _chrome_site_user_data_dir(site_key: str) -> str:
    p = Path(tempfile.gettempdir()) / "clap-trigger-chrome" / site_key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _chrome_new_window_wait_timeout_s() -> float:
    try:
        return max(3.0, float((os.environ.get("CHROME_NEW_WINDOW_WAIT_S") or "25").strip()))
    except ValueError:
        return 25.0


def _chrome_top_level_browser_hwnds_win32() -> set[int]:
    """HWND ints for visible-or-minimized top-level Chrome browser windows."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    GW_OWNER = 4
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    found: set[int] = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: wintypes.HWND, _lp: wintypes.LPARAM) -> bool:
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        if not user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return True
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        try:
            buf = ctypes.create_unicode_buffer(4096)
            sz = wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                return True
            exe_path = buf.value
        finally:
            kernel32.CloseHandle(hproc)
        if os.path.basename(exe_path).lower() != "chrome.exe":
            return True
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < 80 or h < 80:
            return True
        found.add(int(hwnd))
        return True

    user32.EnumWindows(_enum, 0)
    return found


def _wait_new_chrome_hwnd_win32(before: set[int], timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.12)
        now = _chrome_top_level_browser_hwnds_win32()
        new = now - before
        if not new:
            continue
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        best: int | None = None
        best_area = 0
        for h in new:
            r = wintypes.RECT()
            if user32.GetWindowRect(h, ctypes.byref(r)):
                a = max(0, r.right - r.left) * max(0, r.bottom - r.top)
                if a > best_area:
                    best_area = a
                    best = h
        if best is not None:
            return best
    return None


def _chrome_snap_window_to_monitor_win32(
    hwnd: int,
    one_based_monitor: int,
    *,
    fullscreen: bool,
    windowed_size: tuple[int, int] | None,
) -> None:
    import ctypes
    from ctypes import wintypes

    ml, mt, mr, mb = _chrome_monitor_bounds(one_based_monitor)
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    SW_SHOWMAXIMIZED = 3
    HWND_TOP = 0
    SWP_SHOWWINDOW = 0x0040
    SWP_FRAMECHANGED = 0x0020
    flags = SWP_SHOWWINDOW | SWP_FRAMECHANGED

    user32.ShowWindow(hwnd, SW_RESTORE)
    if fullscreen:
        w, h = mr - ml, mb - mt
        x, y = ml, mt
    else:
        ww, wh = windowed_size or _chrome_window_size()
        w, h = ww, wh
        x = ml + max(0, (mr - ml - w) // 2)
        y = mt + max(0, (mb - mt - h) // 2)
    user32.SetWindowPos(hwnd, HWND_TOP, x, y, w, h, flags)

    if fullscreen:
        user32.ShowWindow(hwnd, SW_SHOWMAXIMIZED)
        KEYEVENTF_KEYUP = 0x0002
        VK_F11 = 0x7A
        fg = user32.GetForegroundWindow()
        tid_tgt = user32.GetWindowThreadProcessId(hwnd, None)
        tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        if tid_fg and tid_tgt:
            user32.AttachThreadInput(tid_fg, tid_tgt, True)
        user32.SetForegroundWindow(hwnd)
        if tid_fg and tid_tgt:
            user32.AttachThreadInput(tid_fg, tid_tgt, False)
        user32.keybd_event(VK_F11, 0, 0, 0)
        user32.keybd_event(VK_F11, 0, KEYEVENTF_KEYUP, 0)


def _open_url_in_chrome(
    url: str,
    *,
    new_window: bool = True,
    label: str = "URL",
    window_position: tuple[int, int] | None = None,
    window_size: tuple[int, int] | None = None,
    fullscreen: bool = False,
    win32_post_fullscreen_monitor: int | None = None,
    user_data_dir: str | None = None,
) -> None:
    u = url.strip()
    if not u:
        return
    chrome = _chrome_executable()
    try:
        if chrome:
            args = [chrome]
            if user_data_dir:
                args.append(f"--user-data-dir={user_data_dir}")
                args.append("--no-first-run")
            if new_window:
                args.append("--new-window")
            if window_position is not None:
                x, y = window_position
                args.append(f"--window-position={x},{y}")
            if window_size:
                args.append(f"--window-size={window_size[0]},{window_size[1]}")
            if fullscreen and not (
                sys.platform == "win32" and win32_post_fullscreen_monitor is not None
            ):
                args.append("--start-fullscreen")
            args.append(u)
            popen_kw: dict = {
                "args": args,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            before: set[int] | None = None
            if sys.platform == "win32" and win32_post_fullscreen_monitor is not None:
                before = _chrome_top_level_browser_hwnds_win32()
            subprocess.Popen(**popen_kw)
            if sys.platform == "win32" and win32_post_fullscreen_monitor is not None:
                mon = win32_post_fullscreen_monitor
                hwnd = _wait_new_chrome_hwnd_win32(before, _chrome_new_window_wait_timeout_s())
                if hwnd is not None:
                    _chrome_snap_window_to_monitor_win32(
                        hwnd,
                        mon,
                        fullscreen=fullscreen,
                        windowed_size=window_size if not fullscreen else None,
                    )
                else:
                    log.warning(
                        "Chrome: timed out waiting for new window (%s); check "
                        "CHROME_NEW_WINDOW_WAIT_S or close extra Chrome instances.",
                        label,
                    )
        else:
            log.warning("Chrome not found; opening %s in default browser.", label)
            webbrowser.open(u)
    except OSError as e:
        log.warning("Could not open %s in Chrome: %s", label, e)


def ai_agent_response(prompt: str) -> str:
    prompt = (prompt or "").strip()
    lower = prompt.lower()
    if not lower:
        return "I am here and ready to assist. Please tell me what you need."

    if any(term in lower for term in ("exit", "quit", "bye", "stop")):
        return "Goodbye. I am standing by when you need me next."

    if any(term in lower for term in ("how are you", "how have you")) or lower.startswith(("hi", "hello", "hey jarvis", "jarvis")):
        return "I am operational and ready. I can listen, think, open Claude search, play Spotify, and speak back to you."

    # LLM-backed reasoning: preferring Groq if GROQ_API_KEY present
    m = re.match(r"(?:reason about|reason:|--reason)\s+(.+)", prompt, re.IGNORECASE)
    if m:
        question = m.group(1).strip()
        return llm_reason(question)

    if "play" in lower and ("spotify" in lower or "song" in lower or "music" in lower):
        play_song(SONG_URI)
        return "Starting Spotify playback now."

    if "open" in lower and "claude" in lower:
        open_claude_in_default_browser()
        return "Opening Claude code search in your default browser."

    if "search" in lower and "claude" in lower:
        open_claude_in_default_browser()
        return "Searching for Claude code in your default browser."

    if "open" in lower and "browser" in lower:
        open_search_in_default_browser("Claude code")
        return "Opening your default browser to search for Claude code."

    if "analyze" in lower or "analysis" in lower or "summarize" in lower or "review" in lower:
        return "I can analyze your request and offer next steps. Tell me what you want reviewed or summarized."

    if "think" in lower or "agent" in lower or "reason" in lower:
        return "I am thinking through your request. Give me a specific task or topic and I will respond with an answer."

    if "help" in lower or "can you" in lower or "what can you do" in lower:
        return (
            "I can reply, analyze, think, and act. Say commands like 'play music', 'open Claude', "
            "or 'search Claude code' and I will do it in your default browser."
        )

    if "device info" in lower or "system info" in lower:
        return get_system_info()

    if (
        "list input devices" in lower
        or "list microphones" in lower
        or "list mic" in lower
        or "audio devices" in lower
    ):
        return format_input_devices()

    match = re.search(r"(?i)(?:open|show|browse) (?:path|folder|file) (.+)", prompt)
    if match:
        return open_path(match.group(1).strip())

    if "read clipboard" in lower or "clipboard read" in lower or "show clipboard" in lower:
        return _read_clipboard()

    match = re.search(r"(?i)(?:write|copy|set) clipboard(?: to)?[: ]+(.+)", prompt)
    if match:
        return _write_clipboard(match.group(1).strip())

    if "screenshot" in lower or "take screenshot" in lower:
        return take_screenshot()

    if "spotify" in lower or "song" in lower or "music" in lower:
        if "search" in lower or "find" in lower:
            open_search_in_default_browser("spotify music")
            return "Searching for Spotify music in your default browser."
        if "play" in lower:
            play_song(SONG_URI)
            return "Playing your configured Spotify track now."

    match = re.search(r"search for (.+)", lower)
    if match:
        query = match.group(1).strip()
        open_search_in_default_browser(query)
        return f"Searching for {query} in your default browser."

    if "default" in lower and "browser" in lower:
        open_search_in_default_browser("Claude code")
        return "Opening your default browser."

    # Run shell/OS command (safe mode: require confirmation)
    m = re.match(r"run command[: ]+(.+)", lower)
    if m:
        cmd = m.group(1).strip()
        aid = JARVIS_CORE.add_pending_action({"type": "run_command", "cmd": cmd})
        return f"Command queued as id={aid}. Reply 'confirm run {aid}' to execute."

    m = re.match(r"confirm run\s+(\d+)", lower)
    if m:
        aid = int(m.group(1))
        res = JARVIS_CORE.confirm_pending_action(aid)
        if not res:
            return "No pending action by that id."
        out = res.get("stdout") or res.get("stderr") or ""
        return f"Executed. rc={res.get('rc')}. output: {out[:800]}"

    # Fallback to any configured LLM for general queries (GROQ/XAI or local Ollama).
    if os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("LLM_PROVIDER"):
        return llm_reason(prompt)

    if re.match(r"(?:test groq key|test llm key|test api key)$", lower, re.IGNORECASE):
        return llm_test_key()

    return (
        "I can handle specific actions like opening Claude, playing music, or managing tasks. "
        "For general questions, set GROQ_API_KEY (or XAI_API_KEY) in your environment, "
        "then ask again or use 'reason about <question>'."
    )


def start_agent_interface() -> None:
    if not JARVIS_AGENT_ENABLED or not sys.stdin or not sys.stdin.isatty():
        return

    def _agent_loop() -> None:
        print("\nJarvis agent mode is active. Type a command and press Enter.")
        while True:
            try:
                prompt = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nJarvis: Agent mode ended.")
                break
            prompt = prompt.strip()
            if not prompt:
                continue
            response = ai_agent_response(prompt)
            print(f"Jarvis: {response}")
            if JARVIS_WELCOME_ENABLED:
                threading.Thread(target=say_text, args=(response,), daemon=True).start()
            if prompt.lower() in ("exit", "quit", "bye", "stop"):
                break

    threading.Thread(target=_agent_loop, daemon=True).start()


def _record_until_silence(max_duration_s: float = JARVIS_VOICE_COMMAND_MAX_S) -> bytes | None:
    blocksize = int(round(SAMPLE_RATE * JARVIS_VOICE_COMMAND_BLOCK_MS / 1000))
    max_blocks = max(1, int(round(max_duration_s * SAMPLE_RATE / blocksize)))
    silence_blocks_needed = max(1, int(round(JARVIS_VOICE_COMMAND_END_S * SAMPLE_RATE / blocksize)))
    collected: list[np.ndarray] = []
    in_speech = False
    silence_blocks = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            for _ in range(max_blocks):
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    log.warning("Voice capture input overflow detected.")
                level = rms_mono(data)
                if in_speech:
                    collected.append(data.copy())
                    if level < JARVIS_VOICE_COMMAND_SILENCE_THRESHOLD:
                        silence_blocks += 1
                        if silence_blocks >= silence_blocks_needed:
                            break
                    else:
                        silence_blocks = 0
                elif level >= JARVIS_VOICE_COMMAND_SILENCE_THRESHOLD:
                    in_speech = True
                    collected.append(data.copy())
                    silence_blocks = 0
            if not collected:
                return None
            audio = np.concatenate(collected, axis=0)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            pcm_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            return pcm_i16.tobytes()
    except Exception as e:
        log.warning("Voice recording failed: %s", e)
        return None


def start_voice_agent() -> None:
    if not JARVIS_VOICE_AGENT_ENABLED:
        return
    try:
        import speech_recognition as sr
    except ImportError:
        log.warning(
            "SpeechRecognition is not installed. Install it with `pip install SpeechRecognition` to use voice commands."
        )
        return

    recognizer = sr.Recognizer()
    print("\nJarvis voice mode is active. Speak a command after the prompt.")
    while True:
        try:
            print("Jarvis is listening for a voice command...")
            audio_bytes = _record_until_silence()
            if audio_bytes is None:
                print("Jarvis: I did not hear a command. Please speak again.")
                time.sleep(0.5)
                continue
            audio = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
            try:
                phrase = recognizer.recognize_google(audio)
                phrase = phrase.strip()
                if not phrase:
                    print("Jarvis: I did not catch that. Please try again.")
                    continue
                print(f"You said: {phrase}")
                response = ai_agent_response(phrase)
                print(f"Jarvis: {response}")
                if JARVIS_WELCOME_ENABLED:
                    threading.Thread(target=say_text, args=(response,), daemon=True).start()
                if phrase.lower() in ("exit", "quit", "bye", "stop"):
                    print("Jarvis: Voice agent stopped.")
                    break
            except sr.UnknownValueError:
                print("Jarvis: Sorry, I could not understand that. Please speak clearly.")
            except sr.RequestError as e:
                print(f"Jarvis: Speech recognition failed (network or service issue): {e}")
                time.sleep(5.0)
        except KeyboardInterrupt:
            print("\nJarvis: Voice agent ended.")
            break


WAKEWORD_STOP = None


def start_wakeword_listener() -> None:
    """Start a background wake-word listener that launches the voice agent on keyword."""
    global WAKEWORD_STOP
    if not JARVIS_WAKEWORD_ENABLED:
        log.info("Wakeword listener disabled (JARVIS_WAKEWORD_ENABLED=False)")
        return
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
    except Exception as e:
        log.warning("Could not initialize wakeword listener: %s", e)
        return

    def _callback(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            if JARVIS_WAKEWORD.lower() in text.lower():
                log.info("Wakeword detected: %s", text)
                # Launch a short-lived voice agent to collect a command
                threading.Thread(target=start_voice_agent, daemon=True).start()
        except Exception:
            pass

    try:
        WAKEWORD_STOP = recognizer.listen_in_background(mic, _callback)
        log.info("Wakeword listener started")
    except Exception as e:
        log.warning("Failed to start wakeword background listener: %s", e)


def open_claude_in_chrome() -> None:
    if not OPEN_CLAUDE_CODE_IN_CHROME:
        return
    url = (os.environ.get("CLAUDE_CODE_URL") or "https://claude.ai/new").strip()
    pos: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    fs = OPEN_CHROME_FULLSCREEN
    post_mon: int | None = None
    user_data: str | None = None
    if sys.platform == "win32":
        post_mon = CLAUDE_CHROME_MONITOR
        pos = _chrome_monitor_top_left(CLAUDE_CHROME_MONITOR)
        if fs:
            size = _chrome_monitor_pixel_size(CLAUDE_CHROME_MONITOR)
        else:
            size = _chrome_window_size()
        if CHROME_SEPARATE_SITE_PROFILES:
            user_data = _chrome_site_user_data_dir("claude")
    elif not fs:
        size = _chrome_window_size()
    else:
        size = None
    _open_url_in_chrome(
        url,
        new_window=True,
        label="Claude",
        window_position=pos,
        window_size=size,
        fullscreen=fs,
        win32_post_fullscreen_monitor=post_mon,
        user_data_dir=user_data,
    )


def open_binance_btc_in_chrome() -> None:
    if not OPEN_BINANCE_BTC_IN_CHROME:
        return
    url = (
        os.environ.get("TASARADAR_URL")
        or os.environ.get("BINANCE_BTC_URL")
        or "https://www.binance.com/en/trade/BTC_USDT"
    ).strip()
    pos: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    fs = OPEN_CHROME_FULLSCREEN
    post_mon: int | None = None
    user_data: str | None = None
    if sys.platform == "win32":
        post_mon = BINANCE_CHROME_MONITOR
        pos = _chrome_monitor_top_left(BINANCE_CHROME_MONITOR)
        if fs:
            size = _chrome_monitor_pixel_size(BINANCE_CHROME_MONITOR)
        else:
            size = _chrome_window_size()
        if CHROME_SEPARATE_SITE_PROFILES:
            user_data = _chrome_site_user_data_dir("binance")
    elif not fs:
        size = _chrome_window_size()
    else:
        size = None
    _open_url_in_chrome(
        url,
        new_window=True,
        label="Binance BTC",
        window_position=pos,
        window_size=size,
        fullscreen=fs,
        win32_post_fullscreen_monitor=post_mon,
        user_data_dir=user_data,
    )


def _cursor_executable() -> str | None:
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        for sub in ("Programs\\cursor\\Cursor.exe", "Programs\\Cursor\\Cursor.exe"):
            if local:
                p = os.path.join(local, *sub.split("\\"))
                if os.path.isfile(p):
                    return p
    return shutil.which("cursor")


def _cursor_largest_main_hwnd_win32() -> int | None:
    """Largest top-level Cursor.exe window (visible or minimized)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    GW_OWNER = 4
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    candidates: list[tuple[int, wintypes.HWND]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: wintypes.HWND, _lp: wintypes.LPARAM) -> bool:
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        if not user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return True
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        try:
            buf = ctypes.create_unicode_buffer(4096)
            sz = wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                return True
            exe_path = buf.value
        finally:
            kernel32.CloseHandle(hproc)
        if os.path.basename(exe_path).lower() != "cursor.exe":
            return True
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < 200 or h < 200:
            return True
        candidates.append((w * h, hwnd))
        return True

    user32.EnumWindows(_enum, 0)
    if not candidates:
        return None
    return int(max(candidates, key=lambda t: t[0])[1])


def _cursor_foreground_hwnd_win32(hwnd: int) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    fg = user32.GetForegroundWindow()
    tid_tgt = user32.GetWindowThreadProcessId(hwnd, None)
    tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    if tid_fg and tid_tgt:
        user32.AttachThreadInput(tid_fg, tid_tgt, True)
    user32.SetForegroundWindow(hwnd)
    if tid_fg and tid_tgt:
        user32.AttachThreadInput(tid_fg, tid_tgt, False)


def _cursor_send_f11_fullscreen_win32(hwnd: int) -> None:
    """F11 toggles Zen/fullscreen in Cursor (Electron)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    KEYEVENTF_KEYUP = 0x0002
    VK_F11 = 0x7A
    _cursor_foreground_hwnd_win32(hwnd)
    user32.keybd_event(VK_F11, 0, 0, 0)
    user32.keybd_event(VK_F11, 0, KEYEVENTF_KEYUP, 0)


def _focus_existing_cursor_window_win32() -> bool:
    """Bring an existing Cursor.exe main window to the foreground (no new process)."""
    if sys.platform != "win32":
        return False
    hwnd = _cursor_largest_main_hwnd_win32()
    if hwnd is None:
        return False
    _cursor_foreground_hwnd_win32(hwnd)
    return True


def run_double_clap_actions() -> None:
    """Run outside the mic loop so sleeps do not stall capture."""
    play_song(SONG_URI)
    if JARVIS_WELCOME_ENABLED and JARVIS_WELCOME_PHRASE.strip():
        def _delayed_welcome() -> None:
            time.sleep(max(0.0, JARVIS_AFTER_SONG_DELAY_S))
            say_jarvis_welcome()

        threading.Thread(target=_delayed_welcome, daemon=True).start()
    open_claude_in_default_browser()
    open_cursor_window()


def open_cursor_window() -> None:
    if not FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP and not OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
        return
    exe = _cursor_executable()
    if not exe:
        log.warning(
            "Could not find Cursor (install app or add the `cursor` command to PATH)."
        )
        return
    popen_kw: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        if FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP:
            focused = (
                sys.platform == "win32" and _focus_existing_cursor_window_win32()
            )
            if not focused:
                subprocess.Popen([exe], **popen_kw)
        if OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
            subprocess.Popen([exe, "-n"], **popen_kw)
    except OSError as e:
        log.warning("Could not start or focus Cursor: %s", e)
        return
    if sys.platform == "win32" and CURSOR_OPEN_FULLSCREEN:
        time.sleep(0.5)
        hwnd = _cursor_largest_main_hwnd_win32()
        if hwnd is not None:
            _cursor_send_f11_fullscreen_win32(hwnd)
        else:
            log.warning("Cursor fullscreen: no Cursor window found to send F11.")


def main() -> int:
    blocksize = block_samples()
    noise_floor = 1e-4
    last_logged_double = 0.0
    first_clap_time: float | None = None
    spike_armed = True
    welcome_sequence_done = False

    log.info(
        "Listening (double clap: %.2f–%.2fs apart, rate=%d, block=%d ms, "
        "spike_ratio=%.1f, cooldown=%.2fs). Ctrl+C to stop.",
        MIN_DOUBLE_GAP_S,
        MAX_DOUBLE_GAP_S,
        SAMPLE_RATE,
        BLOCK_MS,
        SPIKE_RATIO,
        COOLDOWN_S,
    )
    if SONG_URI.strip():
        log.info("Double clap opens this track: %s", SONG_URI.strip())
    else:
        log.info("SONG_URI is empty — set it to play one song on each double clap.")
    if FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP:
        log.info(
            "Double clap will foreground an existing Cursor window (Windows API); "
            "falls back to launching Cursor if none is running."
        )
    if OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
        log.info("Double clap will also open a new Cursor window (-n).")
    if CURSOR_OPEN_FULLSCREEN and sys.platform == "win32":
        log.info("Cursor will be sent F11 for fullscreen after focus/launch.")
    if OPEN_CLAUDE_CODE_IN_CHROME:
        cu = (os.environ.get("CLAUDE_CODE_URL") or "https://claude.ai/new").strip()
        log.info(
            "After Spotify, open Claude in Chrome%s on monitor %d: %s",
            " fullscreen" if OPEN_CHROME_FULLSCREEN else "",
            CLAUDE_CHROME_MONITOR,
            cu,
        )
    if OPEN_BINANCE_BTC_IN_CHROME:
        bu = (
            os.environ.get("BINANCE_BTC_URL")
            or "https://www.binance.com/en/trade/BTC_USDT"
        ).strip()
        log.info(
            "After Spotify, open Binance BTC in Chrome%s on monitor %d: %s",
            " fullscreen" if OPEN_CHROME_FULLSCREEN else "",
            BINANCE_CHROME_MONITOR,
            bu,
        )
    if JARVIS_WELCOME_ENABLED:
        ev, em, ef, er = elevenlabs_env_config()
        log.info(
            "After song + %.2fs: %r (ElevenLabs voice=%s, model=%s, format=%s, pcm_rate=%d)",
            JARVIS_AFTER_SONG_DELAY_S,
            JARVIS_WELCOME_PHRASE.strip(),
            ev or "(unset)",
            em,
            ef,
            er,
        )

    if JARVIS_AGENT_ENABLED:
        log.info("Jarvis agent mode is active. Type a command and press Enter.")
        start_agent_interface()

    if JARVIS_VOICE_AGENT_ENABLED:
        log.info("Jarvis voice agent is enabled. Listening for spoken commands.")
        threading.Thread(target=start_voice_agent, daemon=True).start()

    input_idx = _choose_input_device(blocksize)

    try:
        with sd.InputStream(
            device=input_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            while True:
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    log.warning("Input overflow; try a larger BLOCK_MS")

                level = rms_mono(data)

                quiet_gate = noise_floor * QUIET_GATE_MULT
                if level < quiet_gate:
                    noise_floor = NOISE_FLOOR_ALPHA * noise_floor + (
                        1.0 - NOISE_FLOOR_ALPHA
                    ) * level
                    noise_floor = max(noise_floor, 1e-7)

                threshold = max(noise_floor * SPIKE_RATIO, MIN_RMS)
                now = time.monotonic()
                retrigger_level = threshold * RETRIGGER_RATIO

                if level < retrigger_level:
                    spike_armed = True

                if (
                    spike_armed
                    and level >= threshold
                    and (now - last_logged_double) >= COOLDOWN_S
                ):
                    spike_armed = False
                    if first_clap_time is None:
                        first_clap_time = now
                    else:
                        gap = now - first_clap_time
                        if gap < MIN_DOUBLE_GAP_S:
                            pass
                        elif gap <= MAX_DOUBLE_GAP_S:
                            first_clap_time = None
                            last_logged_double = now
                            if not welcome_sequence_done:
                                welcome_sequence_done = True
                                log.info(
                                    "Double clap detected (gap=%.3fs, rms=%.5f, "
                                    "noise_floor=%.5f, threshold=%.5f) — running welcome once",
                                    gap,
                                    level,
                                    noise_floor,
                                    threshold,
                                )
                                threading.Thread(
                                    target=run_double_clap_actions, daemon=True
                                ).start()
                        else:
                            first_clap_time = now

    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0
    except sd.PortAudioError as e:
        log.error("Audio error: %s", e)
        log.error("If PortAudio fails, install/repair drivers or try another SAMPLE_RATE.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
