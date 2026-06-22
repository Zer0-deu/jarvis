"""Comprehensive diagnostics for GROQ DNS/connectivity issues.

Run this on the machine where Jarvis is failing. It performs local checks and
prints actionable suggestions.
"""
from __future__ import annotations

import os
import platform
import socket
import shutil
import subprocess
import sys
import time


def run() -> None:
    print("===== GROQ/XAI Connectivity Diagnosis =====")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.splitlines()[0]}")
    print()

    # Environment variables
    print("-- Environment variables (relevant)")
    for k in ("GROQ_API_KEY", "XAI_API_KEY", "LLM_PROVIDER", "OLLAMA_HTTP_URL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        print(f"{k}={os.environ.get(k)!r}")
    print()

    host = "api.groq.x.ai"
    url = "https://api.groq.x.ai/v1"

    # Hosts file check (Windows and Unix)
    print("-- Hosts file entries (if accessible)")
    hosts_paths = []
    if sys.platform.startswith("win"):
        hosts_paths = [r"C:\\Windows\\System32\\drivers\\etc\\hosts"]
    else:
        hosts_paths = ["/etc/hosts"]
    for p in hosts_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = f.read()
            print(f"Contents of {p} (showing lines that mention '{host}'):")
            for i, line in enumerate(data.splitlines(), start=1):
                if host in line:
                    print(f"{i:4d}: {line}")
        except Exception as e:
            print(f"Could not read {p}: {e}")
    print()

    # Socket resolution
    print("-- Socket.getaddrinfo resolution")
    try:
        ai = socket.getaddrinfo(host, 443)
        for item in ai:
            print(item)
    except Exception as e:
        print(f"getaddrinfo failed: {e}")
    print()

    # nslookup / dig
    print("-- DNS tools (nslookup / dig if available)")
    nslookup = shutil.which("nslookup")
    dig = shutil.which("dig") or shutil.which("host")
    if nslookup:
        try:
            out = subprocess.check_output([nslookup, host], stderr=subprocess.STDOUT, timeout=10, text=True)
            print("nslookup result:\n" + out)
        except Exception as e:
            print(f"nslookup failed: {e}")
    else:
        print("nslookup not found on PATH")
    if dig:
        try:
            out = subprocess.check_output([dig, host], stderr=subprocess.STDOUT, timeout=10, text=True)
            print(f"{dig} result:\n" + out)
        except Exception as e:
            print(f"{dig} failed: {e}")
    else:
        print("dig/host not found on PATH")
    print()

    # nslookup against public resolvers (Google and Cloudflare)
    print("-- nslookup against public DNS servers (8.8.8.8, 1.1.1.1)")
    if nslookup:
        for dns in ("8.8.8.8", "1.1.1.1"):
            try:
                out = subprocess.check_output([nslookup, host, dns], stderr=subprocess.STDOUT, timeout=10, text=True)
                print(f"nslookup {host} {dns}:\n{out}")
            except Exception as e:
                print(f"nslookup {dns} failed: {e}")
    else:
        print("nslookup not available to query alternate resolvers.")
    print()

    # traceroute / tracert
    print("-- Tracepath/Traceroute")
    if sys.platform.startswith("win"):
        tracert = shutil.which("tracert")
        if tracert:
            try:
                out = subprocess.check_output([tracert, host], stderr=subprocess.STDOUT, timeout=30, text=True)
                print("tracert result:\n" + out)
            except Exception as e:
                print(f"tracert failed: {e}")
        else:
            print("tracert not found on PATH")
    else:
        traceroute = shutil.which("traceroute") or shutil.which("tracepath")
        if traceroute:
            try:
                out = subprocess.check_output([traceroute, host], stderr=subprocess.STDOUT, timeout=30, text=True)
                print(f"{traceroute} result:\n" + out)
            except Exception as e:
                print(f"{traceroute} failed: {e}")
        else:
            print("traceroute/tracepath not found on PATH")
    print()

    # powershell Test-NetConnection (Windows) for TCP port 443
    if sys.platform.startswith("win") and shutil.which("powershell.exe"):
        print("-- PowerShell Test-NetConnection for TCP port 443")
        try:
            cmd = ["powershell.exe", "-NoProfile", "-Command", f"Test-NetConnection -ComputerName {host} -Port 443 | ConvertTo-Json"]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15, text=True)
            print(out)
        except Exception as e:
            print(f"PowerShell Test-NetConnection failed: {e}")
        print()

    # ipconfig / resolvectl
    print("-- Local DNS configuration / resolvers")
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(["ipconfig", "/all"], stderr=subprocess.STDOUT, timeout=15, text=True)
            print(out)
        else:
            resolv = shutil.which("resolvectl") or shutil.which("systemd-resolve")
            if resolv:
                out = subprocess.check_output([resolv, "status"], stderr=subprocess.STDOUT, timeout=15, text=True)
                print(out)
            else:
                out = subprocess.check_output(["cat", "/etc/resolv.conf"], stderr=subprocess.STDOUT, timeout=5, text=True)
                print(out)
    except Exception as e:
        print(f"Could not collect resolver info: {e}")
    print()

    # HTTP(S) request via requests if available
    print("-- Attempt HTTPS request using Python requests")
    try:
        import requests

        try:
            r = requests.post(url, json={"prompt": "hello"}, timeout=8)
            print(f"HTTP {r.status_code} response headers:\n{r.headers}\nBody:\n{r.text[:400]}")
        except Exception as e:
            print(f"requests.post failed: {e}")
    except Exception:
        print("requests not installed in this Python environment.")
    print()

    # final suggestions
    print("-- Suggestions:")
    print("1) If getaddrinfo/nslookup fail, check your DNS server or try a public DNS (8.8.8.8 / 1.1.1.1).")
    print("2) If hosts file contains api.groq.x.ai pointing to 0.0.0.0/127.0.0.1, remove that entry (requires admin).")
    print("3) If you're on a VPN, proxy, or corporate network, try disabling it or switch networks.")
    print("4) If DNS works but HTTPS fails, check firewall or outbound port 443 blocking.")
    print("5) As a workaround, use a local LLM (OLLAMA) and set LLM_PROVIDER=OLLAMA in your .env.")
    print()


if __name__ == "__main__":
    run()
