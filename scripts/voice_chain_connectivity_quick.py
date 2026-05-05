import json
import time
import urllib.request

import websocket

BASE_HTTP = "http://localhost:8080"

def http_post_json(url: str, payload: dict, timeout_s: int = 30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)

def http_delete(url: str, timeout_s: int = 10):
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        _ = resp.read()


def main():
    # Allow the wait time to be controlled by env vars so video chunks can be observed.
    duration_s = int(__import__("os").environ.get("VOICE_TEST_DURATION_S", "180"))

    # Create voice_llm session
    resp = http_post_json(BASE_HTTP + "/api/v1/sessions", {"mode": "voice_llm"})
    session_id = resp.get("session_id")
    if not session_id:
        raise RuntimeError(f"missing session_id: {resp}")

    ws_url = f"ws://localhost:8080/ws/chat/{session_id}"
    print(f"[voice] session_id={session_id}")
    print(f"[voice] connecting ws: {ws_url}")

    ws = websocket.create_connection(ws_url, timeout=10)
    ws.settimeout(5)

    got_speaking = False
    got_processing = False
    got_error = None
    got_transcript = 0
    got_status = set()

    started = time.time()
    try:
        while time.time() - started < duration_s:
            try:
                raw = ws.recv()
            except Exception:
                continue
            if not raw:
                continue
            data = json.loads(raw)
            mtype = data.get("type")
            if mtype == "error":
                got_error = data.get("message") or data
                print(f"[voice][error] {got_error}")
                break
            if mtype == "avatar_status":
                st = data.get("status")
                if st:
                    got_status.add(st)
                if st == "speaking":
                    got_speaking = True
                if st == "processing":
                    got_processing = True
                print(f"[voice][status] {data}")
            if mtype == "transcript":
                got_transcript += 1
            # If we have gone through speaking->idle, it's a strong hint
            if got_speaking and "idle" in got_status:
                break
    finally:
        try:
            ws.close()
        except Exception:
            pass

    # Cleanup
    try:
        http_delete(BASE_HTTP + f"/api/v1/sessions/{session_id}")
    except Exception:
        pass

    print("[voice] done:", {
        "got_processing": got_processing,
        "got_speaking": got_speaking,
        "got_status": sorted(got_status),
        "got_transcript_messages": got_transcript,
        "got_error": got_error,
        "duration_s": duration_s,
    })


if __name__ == "__main__":
    main()
