import requests


def fetch_task_file(api_url: str, task_id: str) -> tuple[bytes, str]:
    """
    Returns (file_bytes, content_type) or (b'', '') if no attachment found.
    Follows any redirect the endpoint issues.
    """
    url = f"{api_url}/files/{task_id}"
    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
    except requests.RequestException as e:
        print(f"[DEBUG] GET {url} failed → {e}")
        return b"", ""
    if r.status_code != 200:
        print(f"[DEBUG] GET {url} → {r.status_code}")
        return b"", ""
    return r.content, r.headers.get("content-type", "").lower()
