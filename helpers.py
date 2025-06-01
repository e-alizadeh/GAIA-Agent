import csv
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import requests
from yaml import safe_load

CURRENT_DIR = Path(__file__).parent

_PROMPTS = safe_load(CURRENT_DIR.joinpath("prompts.yaml").read_text())


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


def sniff_excel_type(blob: bytes) -> str:
    """
    Return one of 'xlsx', 'xls', 'csv', or '' (unknown) given raw bytes.
    """
    # 1️⃣ XLSX / XLSM / ODS  (ZIP container)
    if blob[:4] == b"PK\x03\x04":
        try:
            with ZipFile(BytesIO(blob)) as zf:
                names = set(zf.namelist())
                if {"xl/workbook.xml", "[Content_Types].xml"} & names:
                    return "xlsx"
        except BadZipFile:
            pass  # fall through

    # 2️⃣ Legacy XLS (OLE Compound File)
    if blob[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "xls"

    # 3️⃣ Text-like -> CSV/TSV
    try:
        sample = blob[:1024].decode("utf-8", "ignore")
        first_line = sample.splitlines()[0]
        if any(sep in first_line for sep in (",", ";", "\t")):
            # Confirm via csv.Sniffer to avoid random text
            csv.Sniffer().sniff(sample)
            return "csv"
    except (UnicodeDecodeError, csv.Error):
        pass

    return ""


def get_prompt(prompt_key: str, **kwargs: str) -> str:
    """Get a prompt by key and fill in placeholders via `.format(**kwargs)`"""
    return _PROMPTS[prompt_key].format(**kwargs)
