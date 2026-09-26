"""
Search SERP — raw web skill.

The default backend is Serper when SERPER_API_KEY is set. Otherwise this falls
back to the legacy Novada/SERP endpoint configured by SERP_DEV_KEY.

Serper:
POST https://google.serper.dev/search
Headers:
    X-API-KEY: <SERPER_API_KEY>
    Content-Type: application/json
Body:
    q:      <query>
    num:    <int, 1-10>
    hl:     "zh" | "en"  (auto-detected from query)
    gl:     "cn" | "us"  (auto-detected from query)
    page:   <int, 1-based page number>

Legacy Novada/SERP:
GET https://scraperapi.novada.com/search
Query params:
    engine:     "google"
    api_key:    <SERP_DEV_KEY>
    q:          <query>
    num:        <str int, 1-10>
    hl:         "zh" | "en"  (auto-detected from query)
    gl:         "cn" | "us"  (auto-detected from query)
    start:      <int, 0-based offset>
    fetch_mode: "static"
    no_cache:   "true"

Input:  query (str), timeout (int), num (int), start (int)
Output: {"status": <int>, "output": <list[dict]>}
"""

import os
import re
import requests

SERP_API_URL = os.getenv("SERP_API_URL", "https://scraperapi.novada.com/search")
SERP_DEV_KEY = os.getenv("SERP_DEV_KEY", "YOUR_API_KEY")
SERPER_API_URL = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")


def _detect_language(query: str) -> tuple[str, str]:
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh", "cn"
    return "en", "us"


def _save_raw_response(raw_save_path: str | None, text: str) -> None:
    if not raw_save_path:
        return
    os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
    with open(raw_save_path, "w", encoding="utf-8") as f:
        f.write(text)


def _normalize_serper_result(item: dict, query: str) -> dict:
    return {
        "title": item.get("title", ""),
        "link": item.get("link", ""),
        "snippet": item.get("snippet", ""),
        "date": item.get("date", ""),
        "query": query,
    }


def _normalize_novada_result(item: dict, query: str) -> dict:
    return {
        "title": item.get("title", ""),
        "link": item.get("url", ""),
        "snippet": item.get("description", ""),
        "date": item.get("date", ""),
        "query": query,
    }


def _search_serper(
    query: str,
    timeout: int,
    num: int,
    start: int,
    raw_save_path: str | None,
) -> dict:
    hl, gl = _detect_language(query)
    page = max(((max(start, 1) - 1) // max(num, 1)) + 1, 1)
    payload = {
        "q": query,
        "num": min(max(num, 1), 10),
        "hl": hl,
        "gl": gl,
        "page": page,
    }
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(SERPER_API_URL, json=payload, headers=headers, timeout=timeout)
    if resp.status_code == 200:
        _save_raw_response(raw_save_path, resp.text)
    if resp.status_code != 200:
        return {
            "status": resp.status_code,
            "output": [],
            "provider": "serper",
            "error": resp.text[:200],
        }

    data = resp.json()
    results = [_normalize_serper_result(item, query) for item in data.get("organic", [])]
    return {"status": resp.status_code, "output": results, "provider": "serper"}


def _search_novada(
    query: str,
    timeout: int,
    num: int,
    start: int,
    raw_save_path: str | None,
) -> dict:
    hl, gl = _detect_language(query)
    params = {
        "engine": "google",
        "api_key": SERP_DEV_KEY,
        "q": query,
        "num": str(min(max(num, 1), 10)),
        "hl": hl,
        "gl": gl,
        "start": str(max(start, 1)),
        "fetch_mode": "static",
        "no_cache": "true",
    }
    resp = requests.get(SERP_API_URL, params=params, timeout=timeout)
    if resp.status_code == 200:
        _save_raw_response(raw_save_path, resp.text)
    if resp.status_code != 200:
        return {"status": resp.status_code, "output": [], "provider": "novada"}

    body = resp.json()
    if "data" not in body and (body.get("code") or body.get("msg")):
        return {
            "status": resp.status_code,
            "output": [],
            "provider": "novada",
            "error": f"{body.get('code')}: {body.get('msg')}",
        }

    data = body.get("data", {})
    results = [_normalize_novada_result(item, query) for item in data.get("organic_results", [])]
    return {"status": resp.status_code, "output": results, "provider": "novada"}


def search_serp(
    query: str,
    timeout: int = 20,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google and return extracted results.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset.

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on error.
            output (list[dict]): List of result dicts with keys:
                title, link, snippet, date, query.
    """
    try:
        if SERPER_API_KEY:
            return _search_serper(query, timeout, num, start, raw_save_path)
        return _search_novada(query, timeout, num, start, raw_save_path)
    except Exception as e:
        provider = "serper" if SERPER_API_KEY else "novada"
        return {"status": -1, "output": [], "provider": provider, "error": str(e)[:200]}


if __name__ == "__main__":
    import json

    result = search_serp("Python web scraping", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:1000])
