import ast
import json
import operator
import re
from functools import lru_cache
from io import BytesIO

import requests
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# --------------------------------------------------------------------------- #
#                       ARITHMETIC (SAFE CALCULATOR)                         #
# --------------------------------------------------------------------------- #
_ALLOWED_AST_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float | int | complex:
    """Recursively evaluate a *restricted* AST expression tree."""
    if isinstance(node, ast.Constant):
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_AST_OPS:
        return _ALLOWED_AST_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_AST_OPS:
        return _ALLOWED_AST_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    raise ValueError("Unsafe or unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate basic arithmetic expressions (no variables, functions)."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree.body)
        return str(value)
    except Exception as exc:  # pragma: no cover – we surface errors to the agent
        return f"calc_error:{exc}"


# --------------------------------------------------------------------------- #
#                             WEB  &  WIKI  SEARCH                           #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=256)
def _ddg_search(query: str, k: int = 6) -> list[dict[str, str]]:
    """Cached DuckDuckGo JSON search."""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=k)
    hits = wrapper.results(query)
    return [
        {
            "title": hit.get("title", "")[:120],
            "snippet": hit.get("snippet", "")[:300],
            "link": hit.get("link", "")[:200],
        }
        for hit in hits[:k]
    ]


@tool
def web_multi_search(query: str, k: int = 6) -> str:
    """Run DuckDuckGo → Tavily fallback search. Returns JSON list[dict]."""
    try:
        hits = _ddg_search(query, k)
        if hits:
            return json.dumps(hits, ensure_ascii=False)
    except Exception:  # fall through to Tavily
        pass

    try:
        tavily_hits = TavilySearchResults(max_results=k).invoke(query=query)
        print(
            f"[TOOL] TAVILY search is triggered with following response: {tavily_hits}"
        )
        formatted = [
            {
                "title": d.metadata.get("title", "")[:120],
                "snippet": d.page_content[:300],
                "link": d.metadata.get("source", "")[:200],
            }
            for d in tavily_hits
        ]
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as exc:
        return f"search_error:{exc}"


@tool
def wiki_search(query: str, max_pages: int = 2) -> str:
    """Lightweight wrapper on WikipediaLoader; returns concatenated page texts."""
    print(f"[TOOL] wiki_search called with query: {query}")
    docs = WikipediaLoader(query=query, load_max_docs=max_pages).load()
    joined = "\n\n---\n\n".join(d.page_content for d in docs)
    return joined[:8_000]  # simple guardrail – stay within context window


# --------------------------------------------------------------------------- #
#                               YOUTUBE  TRANSCRIPT                          #
# --------------------------------------------------------------------------- #


@tool
def youtube_transcript(url: str, chars: int = 10_000) -> str:
    """Fetch full YouTube transcript (first *chars* characters)."""
    video_id_match = re.search(r"[?&]v=([A-Za-z0-9_\-]{11})", url)
    if not video_id_match:
        return "yt_error:id_not_found"
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id_match.group(1))
        text = " ".join(piece["text"] for piece in transcript)
        return text[:chars]
    except Exception as exc:
        return f"yt_error:{exc}"


# --------------------------------------------------------------------------- #
#                                IMAGE  DESCRIPTION                           #
# --------------------------------------------------------------------------- #

# Instantiate a lightweight CLIP‑based zero‑shot image classifier (runs on CPU)
### The model 'openai/clip-vit-base-patch32' is a vision transformer (ViT) model trained as part of OpenAI’s CLIP project.
### It performs zero-shot image classification by mapping images and labels into the same embedding space.
_image_pipe = pipeline(
    "image-classification", model="openai/clip-vit-base-patch32", device="cpu"
)


@tool
def image_describe(image_url: str, top_k: int = 3) -> str:
    """Download an image and return top-k labels using CLIP zero-shot classification."""
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        labels = _image_pipe(BytesIO(resp.content))[:top_k]
        return ", ".join(f"{d['label']} ({d['score']:.2f})" for d in labels)
    except Exception as exc:
        return f"img_error:{exc}"


# --------------------------------------------------------------------------- #
#                                 FILE  UTILS                                 #
# --------------------------------------------------------------------------- #


@tool
def csv_sum(url: str, column: str) -> str:
    """Download a CSV and return the sum of the specified numeric column."""
    try:
        import pandas as pd  # local import to avoid mandatory pandas if unused

        df = pd.read_csv(url)
        total = df[column].sum()
        return str(total)
    except Exception as exc:
        return f"csv_error:{exc}"


__all__ = [
    "calculator",
    "web_multi_search",
    "wiki_search",
    "youtube_transcript",
    "image_describe",
    "csv_sum",
]
