import ast
import json
import operator
import re
import subprocess
from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from helpers import get_prompt

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
    except Exception as exc:
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
            "title": hit.get("title", "")[:500],
            "snippet": hit.get("snippet", "")[:750],
            "link": hit.get("link", "")[:300],
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
                "title": d.metadata.get("title", "")[:500],
                "snippet": d.page_content[:750],
                "link": d.metadata.get("source", "")[:300],
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
# _image_pipe = pipeline(
#     "image-classification", model="openai/clip-vit-base-patch32", device="cpu"
# )

# @tool
# def image_describe(img_bytes: bytes, top_k: int = 3) -> str:
#     """Return the top-k CLIP labels for an image supplied as raw bytes.

#     typical result for a random cat photo can be:
#     [
#         {'label': 'tabby, tabby cat', 'score': 0.41},
#         {'label': 'tiger cat', 'score': 0.24},
#         {'label': 'Egyptian cat', 'score': 0.22}
#     ]
#     """

#     try:
#         labels = _image_pipe(BytesIO(img_bytes))[:top_k]
#         return ", ".join(f"{d['label']} (score={d['score']:.2f})" for d in labels)
#     except Exception as exc:
#         return f"img_error:{exc}"


@tool
def vision_task(img_bytes: bytes, question: str) -> str:
    """
    Pass the user's question AND the referenced image to a multimodal LLM and
    return its first line of text as the answer.  No domain assumptions made.
    """
    vision_llm = ChatOpenAI(
        model="gpt-4o-mini",  # set OPENAI_API_KEY in env
        temperature=0,
        max_tokens=64,
    )
    try:
        b64 = b64encode(img_bytes).decode()
        messages = [
            SystemMessage(content=get_prompt(prompt_key="vision_system")),
            HumanMessage(
                content=[
                    {"type": "text", "text": question.strip()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            ),
        ]
        reply = vision_llm.invoke(messages).content.strip()
        return reply
    except Exception as exc:
        return f"img_error:{exc}"


# --------------------------------------------------------------------------- #
#                                 FILE  UTILS                                 #
# --------------------------------------------------------------------------- #
@tool
def run_py(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return last stdout line."""
    try:
        with NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
            f.write(code)
            path = f.name
        proc = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=30
        )
        out = proc.stdout.strip().splitlines()
        return out[-1] if out else ""
    except Exception as exc:
        return f"py_error:{exc}"


@tool
def transcribe_via_whisper(audio_bytes: bytes) -> str:
    """Transcribe audio with Whisper (CPU)."""
    with NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        import whisper  # openai-whisper

        model = whisper.load_model("base")
        output = model.transcribe(path)["text"].strip()
        print(f"[DEBUG] Whisper transcript (first 200 chars): {output[:200]}")
        return output
    except Exception as exc:
        return f"asr_error:{exc}"


@tool
def analyze_excel_file(xls_bytes: bytes, question: str) -> str:
    "Analyze Excel or CSV file by passing the data preview to LLM and getting the Python Pandas operation to run"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=64)

    try:
        df = pd.read_excel(BytesIO(xls_bytes))
    except Exception:
        df = pd.read_csv(BytesIO(xls_bytes))

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].astype(float)

    # Ask the LLM for a single expression
    prompt = get_prompt(
        prompt_key="excel_system",
        question=question,
        preview=df.head(5).to_dict(orient="list"),
    )
    expr = llm.invoke(prompt).content.strip()

    # Run generated Pandas' one-line expression
    try:
        result = eval(expr, {"df": df, "pd": pd, "__builtins__": {}})
        # Normalize scalars to string
        if isinstance(result, np.generic):
            result = float(result)  # → plain Python float
            return f"{result:.2f}"  # or str(result) if no decimals needed

        # DataFrame / Series → single-line string
        return (
            result.to_string(index=False)
            if hasattr(result, "to_string")
            else str(result)
        )
    except Exception as e:
        return f"eval_error:{e}"


__all__ = [
    "calculator",
    "web_multi_search",
    "wiki_search",
    "youtube_transcript",
    "vision_task",
    "run_py",
    "transcribe_via_whisper",
    "analyze_excel_file",
]
