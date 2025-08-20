import os
import json
import time
import requests
import psycopg
from psycopg.rows import dict_row
from typing import List, Dict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse, PlainTextResponse

# --- Config (env-first, with sane defaults matching your CLI) ---
PG_HOST = os.getenv("PG_HOST", "5.78.72.110")
PG_USER = os.getenv("PG_USER", "dylan25")
PG_DB   = os.getenv("PG_DB",   "medical_db")
PG_PASS = os.getenv("PG_PASS",  "")

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "3.135.119.211")
OLLAMA_PORT     = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL      = os.getenv("CHAT_MODEL",  "llama2")
IVFFLAT_PROBES  = os.getenv("IVFFLAT_PROBES")  # e.g., "10"

DEFAULT_TOP_K   = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT     = int(os.getenv("MAX_CONTEXT", "8000"))

# --- FastAPI app & CORS ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB connect (simple sync for MVP). In prod consider pgbouncer. ---
_conn = None

def get_conn():
    """Singleton psycopg3 connection with optional ivfflat probes setting."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(
            host=PG_HOST,
            user=PG_USER,
            dbname=PG_DB,
            password=PG_PASS,
            autocommit=True,
        )
        if IVFFLAT_PROBES:
            try:
                with _conn.cursor() as cur:
                    cur.execute("SET ivfflat.probes = %s;", (int(IVFFLAT_PROBES),))
            except Exception as e:
                print(f"[warn] could not set ivfflat.probes: {e}")
    return _conn

# --- Helpers copied from your CLI, slightly adapted ---

def ollama_url(path: str) -> str:
    return f"http://{OLLAMA_HOST}:{OLLAMA_PORT}{path}"

def embed_query(text: str, model: str) -> List[float]:
    payload = {"model": model, "prompt": text}
    r = requests.post(ollama_url("/api/embeddings"), json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise ValueError(f"Unexpected embeddings response: {data}")
    return emb

def to_vector_literal(vec: List[float]) -> str:
    # pgvector expects like '[0.12,-0.34,...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def retrieve_chunks(query_vec: List[float], k: int) -> List[Dict]:
    conn = get_conn()
    vec_lit = to_vector_literal(query_vec)
    sql = """
        SELECT
            book_id,
            chunk_id,
            content,
            (1.0 - (embeddings <=> %s::vector)) AS cosine_sim
        FROM book_vector
        ORDER BY embeddings <=> %s::vector
        LIMIT %s;
    """
    # psycopg3: use row_factory=dict_row to get dict-like rows
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (vec_lit, vec_lit, k))
        return cur.fetchall()

def build_context(snippets: List[Dict], max_chars: int):
    parts, cites = [], []
    total = 0
    for row in snippets:
        tag = f"{row['book_id']}#{row['chunk_id']}"
        block = f"[{tag}] {row['content'].strip()}"
        if total + len(block) > max_chars:
            block = block[: max(0, max_chars - total)]
        parts.append(block)
        cites.append(tag)
        total += len(block)
        if total >= max_chars:
            break
    return "\n\n".join(parts), cites

SYSTEM_INSTRUCTIONS = (
    "You are a careful medical assistant. Answer ONLY using the provided context from medical books. "
    "If the answer is not present or is uncertain, say you don't know based on the provided sources. "
    "Keep the explanation very concise."
)

def sse_format(event: str, data: str) -> str:
    # Basic Server-Sent Event line formatting
    return f"event: {event}\ndata: {data}\n\n"

@app.get("/healthz")
def healthz():
    try:
        get_conn()
        return PlainTextResponse("ok")
    except Exception as e:
        return PlainTextResponse(f"unhealthy: {e}", status_code=500)

@app.get("/api/source")
def get_source(book_id: str = Query(...), chunk_id: int = Query(...)):
    conn = get_conn()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT book_id, chunk_id, content FROM book_vector WHERE book_id = %s AND chunk_id = %s;",
            (book_id, chunk_id),
        )
        row = cur.fetchone()
        if not row:
            return JSONResponse({"error": "not found"}, status_code=404)
        # Optionally load neighbors for ±1 context
        cur.execute(
            "SELECT chunk_id, content FROM book_vector WHERE book_id = %s AND chunk_id IN (%s, %s, %s) ORDER BY chunk_id;",
            (book_id, chunk_id - 1, chunk_id, chunk_id + 1),
        )
        neighbors = cur.fetchall()
        return JSONResponse({"book_id": book_id, "focus": row, "neighbors": neighbors})

@app.get("/api/ask")
def ask(
    question: str = Query(..., min_length=1, max_length=2000),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
    max_context: int = Query(MAX_CONTEXT, ge=1000, le=20000),
    chat_model: str = Query(CHAT_MODEL),
    embed_model: str = Query(EMBED_MODEL),
):
    """SSE stream using GET query params (so the frontend can use EventSource)."""

    model_chat = chat_model or CHAT_MODEL
    model_embed = embed_model or EMBED_MODEL

    def event_stream():
        start = time.time()
        try:
            # 1) Embed the question
            qvec = embed_query(question, model_embed)

            # 2) Retrieve top-k chunks
            hits = retrieve_chunks(qvec, top_k)

            # 3) Build context
            context, cites = build_context(hits, max_context)

            if not context:
                yield sse_format("token", json.dumps("No relevant context found. Try rephrasing."))
                yield sse_format("done", json.dumps({"cites": [], "elapsed": round(time.time() - start, 2)}))
                return

            # 4) Build prompt
            prompt = f"""
{SYSTEM_INSTRUCTIONS}

# Context (excerpts)
{context}

# Task
Answer the user’s question using only the context above.

# Question
{question}

# Answer
""".strip()

            # 5) Try streaming tokens from Ollama
            sent_any = False
            with requests.post(
                ollama_url("/api/generate"),
                json={"model": model_chat, "prompt": prompt, "stream": True},
                stream=True,
                timeout=300,
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    # Normalize to text
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = obj.get("response")
                    if token:
                        sent_any = True
                        yield sse_format("token", json.dumps(token))
                    if obj.get("done"):
                        break

            # 6) Fallback to non-streaming if nothing was sent
            if not sent_any:
                try:
                    r2 = requests.post(
                        ollama_url("/api/generate"),
                        json={"model": model_chat, "prompt": prompt, "stream": False},
                        timeout=300,
                    )
                    r2.raise_for_status()
                    resp_obj = r2.json()
                    full = resp_obj.get("response", "") or ""
                    if full:
                        yield sse_format("token", json.dumps(full))
                except Exception as e:
                    yield sse_format("token", json.dumps(f"[error] ollama fallback failed: {e}"))

            # 7) Finish with done event and cite list
            yield sse_format(
                "done",
                json.dumps({
                    "cites": list(dict.fromkeys([f"{h['book_id']}#{h['chunk_id']}" for h in hits])),
                    "elapsed": round(time.time() - start, 2),
                }),
            )

        except Exception as e:
            yield sse_format("token", json.dumps(f"[error] {e}"))
            yield sse_format("done", json.dumps({"cites": [], "elapsed": round(time.time() - start, 2)}))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
