"""CofeuAI Web Sunucusu.

Eğitilmiş modeli bir REST API olarak sunar ve tarayıcıda sohbet
edebileceğiniz güzel bir arayüz sağlar.

Özellikler:
  - Streaming SSE endpoint (token-by-token)
  - C++ inference motoru (varsa) — PyTorch'a otomatik düşüş
  - Sampling parametreleri (top-k, top-p, repetition penalty)
  - Structured logging
  - Health check

Çalıştırma:
    cd python
    ../.venv/bin/python server.py
    # Tarayıcıda: http://localhost:8000

Not: Ne C++ motoru ne de PyTorch modeli thread-safe'dir (KV cache ve workspace
tamponlarını paylaşırlar). Bu yüzden her üretim `Backend.lock()` üzerinden
geçer; eşzamanlı istekler sıraya girer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import runtime
from runtime import Backend, ModelLoadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cofeu.server")

app = FastAPI(title="CofeuAI", description="Kendi LLM'imiz — sıfırdan eğitildi")

_backend: Optional[Backend] = None
_load_error: Optional[str] = None


def load_backend() -> Backend:
    """Modeli bir kez yükler, sonraki isteklerde aynısını döndürür."""
    global _backend, _load_error
    if _backend is not None:
        return _backend
    if _load_error is not None:
        raise HTTPException(status_code=503, detail=_load_error)
    try:
        _backend = runtime.load_backend()
        logger.info("Motor hazır: %s", _backend.name)
    except ModelLoadError as e:
        _load_error = str(e)
        logger.error("Model yüklenemedi: %s", e)
        raise HTTPException(status_code=503, detail=_load_error)
    return _backend


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Başlangıç metni")
    max_tokens: int = Field(default=200, ge=1, le=4096, description="Maksimum token sayısı")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0,
                               description="Sıcaklık (0 = greedy)")
    top_k: int | None = Field(default=None, ge=1, le=1000, description="Top-k sampling")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0,
                                description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=5.0, description="Tekrar cezası")
    seed: int = Field(default=0, ge=0, description="Tohum (0 = rastgele)")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int = 0
    elapsed_ms: float = 0.0
    engine: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    backend = load_backend()
    start = time.time()
    logger.info(
        "İstek: prompt='%s' max_tokens=%d temp=%.2f top_k=%s top_p=%s rp=%.2f",
        req.prompt[:60], req.max_tokens, req.temperature,
        req.top_k, req.top_p, req.repetition_penalty,
    )

    prompt_ids = backend.tokenizer.encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="Prompt tokenize edilemedi")

    try:
        ids = runtime.generate_ids(
            backend, prompt_ids, req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            eos_token_id=backend.tokenizer.eos_id, seed=req.seed,
        )
    except RuntimeError as e:
        logger.error("Üretim hatası: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.time() - start) * 1000
    logger.info("Üretim tamamlandı: %d token (%.0f ms, %.0f tok/s)",
                len(ids), elapsed, len(ids) / max(elapsed / 1000, 1e-9))
    return GenerateResponse(
        text=backend.tokenizer.decode(ids),
        tokens_generated=len(ids),
        elapsed_ms=elapsed,
        engine=backend.name,
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest, request: Request):
    """Streaming SSE endpoint — token-by-token üretim."""
    backend = load_backend()
    prompt_ids = backend.tokenizer.encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="Prompt tokenize edilemedi")

    logger.info("Streaming istek: prompt='%s' max_tokens=%d (motor=%s)",
                req.prompt[:60], req.max_tokens, backend.name)
    tok = backend.tokenizer
    eos = tok.eos_id

    async def event_generator():
        start = time.time()
        count = 0
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        # runtime.stream_ids senkron bir generator'dır ve kilidi tutar; event
        # loop'unu bloklamamak için ayrı bir thread'de tüketiyoruz. Her token
        # üretildiği anda kuyruğa düşer, yani akış gerçekten token-by-token.
        gen = runtime.stream_ids(
            backend, prompt_ids, req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            eos_token_id=eos, seed=req.seed,
        )

        def _pump() -> None:
            """Generator'ı süren tek thread. close() da burada çağrılır ki
            generator'a eşzamanlı erişim olmasın."""
            try:
                for token in gen:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                    if stop.is_set():
                        break
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except BaseException as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                gen.close()  # kilidi burada serbest bırakır

        threading.Thread(target=_pump, name="cofeu-sse", daemon=True).start()

        try:
            while True:
                kind, payload = await queue.get()

                if kind == "done":
                    break
                if kind == "error":
                    logger.error("Streaming hatası: %s", payload)
                    yield "data: " + json.dumps(
                        {"error": str(payload)}
                    ) + "\n\n"
                    return

                if await request.is_disconnected():
                    logger.info("İstemci koptu, üretim durduruluyor")
                    stop.set()
                    return

                count += 1
                yield "data: " + json.dumps({
                    "token": tok.decode([payload]),
                    "token_id": payload,
                    "count": count,
                }) + "\n\n"

            elapsed = (time.time() - start) * 1000
            yield "data: " + json.dumps({
                "done": True, "tokens": count, "elapsed_ms": elapsed,
                "engine": backend.name,
            }) + "\n\n"
            logger.info("Streaming tamamlandı: %d token (%.0f ms)", count, elapsed)
        finally:
            stop.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    """Sağlık kontrolü + yüklü motor bilgisi."""
    try:
        backend = load_backend()
    except HTTPException as e:
        return {"status": "unavailable", "detail": e.detail}

    info = {
        "status": "ok",
        "model_loaded": True,
        "engine": backend.name,
        "vocab_size": backend.tokenizer.vocab_size,
        "eos_token_id": backend.tokenizer.eos_id,
    }
    if backend.uses_cpp:
        info.update({
            "cpp_available": True,
            "block_size": backend.model.block_size,
            "max_position": backend.model.max_position,
        })
    else:
        cfg = backend.model.config
        info.update({
            "cpp_available": False,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "block_size": cfg.block_size,
            "rope_base": cfg.rope_base,
        })
    return info


@app.get("/model")
def model_info():
    """Eğitilmiş modelin mimarisi."""
    backend = load_backend()
    if backend.uses_cpp:
        return {
            "engine": backend.name,
            "vocab_size": backend.model.vocab_size,
            "block_size": backend.model.block_size,
            "max_position": backend.model.max_position,
        }
    cfg = backend.model.config
    return {
        "engine": backend.name,
        "vocab_size": cfg.vocab_size,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_layer": cfg.n_layer,
        "block_size": cfg.block_size,
        "rope_base": cfg.rope_base,
        "n_params": cfg.n_params(),
    }



HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CofeuAI — Kendi LLM'imiz</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #1a1d27;
    --border: #2a2e3d;
    --text: #e6e8ef;
    --muted: #8b90a0;
    --accent: #6c8cff;
    --accent2: #9b6cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .logo {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 16px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 13px; }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.ai {
    align-self: flex-start;
    background: var(--panel);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg.ai.loading { color: var(--muted); font-style: italic; }
  .msg .meta {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }
  footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }
  #input {
    flex: 1;
    min-width: 200px;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    font-size: 15px;
    resize: none;
    outline: none;
  }
  #input:focus { border-color: var(--accent); }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .controls label { color: var(--muted); font-size: 12px; }
  .controls input[type=range] { width: 70px; }
  .controls select {
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    font-size: 12px;
  }
  #send {
    padding: 12px 20px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
  }
  #send:hover { opacity: 0.9; }
  #send:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="logo">C</div>
  <div>
    <h1>CofeuAI</h1>
    <div class="sub">Kendi LLM'imiz — sıfırdan eğitildi</div>
  </div>
</header>

<div id="chat"></div>

<footer>
  <textarea id="input" rows="1" placeholder="Bir şeyler yazın... (örn. Bir zamanlar)"></textarea>
  <div class="controls">
    <label>Temp <span id="tempVal">0.8</span></label>
    <input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.8">
    <label>Top-k</label>
    <select id="topk">
      <option value="">Kapalı</option>
      <option value="10">10</option>
      <option value="20">20</option>
      <option value="40">40</option>
      <option value="100">100</option>
    </select>
    <label>Top-p</label>
    <select id="topp">
      <option value="">Kapalı</option>
      <option value="0.9">0.9</option>
      <option value="0.95">0.95</option>
    </select>
    <label>RepPen</label>
    <input type="range" id="reppen" min="1.0" max="2.0" step="0.1" value="1.0">
    <span id="reppenVal">1.0</span>
  </div>
  <button id="send">Gönder</button>
</footer>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const temp = document.getElementById('temp');
const tempVal = document.getElementById('tempVal');
const topk = document.getElementById('topk');
const topp = document.getElementById('topp');
const reppen = document.getElementById('reppen');
const reppenVal = document.getElementById('reppenVal');

temp.addEventListener('input', () => tempVal.textContent = temp.value);
reppen.addEventListener('input', () => reppenVal.textContent = reppen.value);

function addMsg(text, who, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  if (meta) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = meta;
    div.appendChild(metaDiv);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addStreamingMsg() {
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.textContent = '';
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function sendMsg() {
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  addMsg(prompt, 'user');

  send.disabled = true;

  const reqBody = {
    prompt: prompt,
    max_tokens: 200,
    temperature: parseFloat(temp.value)
  };
  if (topk.value) reqBody.top_k = parseInt(topk.value);
  if (topp.value) reqBody.top_p = parseFloat(topp.value);
  if (parseFloat(reppen.value) > 1.0) reqBody.repetition_penalty = parseFloat(reppen.value);

  try {
    const response = await fetch('/generate/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody)
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const aiDiv = addStreamingMsg();
    let fullText = '';
    let meta = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.done) {
              meta = data.tokens + ' token | ' + Math.round(data.elapsed_ms) + 'ms';
              if (meta) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'meta';
                metaDiv.textContent = meta;
                aiDiv.appendChild(metaDiv);
              }
            } else {
              fullText += data.token;
              aiDiv.textContent = fullText;
              chat.scrollTop = chat.scrollHeight;
            }
          } catch (e) {}
        }
      }
    }
  } catch (e) {
    addMsg('Hata: ' + e.message, 'ai');
  }
  send.disabled = false;
}

send.addEventListener('click', sendMsg);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMsg();
  }
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="CofeuAI web sunucusu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--python", action="store_true", help="PyTorch motoru (C++ yerine)")
    ap.add_argument("--cuda", action="store_true", help="PyTorch yolunda GPU kullan")
    ap.add_argument("--reload", action="store_true", help="Geliştirme için auto-reload")
    args = ap.parse_args()

    # Modeli başlangıçta yükle: ilk istekte beklemek yerine hata hemen görünsün.
    try:
        bk = runtime.load_backend(
            prefer_cpp=not args.python, device="cuda" if args.cuda else "cpu"
        )
        _backend = bk
        if bk.uses_cpp:
            logger.info("C++ motoru: V=%d block=%d max_pos=%d",
                        bk.model.vocab_size, bk.model.block_size, bk.model.max_position)
    except ModelLoadError as e:
        logger.error("Model yüklenemedi:\n%s", e)
        logger.error("Sunucu yine de başlıyor; /health 'unavailable' dönecek.")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
