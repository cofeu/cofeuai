"""CofeuAI Web Sunucusu.

Eğitilmiş modeli bir REST API olarak sunar ve tarayıcıda sohbet
edebileceğiniz güzel bir arayüz sağlar.

Özellikler:
  - Streaming SSE endpoint (token-by-token)
  - Sampling parametreleri (top-k, top-p, repetition penalty)
  - Structured logging
  - Health check

Çalıştırma:
    cd python
    ../.venv/bin/python server.py
    # Tarayıcıda: http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cofeu.server")

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "cofeu.pt"
VOCAB_PATH = ROOT / "checkpoints" / "vocab.json"
BIN_PATH = ROOT / "checkpoints" / "cofeu.bin"

app = FastAPI(title="CofeuAI", description="Kendi LLM'imiz — sıfırdan eğitildi")

_model = None
_tokenizer = None
_cpp_model = None
_cpp_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not CKPT_PATH.exists():
        raise RuntimeError("Model bulunamadı. Önce eğitim yapın: python train.py")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config: ModelConfig = ckpt["config"]
    _tokenizer = BPETokenizer(
        merges=ckpt.get("merges", []),
        vocab=ckpt["vocab"],
        special_tokens=ckpt.get("special_tokens"),
    )
    _model = CofeuTransformer(config)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    logger.info("Model yüklendi (%.2fM parametre)", sum(p.numel() for p in _model.parameters()) / 1e6)
    return _model, _tokenizer


def load_cpp():
    global _cpp_model, _cpp_tokenizer
    if _cpp_model is not None:
        return _cpp_model, _cpp_tokenizer
    try:
        from cpp_bridge import CppModel, CppTokenizer

        _cpp_tokenizer = CppTokenizer(VOCAB_PATH)
        _cpp_model = CppModel(BIN_PATH)
        logger.info("C++ inference motoru yüklendi")
        return _cpp_model, _cpp_tokenizer
    except Exception:
        return None, None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Başlangıç metni")
    max_tokens: int = Field(default=200, ge=1, le=4096, description="Maksimum token sayısı")
    temperature: float = Field(default=0.8, ge=0.01, le=2.0, description="Sıcaklık")
    top_k: int | None = Field(default=None, ge=1, le=1000, description="Top-k sampling")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=5.0, description="Tekrar cezası")


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int = 0
    elapsed_ms: float = 0.0


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start_time = time.time()
    logger.info("İstek: prompt='%s', max_tokens=%d, temp=%.2f", req.prompt[:50], req.max_tokens, req.temperature)

    cpp_model, cpp_tok = load_cpp()
    if cpp_model is not None:
        prompt_ids = cpp_tok.encode(req.prompt)
        out_ids = cpp_model.generate(prompt_ids, req.max_tokens, req.temperature)
        text = cpp_tok.decode(out_ids)
        elapsed = (time.time() - start_time) * 1000
        logger.info("C++ üretim tamamlandı (%.0fms)", elapsed)
        return GenerateResponse(text=text, tokens_generated=len(out_ids) - len(prompt_ids), elapsed_ms=elapsed)

    model, tokenizer = load_model()
    prompt_ids = tokenizer.encode(req.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    eos_id = tokenizer.eos_id if tokenizer.eos_id >= 0 else model.config.eos_token_id

    out = model.generate(
        idx, req.max_tokens, temperature=req.temperature,
        top_k=req.top_k, top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=eos_id,
    )
    text = tokenizer.decode(out[0].tolist())

    elapsed = (time.time() - start_time) * 1000
    tokens_generated = out.shape[1] - len(prompt_ids)
    logger.info("Üretim tamamlandı: %d token (%.0fms)", tokens_generated, elapsed)

    return GenerateResponse(text=text, tokens_generated=tokens_generated, elapsed_ms=elapsed)


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest, request: Request):
    """Streaming SSE endpoint — token-by-token üretim."""
    logger.info("Streaming istek: prompt='%s', max_tokens=%d", req.prompt[:50], req.max_tokens)

    model, tokenizer = load_model()
    prompt_ids = tokenizer.encode(req.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    eos_id = tokenizer.eos_id if tokenizer.eos_id >= 0 else model.config.eos_token_id

    async def event_generator():
        start_time = time.time()
        token_count = 0

        for token_tensor in model.generate_stream(
            idx, req.max_tokens, temperature=req.temperature,
            top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            eos_token_id=eos_id,
        ):
            if await request.is_disconnected():
                logger.info("Streaming istek kesildi")
                break

            token_id = token_tensor.item()
            decoded = tokenizer.decode([token_id])
            token_count += 1

            data = json.dumps({"token": decoded, "token_id": token_id, "count": token_count})
            yield f"data: {data}\n\n"

        elapsed = (time.time() - start_time) * 1000
        done_data = json.dumps({"done": True, "tokens": token_count, "elapsed_ms": elapsed})
        yield f"data: {done_data}\n\n"
        logger.info("Streaming tamamlandı: %d token (%.0fms)", token_count, elapsed)

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
    cpp_model, _ = load_cpp()
    model_loaded = _model is not None or cpp_model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "cpp_available": cpp_model is not None,
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
    import uvicorn

    try:
        load_model()
    except RuntimeError as e:
        logger.warning("Model yüklenemedi: %s", e)

    uvicorn.run(app, host="0.0.0.0", port=8000)
