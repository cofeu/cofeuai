"""CofeuAI Web Sunucusu.

Eğitilmiş modeli bir REST API olarak sunar ve tarayıcıda sohbet
edebileceğiniz güzel bir arayüz sağlar.

Çalıştırma:
    cd python
    ../.venv/bin/python server.py
    # Tarayıcıda: http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from model import CofeuTransformer
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "cofeu.pt"
VOCAB_PATH = ROOT / "checkpoints" / "vocab.json"
BIN_PATH = ROOT / "checkpoints" / "cofeu.bin"

app = FastAPI(title="CofeuAI", description="Kendi LLM'imiz")

# Modeli yükle (başlangıçta bir kez)
_model = None
_tokenizer = None
_cpp_model = None
_cpp_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not CKPT_PATH.exists():
        raise RuntimeError(
            "Model bulunamadı. Önce eğitim yapın: python train.py"
        )

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    _tokenizer = BPETokenizer(merges=ckpt.get("merges", []), vocab=ckpt["vocab"])
    _model = CofeuTransformer(config)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    return _model, _tokenizer


def load_cpp():
    """C++ inference motorunu yükler (hızlı üretim için)."""
    global _cpp_model, _cpp_tokenizer
    if _cpp_model is not None:
        return _cpp_model, _cpp_tokenizer
    try:
        from cpp_bridge import CppModel, CppTokenizer

        _cpp_tokenizer = CppTokenizer(VOCAB_PATH)
        _cpp_model = CppModel(BIN_PATH)
        return _cpp_model, _cpp_tokenizer
    except Exception:
        return None, None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8


class GenerateResponse(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Önce C++ ile dene (hızlı)
    cpp_model, cpp_tok = load_cpp()
    if cpp_model is not None:
        prompt_ids = cpp_tok.encode(req.prompt)
        out_ids = cpp_model.generate(prompt_ids, req.max_tokens, req.temperature)
        return GenerateResponse(text=cpp_tok.decode(out_ids))

    # C++ yoksa Python'a düş
    model, tokenizer = load_model()
    prompt_ids = tokenizer.encode(req.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(idx, req.max_tokens, temperature=req.temperature)
    text = tokenizer.decode(out[0].tolist())
    return GenerateResponse(text=text)


@app.get("/health")
def health():
    cpp_model, _ = load_cpp()
    return {"status": "ok", "model_loaded": _model is not None or cpp_model is not None}


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
  footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: center;
  }
  #input {
    flex: 1;
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
  .controls { display: flex; gap: 8px; align-items: center; }
  .controls label { color: var(--muted); font-size: 12px; }
  .controls input[type=range] { width: 80px; }
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
  </div>
  <button id="send">Gönder</button>
</footer>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const temp = document.getElementById('temp');
const tempVal = document.getElementById('tempVal');

temp.addEventListener('input', () => tempVal.textContent = temp.value);

function addMsg(text, who) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function sendMsg() {
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  addMsg(prompt, 'user');

  const loading = addMsg('Düşünüyorum...', 'ai loading');
  send.disabled = true;

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: 200,
        temperature: parseFloat(temp.value)
      })
    });
    const data = await res.json();
    loading.remove();
    addMsg(data.text, 'ai');
  } catch (e) {
    loading.remove();
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

    # Modeli önceden yüklemeyi dene (opsiyonel)
    try:
        load_model()
        print("Model yüklendi.")
    except RuntimeError as e:
        print(f"Uyarı: {e}")

    uvicorn.run(app, host="0.0.0.0", port=8000)