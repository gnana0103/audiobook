"""
app.py — EchoVerse
Generative Audiobook Creator with Streamlit (local) and Gradio (Colab-friendly).

Features:
- Tone-adaptive rewriting (Neutral / Suspenseful / Inspiring)
- Side-by-side text comparison
- High-quality TTS using IBM Watson Text-to-Speech (MP3 output)
- Streamlit UI (desktop) or Gradio UI (Colab/Jupyter-friendly)

Run:
- Local Streamlit:  streamlit run app.py
- Local Gradio:     python app.py --gradio
- Google Colab:     python app.py   (auto-detects Jupyter → launches Gradio)
"""
st
import os
import sys
import argparse
import requests
import tempfile
from dataclasses import dataclass
from typing import List

# UI libs
import streamlit as st
import gradio as gr

# Optional transformers pipeline
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ---------------------------
# Config
# ---------------------------
DEFAULT_GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"
IBM_VOICES = {
    "Lisa (Female, warm/clear)": "en-US_LisaV3Voice",
    "Michael (Male, professional)": "en-US_MichaelV3Voice",
    "Allison (Neutral, conversational)": "en-US_AllisonV3Voice",
}

# ---------------------------
# Utils
# ---------------------------
def smart_chunk_text(text: str, max_chars: int = 3500) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = []
    while len(text) > max_chars:
        cut = text.rfind(".", 0, max_chars)
        if cut == -1:
            cut = max_chars
        parts.append(text[:cut + 1])
        text = text[cut + 1:]
    if text:
        parts.append(text)
    return [p.strip() for p in parts if p.strip()]

def heuristic_rewrite(text: str, tone: str) -> str:
    if tone == "Neutral":
        return text
    elif tone == "Suspenseful":
        return text.replace(".", "...").replace("\n", "\n\n")
    else:  # Inspiring
        return text + "\n\nAnd remember: greatness awaits!"

# ---------------------------
# Rewriter
# ---------------------------
class GraniteRewriter:
    def __init__(self, model_name=DEFAULT_GRANITE_MODEL):
        self.model_name = model_name
        self._pipe = None

    def _init(self):
        if self._pipe or not HAS_TRANSFORMERS:
            return
        try:
            self._pipe = pipeline("text-generation", model=self.model_name)
        except Exception as e:
            print("⚠️ Could not load model:", e)
            self._pipe = None

    def rewrite_chunk(self, text: str, tone: str) -> str:
        self._init()
        if not self._pipe:
            return heuristic_rewrite(text, tone)
        try:
            prompt = f"Rewrite the following text in a {tone} tone, keeping meaning intact:\n\n{text}\n\nRewritten:"
            out = self._pipe(prompt, max_new_tokens=500, do_sample=False)[0]["generated_text"]
            return out.split("Rewritten:")[-1].strip()
        except Exception:
            return heuristic_rewrite(text, tone)

# ---------------------------
# TTS
# ---------------------------
def synthesize_ibm_tts(text: str, voice: str, api_key: str, url: str) -> bytes:
    if not api_key or not url:
        raise ValueError("IBM TTS credentials missing")
    endpoint = url.rstrip("/") + "/v1/synthesize"
    headers = {"Accept": "audio/mp3"}
    auth = ("apikey", api_key)
    r = requests.post(endpoint, headers=headers, params={"voice": voice}, auth=auth, data=text.encode("utf-8"))
    if r.status_code not in (200, 201):
        raise RuntimeError(f"TTS failed {r.status_code}: {r.text}")
    return r.content

# ---------------------------
# Pipeline
# ---------------------------
@dataclass
class EchoVerseResult:
    rewritten_text: str
    audio_bytes: bytes
    audio_filename: str = "echoverse_output.mp3"

def generate_audiobook(original_text: str, tone: str, voice: str, rewriter: GraniteRewriter, api_key: str, url: str) -> EchoVerseResult:
    chunks = smart_chunk_text(original_text, max_chars=2000)
    rewritten_chunks = [rewriter.rewrite_chunk(c, tone) for c in chunks]
    rewritten = "\n\n".join(rewritten_chunks)
    voice_id = IBM_VOICES[voice]
    audio = b"".join([synthesize_ibm_tts(c, voice_id, api_key, url) for c in smart_chunk_text(rewritten, 2000)])
    return EchoVerseResult(rewritten_text=rewritten, audio_bytes=audio)

# ---------------------------
# Streamlit UI
# ---------------------------
def streamlit_app():
    st.set_page_config(page_title="EchoVerse", layout="wide")
    st.title("📚 EchoVerse — Audiobook Creator")

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Input", ["Paste text", "Upload .txt"])
        if mode == "Paste text":
            text = st.text_area("Enter text", height=300)
        else:
            f = st.file_uploader("Upload .txt", type=["txt"])
            text = f.read().decode("utf-8") if f else ""

        tone = st.selectbox("Tone", ["Neutral", "Suspenseful", "Inspiring"])
        voice = st.selectbox("Voice", list(IBM_VOICES.keys()))

        api_key = os.environ.get("IBM_TTS_APIKEY", "")
        url = os.environ.get("IBM_TTS_URL", "")

        if st.button("Generate 🎧"):
            try:
                rewriter = GraniteRewriter()
                result = generate_audiobook(text, tone, voice, rewriter, api_key, url)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if "result" in st.session_state:
            r: EchoVerseResult = st.session_state["result"]
            st.subheader("Original")
            st.text_area("Original", value=text, height=200)
            st.subheader("Rewritten")
            st.text_area("Rewritten", value=r.rewritten_text, height=200)
            st.audio(r.audio_bytes, format="audio/mp3")
            st.download_button("Download MP3", data=r.audio_bytes, file_name=r.audio_filename, mime="audio/mpeg")

# ---------------------------
# Gradio UI
# ---------------------------
def gradio_app():
    def run(text, tone, voice, api_key, url):
        try:
            rewriter = GraniteRewriter()
            r = generate_audiobook(text, tone, voice, rewriter, api_key, url)

            # Save to temp file for playback
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(r.audio_bytes)
            tmp.flush()

            return tmp.name, r.rewritten_text
        except Exception as e:
            return None, f"Error: {e}"

    with gr.Blocks() as demo:
        gr.Markdown("# 📚 EchoVerse — Audiobook Creator")
        text = gr.Textbox(label="Input text", lines=10)
        tone = gr.Radio(["Neutral", "Suspenseful", "Inspiring"], value="Neutral")
        voice = gr.Dropdown(list(IBM_VOICES.keys()), value=list(IBM_VOICES.keys())[0])
        api_key = gr.Textbox(label="IBM TTS API Key", value=os.environ.get("IBM_TTS_APIKEY", ""))
        url = gr.Textbox(label="IBM TTS URL", value=os.environ.get("IBM_TTS_URL", ""))
        btn = gr.Button("Generate 🎧")

        # ✅ fixed: type="filepath" for playback
        audio = gr.Audio(label="Audio", type="filepath")
        rewritten = gr.Textbox(label="Rewritten text", lines=10)

        btn.click(run, [text, tone, voice, api_key, url], [audio, rewritten])
    demo.launch()

# ---------------------------
# Entrypoint
# ---------------------------
def main():
    # Detect if running in Jupyter/Colab
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        print("📌 Detected Colab/Jupyter — launching Gradio UI")
        gradio_app()
        return

    argv = [a for a in sys.argv if not a.startswith("-f")]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradio", action="store_true", help="Run Gradio UI")
    args = parser.parse_args(argv[1:])

    if args.gradio:
        gradio_app()
    else:
        print("👉 Run with: streamlit run app.py (for Streamlit UI)")
        print("👉 Or: python app.py --gradio (for Gradio UI)")

if __name__ == "__main__":
    main()
