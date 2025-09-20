import os
import sys
import argparse
import requests
import tempfile
from dataclasses import dataclass
from typing import List
from concurrent.futures import ThreadPoolExecutor

# UI libs
import streamlit as st
import gradio as gr

# Transformers
from transformers import pipeline

# ---------------------------
# Config
# ---------------------------
DEFAULT_GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

IBM_VOICES = {
    "Lisa (Female, warm/clear)": "en-US_LisaV3Voice",
    "Michael (Male, professional)": "en-US_MichaelV3Voice",
    "Allison (Neutral, conversational)": "en-US_AllisonV3Voice",
    "Madhur (Hindi, male)": "hi-IN_MadhurV3Voice",
    "Venba (Tamil, female)": "ta-IN_VenbaV3Voice",
}

LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta"
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
# Cached Transformers
# ---------------------------
_translation_pipes = {}

def get_translation_pipe(target_lang: str):
    if target_lang == "English":
        return None
    if target_lang not in _translation_pipes:
        lang_code = LANGUAGE_MAP.get(target_lang)
        model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
        _translation_pipes[target_lang] = pipeline("translation", model=model_name)
    return _translation_pipes[target_lang]

def translate_text(text: str, target_lang: str) -> str:
    if target_lang == "English":
        return text
    try:
        trans_pipe = get_translation_pipe(target_lang)
        return trans_pipe(text)[0]['translation_text']
    except Exception as e:
        print(f"[⚠️ Translation Error] {e}")
        return text

# ---------------------------
# Rewriter
# ---------------------------
class GraniteRewriter:
    def _init_(self, model_name=DEFAULT_GRANITE_MODEL):
        self.model_name = model_name
        self._pipe = None

    def _init(self):
        if self._pipe:
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
            out = self._pipe(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
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

def generate_audiobook(original_text: str, tone: str, voice: str, rewriter: GraniteRewriter, api_key: str, url: str, target_lang: str = "English") -> EchoVerseResult:
    chunks = smart_chunk_text(original_text, max_chars=2000)

    # Parallel rewriting
    with ThreadPoolExecutor() as executor:
        rewritten_chunks = list(executor.map(lambda c: rewriter.rewrite_chunk(c, tone), chunks))

    rewritten = "\n\n".join(rewritten_chunks)

    # Translate once
    if target_lang != "English":
        rewritten = translate_text(rewritten, target_lang)

    # Parallel TTS
    voice_id = IBM_VOICES[voice]
    tts_chunks = smart_chunk_text(rewritten, 2000)
    with ThreadPoolExecutor() as executor:
        audio_parts = list(executor.map(lambda c: synthesize_ibm_tts(c, voice_id, api_key, url), tts_chunks))

    return EchoVerseResult(rewritten_text=rewritten, audio_bytes=b"".join(audio_parts))

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
        target_lang = st.selectbox("Target Language", ["English", "Hindi", "Tamil"])

        api_key = os.environ.get("IBM_TTS_APIKEY", "")
        url = os.environ.get("IBM_TTS_URL", "")

        if st.button("Generate 🎧"):
            try:
                rewriter = GraniteRewriter()
                result = generate_audiobook(text, tone, voice, rewriter, api_key, url, target_lang)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if "result" in st.session_state:
            r: EchoVerseResult = st.session_state["result"]
            st.subheader("Rewritten Text")
            st.text_area("Rewritten", value=r.rewritten_text, height=200)
            st.audio(r.audio_bytes, format="audio/mp3")
            st.download_button("Download MP3", data=r.audio_bytes, file_name=r.audio_filename, mime="audio/mpeg")

# ---------------------------
# Gradio UI
# ---------------------------
def gradio_app():
    def run(text, tone, voice, api_key, url, target_lang):
        try:
            rewriter = GraniteRewriter()
            r = generate_audiobook(text, tone, voice, rewriter, api_key, url, target_lang)

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
        lang = gr.Dropdown(["English", "Hindi", "Tamil"], label="Target Language", value="English")
        api_key = gr.Textbox(label="IBM TTS API Key", value=os.environ.get("IBM_TTS_APIKEY", ""))
        url = gr.Textbox(label="IBM TTS URL", value=os.environ.get("IBM_TTS_URL", ""))
        btn = gr.Button("Generate 🎧")

        audio = gr.Audio(label="Audio", type="filepath")
        rewritten = gr.Textbox(label="Rewritten text", lines=10)

        btn.click(run, [text, tone, voice, api_key, url, lang], [audio, rewritten])
    demo.launch()

# ---------------------------
# Entrypoint
# ---------------------------
def main():
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
        streamlit_app()

if _name_ == "_main_":
    main()

