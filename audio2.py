import streamlit as st
import pyttsx3
import io
import base64
import tempfile
import os
import time
from PIL import Image
import re
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import math

# Optional imports with fallbacks
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🎧 AI Audiobook Creator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .feature-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .stSelectbox > div > div > select {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .audio-controls {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .summary-section {
        background: linear-gradient(145deg, #e3f2fd, #ffffff);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #2196f3;
        margin: 1rem 0;
    }
    
    .mindmap-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .error-message {
        background: linear-gradient(145deg, #ffebee, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(145deg, #e8f5e8, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

class AudiobookCreator:
    def __init__(self):
        self.tts_engine = None
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            st.error(f"TTS Engine initialization failed: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            st.error("📋 PyPDF2 is not installed. Install with: `pip install PyPDF2`")
            return ""
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(pdf_reader.pages)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf_reader.pages):
                status_text.text(f"Processing page {i+1}/{total_pages}...")
                text += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / total_pages)
            
            status_text.empty()
            progress_bar.empty()
            
            if not text.strip():
                st.warning("⚠️ No text found in PDF. The PDF might contain only images.")
                return ""
                
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_image(self, image_file):
        """Extract text from image using OCR"""
        if not TESSERACT_AVAILABLE:
            st.error("🖼️ pytesseract is not installed. Install with: `pip install pytesseract`")
            st.info("💡 Also install Tesseract OCR system dependency")
            return ""
        
        try:
            image = Image.open(image_file)
            
            # Display image preview
            st.image(image, caption="Processing this image...", width=300)
            
            with st.spinner("🔍 Extracting text from image..."):
                text = pytesseract.image_to_string(image, lang='eng')
                
            if not text.strip():
                st.warning("⚠️ No text found in image. Make sure the image contains readable text.")
                return ""
                
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("💡 Make sure Tesseract OCR is properly installed on your system")
            return ""
    
    def translate_text(self, text, target_language):
        """Translate text to target language"""
        if not TRANSLATOR_AVAILABLE:
            st.error("🌍 deep-translator is not installed. Install with: `pip install deep-translator`")
            return text
            
        try:
            # Language mapping
            lang_map = {
                'Tamil': 'ta',
                'Hindi': 'hi',
                'English': 'en',
                'Spanish': 'es',
                'French': 'fr',
                'German': 'de',
                'Italian': 'it',
                'Portuguese': 'pt',
                'Russian': 'ru',
                'Japanese': 'ja',
                'Korean': 'ko',
                'Chinese': 'zh'
            }
            
            target_lang = lang_map.get(target_language, 'en')
            
            if target_lang == 'en':  # Skip if already English
                return text
            
            # Split text into smaller chunks for better translation
            max_chunk_size = 4500  # deep_translator limit
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            translated_chunks = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    try:
                        status_text.text(f"Translating chunk {i+1}/{len(chunks)} to {target_language}...")
                        translator = GoogleTranslator(source='auto', target=target_lang)
                        translated = translator.translate(chunk)
                        translated_chunks.append(translated)
                        time.sleep(0.1)  # Small delay to avoid rate limiting
                    except Exception as chunk_error:
                        st.warning(f"⚠️ Failed to translate chunk {i+1}: {str(chunk_error)}")
                        translated_chunks.append(chunk)  # Keep original if translation fails
                
                progress_bar.progress((i + 1) / len(chunks))
            
            status_text.empty()
            progress_bar.empty()
            
            result = ' '.join(translated_chunks)
            return result if result.strip() else text
            
        except Exception as e:
            st.error(f"❌ Translation error: {str(e)}")
            return text
    
    def create_audio_fixed(self, text, voice_settings):
        """Create audio from text with proper file handling"""
        if not self.tts_engine:
            st.error("❌ TTS Engine not available")
            return None
            
        try:
            # Limit text length to prevent too long audio
            if len(text) > 10000:
                text = text[:10000] + "... (truncated for audio generation)"
                st.info("📝 Text truncated to 10,000 characters for audio generation")
            
            # Configure TTS engine
            voices = self.tts_engine.getProperty('voices')
            
            if voices and len(voices) > 0:
                # Set voice based on gender preference
                if voice_settings['gender'] == 'Female' and len(voices) > 1:
                    # Try to find a female voice
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    else:
                        # Fallback to second voice if available
                        self.tts_engine.setProperty('voice', voices[1].id)
                else:
                    # Use first available voice for male or default
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set rate and volume
            self.tts_engine.setProperty('rate', voice_settings['speed'])
            self.tts_engine.setProperty('volume', voice_settings['volume'])
            
            # Create temporary file with unique name
            temp_dir = tempfile.gettempdir()
            temp_filename = f"audiobook_{int(time.time())}_{os.getpid()}.wav"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            try:
                # Generate audio file
                self.tts_engine.save_to_file(text, temp_filepath)
                self.tts_engine.runAndWait()
                
                # Wait a bit for file to be fully written
                time.sleep(0.5)
                
                # Check if file exists and has content
                if not os.path.exists(temp_filepath):
                    st.error("❌ Audio file was not created")
                    return None
                
                file_size = os.path.getsize(temp_filepath)
                if file_size == 0:
                    st.error("❌ Audio file is empty")
                    return None
                
                # Read the audio file
                max_attempts = 5
                audio_data = None
                
                for attempt in range(max_attempts):
                    try:
                        with open(temp_filepath, 'rb') as audio_file:
                            audio_data = audio_file.read()
                        break
                    except (PermissionError, OSError) as e:
                        if attempt < max_attempts - 1:
                            time.sleep(0.2)  # Wait before retry
                            continue
                        else:
                            st.error(f"❌ Could not read audio file after {max_attempts} attempts: {str(e)}")
                            return None
                
                return audio_data
                
            finally:
                # Cleanup: Remove temporary file
                max_cleanup_attempts = 3
                for attempt in range(max_cleanup_attempts):
                    try:
                        if os.path.exists(temp_filepath):
                            os.unlink(temp_filepath)
                        break
                    except (PermissionError, OSError):
                        if attempt < max_cleanup_attempts - 1:
                            time.sleep(0.5)
                        # If we can't delete it, it's not critical
                        
        except Exception as e:
            st.error(f"❌ Audio creation error: {str(e)}")
            return None
    
    def summarize_text(self, text):
        """Create text summary with key points"""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 2:
                return text[:500] + "..." if len(text) > 500 else text
            
            # Simple extractive summarization based on word frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'they', 'them', 'their', 'there', 'then', 'than', 'when', 'where', 'who', 'what',
                'how', 'why', 'which', 'while', 'during', 'before', 'after', 'above', 'below', 'between'
            }
            
            for word in stop_words:
                word_freq.pop(word, None)
            
            # Score sentences based on word frequencies
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                words_in_sentence = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
                score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                if len(words_in_sentence) > 0:
                    sentence_scores[i] = score / len(words_in_sentence)  # Normalize by sentence length
                else:
                    sentence_scores[i] = 0
            
            # Get top sentences (30% of total sentences, minimum 2, maximum 5)
            num_summary_sentences = max(2, min(5, int(len(sentences) * 0.3)))
            top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_summary_sentences]
            
            # Sort by original order to maintain flow
            top_sentence_indices.sort()
            summary_sentences = [sentences[i] for i in top_sentence_indices]
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            st.error(f"❌ Summarization error: {str(e)}")
            return text[:500] + "..." if len(text) > 500 else text
    
    def create_mindmap_data(self, text):
        """Create data for mind map visualization"""
        try:
            # Extract meaningful words (4+ characters, not common words)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = Counter(words)
            
            # Remove common words
            stop_words = {
                'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been',
                'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
                'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were',
                'what', 'your', 'would', 'there', 'could', 'other', 'after', 'first', 'also',
                'back', 'work', 'more', 'most', 'only', 'think', 'where', 'being', 'every',
                'great', 'might', 'shall', 'still', 'those', 'while', 'these', 'their'
            }
            
            for word in stop_words:
                word_freq.pop(word, None)
            
            # Get top concepts (limit to 12 for better visualization)
            top_concepts = dict(word_freq.most_common(12))
            return top_concepts
            
        except Exception as e:
            st.error(f"❌ Mind map creation error: {str(e)}")
            return {}

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎧 AI Audiobook Creator</h1>
        <p style="color: white; font-size: 1.2rem; margin: 0;">Transform Text, PDFs & Images into Interactive Audiobooks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize audiobook creator
    creator = AudiobookCreator()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Audio Settings")
        
        # Voice settings
        st.markdown("### 🎭 Voice Configuration")
        voice_gender = st.selectbox("Voice Gender", ["Male", "Female"], help="Select preferred voice gender")
        voice_speed = st.slider("🏃 Speech Speed (WPM)", 100, 300, 200, help="Words per minute")
        voice_volume = st.slider("🔊 Volume", 0.1, 1.0, 0.9, step=0.1, help="Audio volume level")
        
        # Language settings
        st.markdown("### 🌍 Language Settings")
        target_language = st.selectbox(
            "Translation Language", 
            ["English", "Tamil", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese"],
            help="Select target language for translation"
        )
        
        voice_settings = {
            'gender': voice_gender,
            'speed': voice_speed,
            'volume': voice_volume
        }
        
        # System info
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.write("✅ Streamlit" if True else "❌ Streamlit")
        st.write("✅ TTS Engine" if creator.tts_engine else "❌ TTS Engine")
        st.write("✅ PDF Support" if PDF_AVAILABLE else "❌ PDF Support")
        st.write("✅ OCR Support" if TESSERACT_AVAILABLE else "❌ OCR Support")
        st.write("✅ Translation" if TRANSLATOR_AVAILABLE else "❌ Translation")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input methods
        st.markdown("## 📝 Input Methods")
        
        tab1, tab2, tab3 = st.tabs(["📄 Text Input", "📋 PDF Upload", "🖼️ Image Upload"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### ✏️ Direct Text Input")
            input_text = st.text_area(
                "Enter your text here:",
                height=250,
                placeholder="Type or paste your text here to create an audiobook...",
                help="Enter the text you want to convert to audio"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Process Text", key="process_text", help="Process the entered text"):
                    if input_text.strip():
                        st.session_state.processed_text = input_text.strip()
                        st.success("✅ Text processed successfully!")
                    else:
                        st.warning("⚠️ Please enter some text first")
            
            with col_btn2:
                if st.button("🗑️ Clear Text", key="clear_text", help="Clear all text"):
                    st.session_state.processed_text = ""
                    st.session_state.translated_text = ""
                    st.session_state.audio_data = None
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 📋 PDF Document Upload")
            
            if not PDF_AVAILABLE:
                st.error("📋 PDF support not available. Install PyPDF2: `pip install PyPDF2`")
            else:
                uploaded_pdf = st.file_uploader(
                    "📎 Choose a PDF file", 
                    type=['pdf'],
                    help="Upload a PDF file to extract text from"
                )
                
                if uploaded_pdf:
                    st.info(f"📄 Selected: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")
                    
                    if st.button("🔍 Extract Text from PDF", key="extract_pdf"):
                        with st.spinner("📖 Extracting text from PDF..."):
                            extracted_text = creator.extract_text_from_pdf(uploaded_pdf)
                            if extracted_text:
                                st.session_state.processed_text = extracted_text
                                st.success("✅ PDF processed successfully!")
                                with st.expander("📖 Preview extracted text"):
                                    st.text_area("", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=200)
                            else:
                                st.error("❌ No text could be extracted from the PDF")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Image Text Extraction (OCR)")
            
            if not TESSERACT_AVAILABLE:
                st.error("🖼️ OCR support not available. Install pytesseract: `pip install pytesseract`")
                st.info("💡 Also install Tesseract OCR system dependency")
            else:
                uploaded_image = st.file_uploader(
                    "🖼️ Choose an image file", 
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
                    help="Upload an image containing text"
                )
                
                if uploaded_image:
                    st.info(f"🖼️ Selected: {uploaded_image.name} ({uploaded_image.size} bytes)")
                    
                    if st.button("🔍 Extract Text from Image", key="extract_image"):
                        extracted_text = creator.extract_text_from_image(uploaded_image)
                        if extracted_text:
                            st.session_state.processed_text = extracted_text
                            st.success("✅ Image processed successfully!")
                            with st.expander("📖 Preview extracted text"):
                                st.text_area("", extracted_text, height=200)
                        else:
                            st.error("❌ No text could be extracted from the image")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.processed_text:
            # Text statistics
            st.markdown("## 📊 Text Statistics")
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            text_length = len(st.session_state.processed_text)
            word_count = len(st.session_state.processed_text.split())
            sentence_count = len([s for s in re.split(r'[.!?]+', st.session_state.processed_text) if s.strip()])
            paragraph_count = len([p for p in st.session_state.processed_text.split('\n') if p.strip()])
            estimated_duration = word_count / 150  # Average reading speed
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("📝 Words", f"{word_count:,}")
                st.metric("⏱️ Est. Audio Duration", f"{estimated_duration:.1f} min")
                st.metric("📏 Characters", f"{text_length:,}")
            
            with col_stat2:
                st.metric("📄 Sentences", sentence_count)
                st.metric("📋 Paragraphs", paragraph_count)
                if TEXTSTAT_AVAILABLE:
                    try:
                        reading_level = textstat.flesch_reading_ease(st.session_state.processed_text)
                        st.metric("📖 Reading Level", f"{reading_level:.0f}")
                    except:
                        st.metric("📖 Reading Level", "N/A")
                else:
                    st.metric("📖 Reading Level", "N/A")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing and Audio Generation
    if st.session_state.processed_text:
        st.markdown("## 🎯 Audio Generation & Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🌍 Text Translation")
            
            if not TRANSLATOR_AVAILABLE:
                st.error("🌍 Translation not available. Install deep-translator: `pip install deep-translator`")
            else:
                if st.button("🌍 Translate Text", key="translate", help=f"Translate to {target_language}"):
                    if target_language == "English":
                        st.info("💡 Text is already in English")
                        st.session_state.translated_text = st.session_state.processed_text
                    else:
                        with st.spinner(f"🔄 Translating to {target_language}..."):
                            st.session_state.translated_text = creator.translate_text(
                                st.session_state.processed_text, target_language
                            )
                            if st.session_state.translated_text != st.session_state.processed_text:
                                st.success(f"✅ Successfully translated to {target_language}!")
                            else:
                                st.warning("⚠️ Translation may have failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🎵 Audio Generation")
            
            text_to_convert = st.session_state.translated_text if st.session_state.translated_text else st.session_state.processed_text
            
            if not creator.tts_engine:
                st.error("🎵 TTS Engine not available")
            else:
                word_count_audio = len(text_to_convert.split())
                est_duration = word_count_audio / (voice_settings['speed'] / 60)
                st.info(f"📊 Will process {word_count_audio} words (~{est_duration:.1f} min audio)")
                
                if st.button("🎵 Generate Audio", key="generate_audio", help="Create audiobook from text"):
                    if text_to_convert.strip():
                        with st.spinner("🎧 Creating your audiobook..."):
                            audio_data = creator.create_audio_fixed(text_to_convert, voice_settings)
                            if audio_data:
                                st.session_state.audio_data = audio_data
                                st.success("✅ Audio generated successfully!")
                            else:
                                st.error("❌ Failed to generate audio")
                    else:
                        st.warning("⚠️ No text available to convert")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Audio player and download
        if st.session_state.audio_data:
            st.markdown("### 🎧 Your Audiobook")
            st.markdown('<div class="audio-controls">', unsafe_allow_html=True)
            
            col_audio1, col_audio2 = st.columns([2, 1])
            with col_audio1:
                st.audio(st.session_state.audio_data, format='audio/wav')