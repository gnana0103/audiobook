import streamlit as st
import pyttsx3
import io
import base64
import tempfile
import os
from PIL import Image
import pytesseract
import PyPDF2
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
from textstat import flesch_reading_ease
import networkx as nx
from collections import Counter
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

class AudiobookCreator:
    def __init__(self):
        self.translator = GoogleTranslator()
        self.tts_engine = pyttsx3.init()
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_image(self, image_file):
        """Extract text from image using OCR"""
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return ""
    
    def translate_text(self, text, target_language):
        """Translate text to target language"""
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
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    try:
                        translator = GoogleTranslator(source='auto', target=target_lang)
                        translated = translator.translate(chunk)
                        translated_chunks.append(translated)
                    except Exception as chunk_error:
                        st.warning(f"Failed to translate chunk {i+1}: {str(chunk_error)}")
                        translated_chunks.append(chunk)  # Keep original if translation fails
                
                progress_bar.progress((i + 1) / len(chunks))
            
            return ' '.join(translated_chunks)
            
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text
    
    def create_audio(self, text, voice_settings):
        """Create audio from text with voice settings"""
        try:
            # Configure TTS engine
            voices = self.tts_engine.getProperty('voices')
            
            # Set voice
            if voice_settings['gender'] == 'Female' and len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
            else:
                self.tts_engine.setProperty('voice', voices[0].id)
            
            # Set rate and volume
            self.tts_engine.setProperty('rate', voice_settings['speed'])
            self.tts_engine.setProperty('volume', voice_settings['volume'])
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                self.tts_engine.save_to_file(text, tmp_file.name)
                self.tts_engine.runAndWait()
                
                # Read and return audio data
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                os.unlink(tmp_file.name)
                return audio_data
                
        except Exception as e:
            st.error(f"Audio creation error: {str(e)}")
            return None
    
    def summarize_text(self, text):
        """Create text summary with key points"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Simple extractive summarization
        word_freq = Counter(text.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        for word in stop_words:
            word_freq.pop(word, None)
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words)
            sentence_scores[i] = score
        
        # Get top sentences
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:5]
        summary_sentences = [sentences[i] for i in sorted(top_sentences)]
        
        return '. '.join(summary_sentences) + '.'
    
    def create_mindmap_data(self, text):
        """Create data for mind map visualization"""
        # Extract key topics and concepts
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter(words)
        
        # Remove common words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
        
        for word in stop_words:
            word_freq.pop(word, None)
        
        # Get top concepts
        top_concepts = dict(word_freq.most_common(10))
        return top_concepts

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎧 AI Audiobook Creator</h1>
        <p style="color: white; font-size: 1.2rem;">Transform Text, PDFs & Images into Interactive Audiobooks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize audiobook creator
    creator = AudiobookCreator()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Audio Settings")
        
        # Voice settings
        st.markdown("### Voice Configuration")
        voice_gender = st.selectbox("🎭 Voice Gender", ["Male", "Female"])
        voice_speed = st.slider("🏃 Speech Speed", 100, 300, 200)
        voice_volume = st.slider("🔊 Volume", 0.1, 1.0, 0.9)
        
        # Language settings
        st.markdown("### 🌍 Language Settings")
        target_language = st.selectbox(
            "Translation Language", 
            ["English", "Tamil", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese"]
        )
        
        voice_settings = {
            'gender': voice_gender,
            'speed': voice_speed,
            'volume': voice_volume
        }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input methods
        st.markdown("## 📝 Input Methods")
        
        tab1, tab2, tab3 = st.tabs(["📄 Text Input", "📋 PDF Upload", "🖼️ Image Upload"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            input_text = st.text_area(
                "Enter your text here:",
                height=200,
                placeholder="Type or paste your text here to create an audiobook..."
            )
            if st.button("🔄 Process Text", key="process_text"):
                if input_text:
                    st.session_state.processed_text = input_text
                    st.success("✅ Text processed successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader("📎 Upload PDF", type=['pdf'])
            if uploaded_pdf and st.button("🔍 Extract from PDF", key="extract_pdf"):
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = creator.extract_text_from_pdf(uploaded_pdf)
                    if extracted_text:
                        st.session_state.processed_text = extracted_text
                        st.success("✅ PDF processed successfully!")
                        st.text_area("Extracted Text Preview:", extracted_text[:500] + "...", height=150)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            uploaded_image = st.file_uploader("🖼️ Upload Image", type=['png', 'jpg', 'jpeg'])
            if uploaded_image and st.button("🔍 Extract from Image", key="extract_image"):
                with st.spinner("Extracting text from image..."):
                    extracted_text = creator.extract_text_from_image(uploaded_image)
                    if extracted_text:
                        st.session_state.processed_text = extracted_text
                        st.success("✅ Image processed successfully!")
                        st.text_area("Extracted Text Preview:", extracted_text[:500], height=150)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.processed_text:
            # Text statistics
            st.markdown("## 📊 Text Statistics")
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            text_length = len(st.session_state.processed_text)
            word_count = len(st.session_state.processed_text.split())
            estimated_duration = word_count / 150  # Average reading speed
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("📝 Words", word_count)
                st.metric("⏱️ Est. Duration", f"{estimated_duration:.1f} min")
            with col_stat2:
                st.metric("📏 Characters", text_length)
                reading_level = flesch_reading_ease(st.session_state.processed_text)
                st.metric("📖 Reading Level", f"{reading_level:.0f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing and Audio Generation
    if st.session_state.processed_text:
        st.markdown("## 🎯 Audio Generation & Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌍 Translate Text", key="translate"):
                with st.spinner(f"Translating to {target_language}..."):
                    st.session_state.translated_text = creator.translate_text(
                        st.session_state.processed_text, target_language
                    )
                    st.success(f"✅ Translated to {target_language}!")
        
        with col2:
            text_to_convert = st.session_state.translated_text if st.session_state.translated_text else st.session_state.processed_text
            
            if st.button("🎵 Generate Audio", key="generate_audio"):
                if text_to_convert:
                    with st.spinner("Creating audiobook..."):
                        audio_data = creator.create_audio(text_to_convert, voice_settings)
                        if audio_data:
                            st.markdown('<div class="audio-controls">', unsafe_allow_html=True)
                            st.audio(audio_data, format='audio/wav')
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download button
                            b64_audio = base64.b64encode(audio_data).decode()
                            href = f'<a href="data:audio/wav;base64,{b64_audio}" download="audiobook.wav" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 10px 20px; border-radius: 25px; text-decoration: none; font-weight: bold;">📥 Download Audio</a>'
                            st.markdown(href, unsafe_allow_html=True)
        
        # Display translated text if available
        if st.session_state.translated_text:
            st.markdown("### 🌍 Translated Text")
            st.text_area("", st.session_state.translated_text, height=200, key="translated_display")
        
        # Text Summary and Mind Map
        st.markdown("## 📋 Text Analysis & Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            st.markdown("### 📄 Summary")
            
            summary = creator.summarize_text(st.session_state.processed_text)
            st.markdown(summary)
            
            # Key points
            sentences = summary.split('.')
            st.markdown("### 🎯 Key Points:")
            for i, sentence in enumerate(sentences[:5], 1):
                if sentence.strip():
                    st.markdown(f"• **Point {i}:** {sentence.strip()}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="mindmap-container">', unsafe_allow_html=True)
            st.markdown("### 🧠 Concept Mind Map")
            
            # Create mind map visualization
            mindmap_data = creator.create_mindmap_data(st.session_state.processed_text)
            
            if mindmap_data:
                # Create a simple network-style visualization
                fig = go.Figure()
                
                # Center node
                fig.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers+text',
                    marker=dict(size=50, color='#667eea'),
                    text=['Main Topic'],
                    textposition='middle center',
                    name='Center'
                ))
                
                # Concept nodes
                import math
                angles = [2 * math.pi * i / len(mindmap_data) for i in range(len(mindmap_data))]
                
                for i, (concept, frequency) in enumerate(mindmap_data.items()):
                    x = 2 * math.cos(angles[i])
                    y = 2 * math.sin(angles[i])
                    
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=frequency*3, color='#764ba2', opacity=0.7),
                        text=[concept],
                        textposition='middle center',
                        name=concept
                    ))
                    
                    # Add connection lines
                    fig.add_trace(go.Scatter(
                        x=[0, x], y=[0, y],
                        mode='lines',
                        line=dict(color='#cccccc', width=2),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Word frequency chart
                st.markdown("### 📊 Key Concepts Frequency")
                concept_df = pd.DataFrame(list(mindmap_data.items()), columns=['Concept', 'Frequency'])
                fig_bar = px.bar(
                    concept_df, 
                    x='Concept', 
                    y='Frequency',
                    color='Frequency',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional features
        st.markdown("## 🔧 Additional Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Text Analytics", key="analytics"):
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                
                # Basic analytics
                sentences = len(st.session_state.processed_text.split('.'))
                paragraphs = len([p for p in st.session_state.processed_text.split('\n') if p.strip()])
                avg_sentence_length = word_count / sentences if sentences > 0 else 0
                
                st.markdown("### 📊 Text Analytics")
                st.metric("Sentences", sentences)
                st.metric("Paragraphs", paragraphs)
                st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("☁️ Word Cloud", key="wordcloud"):
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                
                try:
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=400, 
                        height=200, 
                        background_color='white',
                        colormap='viridis'
                    ).generate(st.session_state.processed_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error("Could not generate word cloud")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            if st.button("📑 Export Summary", key="export"):
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                
                # Create exportable summary
                export_text = f"""
AUDIOBOOK SUMMARY REPORT
========================

Original Text Length: {len(st.session_state.processed_text)} characters
Word Count: {len(st.session_state.processed_text.split())} words
Estimated Reading Time: {len(st.session_state.processed_text.split()) / 150:.1f} minutes

SUMMARY:
{creator.summarize_text(st.session_state.processed_text)}

KEY CONCEPTS:
{chr(10).join([f"• {concept}: {freq}" for concept, freq in creator.create_mindmap_data(st.session_state.processed_text).items()])}
                """
                
                st.download_button(
                    label="📥 Download Summary Report",
                    data=export_text,
                    file_name="audiobook_summary.txt",
                    mime="text/plain"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
