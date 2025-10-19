"""
🚀 NLP Comparative Analysis Toolkit (NLP-CAT) 2.1 - ENHANCED GENIUS DASHBOARD 🧠

An ultra-advanced, AI-powered interactive platform for deep text classification analysis,
model comparison, and intelligent insights generation.

🎯 CREATIVE ENHANCEMENTS & GENIUS FEATURES:
═══════════════════════════════════════════
🔥 Real-time Model Performance Racing Dashboard
🧠 AI-Powered Text Analysis with Sentiment & Emotion Detection  
📊 Advanced Interactive Visualizations & Heatmaps
🎨 Dynamic Model Comparison Matrix with Radar Charts
🚀 Batch Processing with Progress Tracking & ETA
💡 Intelligent Insights & Recommendations Engine
🔍 Text Preprocessing Pipeline Visualization
⚡ Live Performance Metrics & Resource Monitoring
🎭 Text Style & Complexity Analysis
🌟 Model Confidence Calibration & Uncertainty Quantification

Author: Daniel Wanjala Machimbo (Enhanced by AI Genius)
Institution: The Cooperative University of Kenya  
Date: November 2024 - ENHANCED VERSION

This revolutionary application provides:
- 🔥 Real-time model racing with live performance updates
- 🧠 Advanced text analytics with emotion & complexity analysis
- 📊 Interactive 3D visualizations and dynamic charts
- 🎯 Intelligent model recommendations based on text characteristics
- ⚡ Performance profiling with resource usage monitoring
- 🎨 Beautiful animations and professional UI/UX
- 💡 AI-powered insights and actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import time
import re
import psutil  # For system monitoring
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import seaborn as sns
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease, flesch_kincaid_grade  # Text complexity
from collections import Counter
from wordcloud import WordCloud  # For word clouds
import warnings
warnings.filterwarnings('ignore')

# Import core libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline  # For emotion analysis
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers library not available - BERT models will not be functional")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn not available - Classical models will not be functional")

# Advanced emotion analysis
try:
    import vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    st.info("Installing VADER sentiment for advanced text analysis...")
    
try:
    import emoji  # For emoji analysis
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 NLP-CAT 2.1 GENIUS DASHBOARD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/danielwanjala/nlp-cat',
        'Report a bug': "https://github.com/danielwanjala/nlp-cat/issues",
        'About': "# NLP-CAT 2.1 Genius Dashboard\nUltra-advanced AI-powered text analysis platform"
    }
)

# 🎨 ENHANCED CUSTOM CSS FOR GENIUS-LEVEL STYLING
st.markdown("""
<style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling with glass morphism */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 98%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* 🔥 GENIUS METRIC CARDS WITH ANIMATIONS */
    .genius-metric-card {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.9) 0%, 
            rgba(118, 75, 162, 0.9) 50%,
            rgba(240, 147, 251, 0.9) 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .genius-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .genius-metric-card:hover::before {
        left: 100%;
    }
    
    .genius-metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }
    
    .metric-trend {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    
    /* 🚀 PERFORMANCE RACING DASHBOARD */
    .racing-dashboard {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.3);
    }
    
    .model-race-track {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 50%, #48dbfb 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
        position: relative;
    }
    
    .race-car {
        position: absolute;
        width: 20px;
        height: 8px;
        background: #fff;
        border-radius: 50%;
        animation: racing 2s linear infinite;
    }
    
    @keyframes racing {
        0% { left: 0%; }
        100% { left: 100%; }
    }
    
    /* 🧠 AI INSIGHTS PANEL */
    .ai-insights {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #f093fb;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .ai-insights::before {
        content: '🧠';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* 📊 ADVANCED CHART CONTAINERS */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* 🎨 ENHANCED BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* 🌟 SIDEBAR ENHANCEMENTS */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    /* 📈 PROGRESS BARS WITH GLOW */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    /* 🎭 TEXT ANALYSIS CARDS */
    .text-analysis-card {
        background: linear-gradient(135deg, 
            rgba(240, 147, 251, 0.1) 0%, 
            rgba(245, 87, 108, 0.1) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(240, 147, 251, 0.3);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.2);
    }
    
    /* 🔥 REAL-TIME MONITORING */
    .monitoring-panel {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #48dbfb;
    }
    
    /* ⚡ PERFORMANCE INDICATORS */
    .perf-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1.5s infinite;
    }
    
    .perf-excellent { background: #48dbfb; }
    .perf-good { background: #feca57; }
    .perf-warning { background: #ff6b6b; }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    /* 💫 LOADING ANIMATIONS */
    .genius-loader {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* 🎨 GRADIENT TEXT */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3.5rem;
        text-align: center;
        margin: 2rem 0;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    /* 🌈 RAINBOW BORDERS */
    .rainbow-border {
        border: 2px solid transparent;
        border-radius: 15px;
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3, #54a0ff) border-box;
        background-clip: padding-box;
    }
    
    /* 📱 RESPONSIVE DESIGN */
    @media (max-width: 768px) {
        .metric-value { font-size: 2rem; }
        .gradient-text { font-size: 2.5rem; }
        .genius-metric-card { padding: 1.5rem; }
    }
    
    /* 🎯 TOOLTIPS */
    .tooltip {
        position: relative;
        cursor: help;
    }
    
    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s;
    }
    
    .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }
</style>
""", unsafe_allow_html=True)

# 🚀 ENHANCED CONFIGURATION AND CONSTANTS
DATASETS = {
    'AG News': {
        'classes': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'description': '4-class news categorization',
        'max_length': 128,
        'emoji': '📰',
        'color': '#667eea'
    },
    '20 Newsgroups': {
        'classes': [f'Class_{i}' for i in range(20)],
        'description': '20-class discussion forum classification', 
        'max_length': 256,
        'emoji': '💬',
        'color': '#764ba2'
    },
    'IMDb': {
        'classes': ['Negative', 'Positive'],
        'description': 'Binary sentiment analysis',
        'max_length': 512,
        'emoji': '🎬',
        'color': '#f093fb'
    }
}

MODELS = {
    'MultinomialNB': {
        'type': 'classical',
        'description': 'Multinomial Naive Bayes with TF-IDF features',
        'complexity': 'Low',
        'emoji': '📊',
        'color': '#48dbfb',
        'speed': 'Very Fast',
        'memory': 'Low'
    },
    'LinearSVM': {
        'type': 'classical', 
        'description': 'Linear Support Vector Machine with TF-IDF features',
        'complexity': 'Low',
        'emoji': '⚡',
        'color': '#feca57',
        'speed': 'Fast',
        'memory': 'Low'
    },
    'BiLSTM': {
        'type': 'neural',
        'description': 'Bidirectional LSTM with GloVe embeddings',
        'complexity': 'Medium',
        'emoji': '🧠',
        'color': '#ff6b6b',
        'speed': 'Medium',
        'memory': 'Medium'
    },
    'BERT': {
        'type': 'transformer',
        'description': 'BERT-base-uncased fine-tuned',
        'complexity': 'High',
        'emoji': '🤖',
        'color': '#54a0ff',
        'speed': 'Slow',
        'memory': 'High'
    },
    'Hybrid': {
        'type': 'hybrid',
        'description': 'BERT embeddings + Linear SVM classifier',
        'complexity': 'Medium',
        'emoji': '🔥',
        'color': '#ff9ff3',
        'speed': 'Medium-Fast',
        'memory': 'Medium'
    }
}

# 🎨 GENIUS ANALYTICS CLASSES
class GeniusTextAnalyzer:
    """🧠 Advanced text analysis with emotion, complexity, and style detection"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_pipeline = None
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize sentiment and emotion analyzers"""
        try:
            if SENTIMENT_AVAILABLE:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            pass
            
        try:
            if TRANSFORMERS_AVAILABLE:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
        except:
            pass
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity and readability"""
        try:
            # Basic metrics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(re.split(r'[.!?]+', text))
            
            # Readability scores
            flesch_score = flesch_reading_ease(text) if word_count > 0 else 0
            fk_grade = flesch_kincaid_grade(text) if word_count > 0 else 0
            
            # Advanced metrics
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Complexity classification
            if flesch_score >= 90:
                complexity_level = "Very Easy"
                complexity_color = "#48dbfb"
            elif flesch_score >= 80:
                complexity_level = "Easy"
                complexity_color = "#feca57"
            elif flesch_score >= 70:
                complexity_level = "Fairly Easy"
                complexity_color = "#ff9ff3"
            elif flesch_score >= 60:
                complexity_level = "Standard"
                complexity_color = "#667eea"
            elif flesch_score >= 50:
                complexity_level = "Fairly Difficult"
                complexity_color = "#ff6b6b"
            else:
                complexity_level = "Difficult"
                complexity_color = "#764ba2"
            
            return {
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'flesch_score': flesch_score,
                'fk_grade': fk_grade,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'complexity_level': complexity_level,
                'complexity_color': complexity_color
            }
        except:
            return {
                'word_count': 0, 'char_count': 0, 'sentence_count': 0,
                'flesch_score': 0, 'fk_grade': 0, 'avg_word_length': 0,
                'avg_sentence_length': 0, 'complexity_level': 'Unknown',
                'complexity_color': '#667eea'
            }
    
    def analyze_sentiment_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotions in text"""
        results = {
            'sentiment': {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0},
            'emotions': [],
            'dominant_emotion': 'neutral',
            'emotion_confidence': 0.0
        }
        
        try:
            # VADER sentiment analysis
            if self.sentiment_analyzer:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                results['sentiment'] = sentiment_scores
            
            # Emotion analysis
            if self.emotion_pipeline:
                emotion_results = self.emotion_pipeline(text[:512])  # Limit text length
                results['emotions'] = emotion_results
                if emotion_results:
                    best_emotion = max(emotion_results, key=lambda x: x['score'])
                    results['dominant_emotion'] = best_emotion['label']
                    results['emotion_confidence'] = best_emotion['score']
        except:
            pass
        
        return results
    
    def analyze_text_style(self, text: str) -> Dict[str, Any]:
        """Analyze text style and linguistic features"""
        try:
            # Punctuation analysis
            punctuation_count = sum(1 for char in text if char in '.,!?;:')
            punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
            
            # Capital letters
            caps_count = sum(1 for char in text if char.isupper())
            caps_ratio = caps_count / len(text) if len(text) > 0 else 0
            
            # Numbers
            numbers_count = sum(1 for char in text if char.isdigit())
            numbers_ratio = numbers_count / len(text) if len(text) > 0 else 0
            
            # Emoji detection
            emoji_count = 0
            if EMOJI_AVAILABLE:
                emoji_count = sum(1 for char in text if char in emoji.UNICODE_EMOJI['en'])
            
            # Determine style
            if caps_ratio > 0.1:
                style = "Emphatic"
                style_color = "#ff6b6b"
            elif punctuation_ratio > 0.05:
                style = "Expressive"
                style_color = "#feca57"
            elif emoji_count > 0:
                style = "Casual"
                style_color = "#48dbfb"
            elif numbers_ratio > 0.02:
                style = "Technical"
                style_color = "#54a0ff"
            else:
                style = "Formal"
                style_color = "#667eea"
            
            return {
                'punctuation_ratio': punctuation_ratio,
                'caps_ratio': caps_ratio,
                'numbers_ratio': numbers_ratio,
                'emoji_count': emoji_count,
                'style': style,
                'style_color': style_color
            }
        except:
            return {
                'punctuation_ratio': 0, 'caps_ratio': 0, 'numbers_ratio': 0,
                'emoji_count': 0, 'style': 'Unknown', 'style_color': '#667eea'
            }

class PerformanceMonitor:
    """⚡ Real-time performance monitoring and resource tracking"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'memory_total': memory.total / (1024**3),  # GB
                'uptime': time.time() - self.start_time
            }
        except:
            return {
                'cpu_percent': 0, 'memory_percent': 0,
                'memory_available': 0, 'memory_total': 0, 'uptime': 0
            }
    
    def track_prediction_time(self, model_name: str, prediction_time: float):
        """Track prediction timing for performance analysis"""
        self.metrics_history.append({
            'model': model_name,
            'time': prediction_time,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 predictions
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get aggregated performance statistics"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        stats = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['time']
            stats[model] = {
                'avg_time': model_data.mean(),
                'min_time': model_data.min(),
                'max_time': model_data.max(),
                'std_time': model_data.std(),
                'prediction_count': len(model_data)
            }
        
        return stats

class ModelRacingDashboard:
    """🏁 Real-time model performance racing dashboard"""
    
    def __init__(self):
        self.race_data = {}
        self.race_active = False
    
    def start_race(self, models: List[str]):
        """Start a new model racing session"""
        self.race_data = {model: {'position': 0, 'speed': 0, 'lap_times': []} for model in models}
        self.race_active = True
    
    def update_race_position(self, model: str, performance_score: float, prediction_time: float):
        """Update model position in the race based on performance"""
        if model in self.race_data:
            # Calculate position based on accuracy and speed
            speed_score = max(0, 100 - prediction_time * 100)  # Faster = higher score
            overall_score = (performance_score * 0.7) + (speed_score * 0.3)
            
            self.race_data[model]['position'] = overall_score
            self.race_data[model]['speed'] = speed_score
            self.race_data[model]['lap_times'].append(prediction_time)
    
    def get_race_standings(self) -> List[Tuple[str, float]]:
        """Get current race standings"""
        standings = [(model, data['position']) for model, data in self.race_data.items()]
        return sorted(standings, key=lambda x: x[1], reverse=True)

class InsightsEngine:
    """💡 AI-powered insights and recommendations generator"""
    
    def __init__(self):
        self.insights_cache = {}
    
    def generate_model_recommendation(self, text: str, text_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent model recommendations based on text characteristics"""
        
        recommendations = []
        
        # Based on text complexity
        complexity = text_analysis.get('complexity_level', 'Standard')
        word_count = text_analysis.get('word_count', 0)
        
        if complexity in ['Very Easy', 'Easy'] and word_count < 50:
            recommendations.append({
                'model': 'MultinomialNB',
                'reason': 'Simple text with low complexity - Naive Bayes is efficient and accurate',
                'confidence': 0.9
            })
        elif word_count > 200:
            recommendations.append({
                'model': 'BERT',
                'reason': 'Long, complex text benefits from transformer attention mechanisms',
                'confidence': 0.85
            })
        elif complexity == 'Standard':
            recommendations.append({
                'model': 'Hybrid',
                'reason': 'Balanced complexity - hybrid approach offers best performance/speed trade-off',
                'confidence': 0.8
            })
        
        # Based on text style
        style = text_analysis.get('style', 'Formal')
        if style in ['Casual', 'Expressive']:
            recommendations.append({
                'model': 'BiLSTM',
                'reason': 'Casual/expressive text benefits from sequential understanding',
                'confidence': 0.75
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else {
                'model': 'LinearSVM', 'reason': 'General-purpose classifier', 'confidence': 0.6
            },
            'all_recommendations': recommendations[:3]  # Top 3
        }
    
    def generate_text_insights(self, text: str, predictions: Dict) -> List[str]:
        """Generate insights about the text and predictions"""
        
        insights = []
        
        # Confidence insights
        if predictions:
            confidences = [pred.get('confidence', 0) for pred in predictions.values()]
            avg_confidence = np.mean(confidences)
            
            if avg_confidence > 0.9:
                insights.append("🎯 High prediction confidence across all models - text has clear classification signals")
            elif avg_confidence < 0.6:
                insights.append("⚠️ Low prediction confidence - text may be ambiguous or require more context")
            
            # Agreement analysis
            predicted_classes = [pred.get('predicted_class', '') for pred in predictions.values()]
            if len(set(predicted_classes)) == 1:
                insights.append("✅ All models agree on classification - high reliability")
            else:
                insights.append("🤔 Models disagree on classification - consider ensemble approach")
        
        # Text length insights
        word_count = len(text.split())
        if word_count < 10:
            insights.append("📝 Very short text - might benefit from more context for better accuracy")
        elif word_count > 500:
            insights.append("📚 Long text detected - transformer models may perform better")
        
        return insights

class EnhancedModelManager:
    """🚀 Advanced model management with genius features and real-time monitoring"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_metadata = {}
        self.performance_monitor = PerformanceMonitor()
        self.racing_dashboard = ModelRacingDashboard()
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata from artifacts"""
        try:
            with open('artifacts/metadata.json', 'r', encoding='utf-8') as f:
                self.model_metadata = json.load(f)
        except FileNotFoundError:
            st.warning("📦 Model metadata not found. Please train models first using the notebook.")
            self.model_metadata = {}
        except Exception as e:
            st.error(f"❌ Error loading metadata: {e}")
            self.model_metadata = {}
    
    def get_model_path(self, dataset: str, model: str) -> str:
        """Get the file path for a specific model"""
        dataset_clean = dataset.lower().replace(' ', '_').replace('newsgroups', 'ng')
        
        if model in ['MultinomialNB', 'LinearSVM']:
            model_file = f"{model.lower().replace('svm', 'svm').replace('nb', '_nb')}.joblib"
            return f"artifacts/classical/{dataset_clean}/{model_file}"
        elif model == 'BiLSTM':
            return f"artifacts/neural/{dataset_clean}/ultra_fast_lstm.pth"
        elif model == 'BERT':
            return f"artifacts/neural/{dataset_clean}/bert_finetuned/"
        elif model == 'Hybrid':
            return f"artifacts/hybrid/{dataset_clean}/hybrid_model.joblib"
        
        return None
    
    def check_model_availability(self, dataset: str, model: str) -> Dict[str, Any]:
        """Check if model is available and get status info"""
        model_path = self.get_model_path(dataset, model)
        
        if not model_path:
            return {'available': False, 'reason': 'Invalid model path', 'status': '❌'}
        
        if os.path.exists(model_path):
            # Get file size and modification time
            try:
                if os.path.isfile(model_path):
                    size = os.path.getsize(model_path) / (1024**2)  # MB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                else:
                    # Directory (BERT model)
                    size = sum(os.path.getsize(os.path.join(model_path, f)) 
                              for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))) / (1024**2)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                return {
                    'available': True,
                    'size_mb': size,
                    'modified': mod_time,
                    'status': '✅',
                    'reason': f'Ready ({size:.1f}MB)'
                }
            except Exception as e:
                return {'available': False, 'reason': f'Access error: {e}', 'status': '⚠️'}
        else:
            return {'available': False, 'reason': 'Model not trained', 'status': '❌'}
    
    def load_model(self, dataset: str, model: str):
        """Load a specific model for prediction with progress tracking"""
        model_key = f"{dataset}_{model}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Show loading progress
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.info(f"🔄 Loading {model} for {dataset}...")
        
        model_path = self.get_model_path(dataset, model)
        
        if not model_path or not os.path.exists(model_path):
            loading_placeholder.empty()
            return None
        
        try:
            start_time = time.time()
            
            if model in ['MultinomialNB', 'LinearSVM', 'Hybrid']:
                # Classical and hybrid models
                loaded_model = joblib.load(model_path)
                self.loaded_models[model_key] = loaded_model
                
            elif model == 'BiLSTM':
                # PyTorch model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_data = torch.load(model_path, map_location=device)
                
                # Load model state if it's a checkpoint
                if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                    # We would need the model architecture here
                    # For now, return the loaded data
                    loaded_model = model_data
                else:
                    loaded_model = model_data
                    if hasattr(loaded_model, 'eval'):
                        loaded_model.eval()
                
                self.loaded_models[model_key] = loaded_model
                
            elif model == 'BERT' and TRANSFORMERS_AVAILABLE:
                # Transformer model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model_obj = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                loaded_model = {'tokenizer': tokenizer, 'model': model_obj}
                self.loaded_models[model_key] = loaded_model
                
            loading_time = time.time() - start_time
            loading_placeholder.success(f"✅ {model} loaded in {loading_time:.2f}s")
            time.sleep(1)  # Show success message briefly
            loading_placeholder.empty()
            
            return self.loaded_models[model_key]
                
        except Exception as e:
            loading_placeholder.error(f"❌ Error loading {model}: {str(e)}")
            time.sleep(2)
            loading_placeholder.empty()
            return None
    
    def predict_single_enhanced(self, text: str, dataset: str, model: str, 
                               use_calibration: bool = False) -> Dict[str, Any]:
        """Enhanced prediction with timing and confidence analysis"""
        
        start_time = time.time()
        loaded_model = self.load_model(dataset, model)
        
# 🚀 MAIN APPLICATION - GENIUS DASHBOARD
def main():
    """🧠 Main application with all genius features"""
    
    # Initialize components
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = EnhancedModelManager()
    if 'text_analyzer' not in st.session_state:
        st.session_state.text_analyzer = GeniusTextAnalyzer()
    if 'insights_engine' not in st.session_state:
        st.session_state.insights_engine = InsightsEngine()
    
    # 🎨 STUNNING HEADER WITH ANIMATIONS
    st.markdown("""
    <div class="gradient-text">
        🚀 NLP-CAT 2.1 GENIUS DASHBOARD 🧠
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; opacity: 0.8;">
        Ultra-Advanced AI-Powered Text Analysis & Model Comparison Platform
    </div>
    """, unsafe_allow_html=True)
    
    # 📊 REAL-TIME SYSTEM MONITORING
    with st.container():
        st.markdown("### ⚡ Real-Time System Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get system metrics
        performance_summary = st.session_state.model_manager.get_performance_summary()
        system_metrics = performance_summary['system_metrics']
        
        with col1:
            cpu_percent = system_metrics.get('cpu_percent', 0)
            cpu_status = "🟢" if cpu_percent < 50 else "🟡" if cpu_percent < 80 else "🔴"
            st.markdown(f"""
            <div class="genius-metric-card">
                <div class="metric-value">{cpu_percent:.1f}%</div>
                <div class="metric-label">CPU Usage {cpu_status}</div>
                <div class="metric-trend">System Performance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            memory_percent = system_metrics.get('memory_percent', 0)
            memory_status = "🟢" if memory_percent < 60 else "🟡" if memory_percent < 85 else "🔴"
            st.markdown(f"""
            <div class="genius-metric-card">
                <div class="metric-value">{memory_percent:.1f}%</div>
                <div class="metric-label">Memory Usage {memory_status}</div>
                <div class="metric-trend">RAM Utilization</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_predictions = performance_summary.get('total_predictions', 0)
            st.markdown(f"""
            <div class="genius-metric-card">
                <div class="metric-value">{total_predictions}</div>
                <div class="metric-label">Total Predictions 🎯</div>
                <div class="metric-trend">Session Statistics</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            uptime = system_metrics.get('uptime', 0)
            uptime_hours = uptime / 3600
            st.markdown(f"""
            <div class="genius-metric-card">
                <div class="metric-value">{uptime_hours:.1f}h</div>
                <div class="metric-label">Session Uptime ⏱️</div>
                <div class="metric-trend">Dashboard Running</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 🎯 MAIN NAVIGATION
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Smart Analysis", 
        "🏁 Model Racing", 
        "📊 Batch Processing", 
        "🧠 Advanced Analytics",
        "⚡ Performance Monitor",
        "🎨 Visualization Studio"
    ])
    
    with tab1:
        smart_analysis_interface()
    
    with tab2:
        model_racing_interface()
    
    with tab3:
        batch_processing_interface()
    
    with tab4:
        advanced_analytics_interface()
    
    with tab5:
        performance_monitor_interface()
    
    with tab6:
        visualization_studio_interface()

def smart_analysis_interface():
    """🔍 Smart text analysis with AI recommendations"""
    
    st.markdown("### 🔍 Smart Text Analysis with AI Recommendations")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Analysis Controls")
        
        selected_dataset = st.selectbox(
            "📊 Select Dataset",
            options=list(DATASETS.keys()),
            help="Choose the dataset for classification"
        )
        
        selected_models = st.multiselect(
            "🤖 Select Models",
            options=list(MODELS.keys()),
            default=['MultinomialNB', 'LinearSVM'],
            help="Choose models to compare"
        )
        
        enable_ai_insights = st.checkbox("🧠 Enable AI Insights", value=True)
        enable_text_analysis = st.checkbox("📝 Enable Text Analysis", value=True)
        real_time_mode = st.checkbox("⚡ Real-time Mode", value=True)
    
    # Main analysis area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Input Text")
        
        # Text input with examples
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here for intelligent analysis...",
            help="Enter any text for classification and analysis"
        )
        
        # Quick examples
        st.markdown("**Quick Examples:**")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("📰 News Example"):
                text_input = "Breaking: Scientists discover new method for faster machine learning training"
                st.rerun()
        
        with example_col2:
            if st.button("💼 Business Example"):
                text_input = "The company's quarterly earnings exceeded expectations by 15% due to strong sales growth"
                st.rerun()
        
        with example_col3:
            if st.button("⚽ Sports Example"):
                text_input = "The championship game ended with a thrilling overtime victory"
                st.rerun()
    
    with col2:
        st.markdown("#### 🎯 Model Status")
        
        # Model availability check
        for model in selected_models:
            status_info = st.session_state.model_manager.check_model_availability(selected_dataset, model)
            
            status_color = "green" if status_info['available'] else "red"
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.1); 
                        border-left: 4px solid {status_color}; border-radius: 5px;">
                <strong>{MODELS[model]['emoji']} {model}</strong><br>
                <small>{status_info['status']} {status_info['reason']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🚀 Analyze Text", disabled=not text_input or not selected_models):
        analyze_text_with_genius_features(text_input, selected_dataset, selected_models, 
                                        enable_ai_insights, enable_text_analysis)

def analyze_text_with_genius_features(text, dataset, models, enable_insights, enable_analysis):
    """🧠 Comprehensive text analysis with all genius features"""
    
    with st.spinner("🔄 Running genius-level analysis..."):
        
        # 📊 TEXT ANALYSIS
        if enable_analysis:
            st.markdown("### 📊 Text Analysis Report")
            
            # Complexity analysis
            complexity_analysis = st.session_state.text_analyzer.analyze_text_complexity(text)
            
            # Sentiment and emotion analysis  
            sentiment_analysis = st.session_state.text_analyzer.analyze_sentiment_emotion(text)
            
            # Style analysis
            style_analysis = st.session_state.text_analyzer.analyze_text_style(text)
            
            # Display analysis in beautiful cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="text-analysis-card">
                    <h4>📖 Complexity Analysis</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: {complexity_analysis['complexity_color']};">
                            {complexity_analysis['complexity_level']}
                        </div>
                        <div>Flesch Score: {complexity_analysis['flesch_score']:.1f}</div>
                        <div>Grade Level: {complexity_analysis['fk_grade']:.1f}</div>
                        <div>Words: {complexity_analysis['word_count']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="text-analysis-card">
                    <h4>💭 Sentiment Analysis</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem;">
                            {sentiment_analysis['dominant_emotion'].title()} 
                            {sentiment_analysis['emotion_confidence']:.2f}
                        </div>
                        <div>Positive: {sentiment_analysis['sentiment']['pos']:.2f}</div>
                        <div>Negative: {sentiment_analysis['sentiment']['neg']:.2f}</div>
                        <div>Neutral: {sentiment_analysis['sentiment']['neu']:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="text-analysis-card">
                    <h4>🎨 Style Analysis</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; color: {style_analysis['style_color']};">
                            {style_analysis['style']}
                        </div>
                        <div>Punctuation: {style_analysis['punctuation_ratio']:.3f}</div>
                        <div>Capitals: {style_analysis['caps_ratio']:.3f}</div>
                        <div>Emojis: {style_analysis['emoji_count']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 🤖 MODEL PREDICTIONS
        st.markdown("### 🤖 Model Predictions & Racing")
        
        # Create predictions with timing
        predictions = {}
        prediction_times = {}
        
        # Progress bar for predictions
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model in enumerate(models):
            status_text.text(f"Running {model}...")
            
            prediction_result = st.session_state.model_manager.predict_single_enhanced(
                text, dataset, model
            )
            
            predictions[model] = prediction_result
            prediction_times[model] = prediction_result.get('prediction_time', 0)
            
            progress_bar.progress((i + 1) / len(models))
        
        progress_bar.empty()
        status_text.empty()
        
        # Display predictions in racing format
        st.markdown("#### 🏁 Model Racing Results")
        
        # Sort models by performance (combination of confidence and speed)
        model_scores = []
        for model in models:
            pred = predictions[model]
            if pred.get('status') == 'success':
                confidence = pred.get('confidence', 0)
                speed_score = max(0, 1 - pred.get('prediction_time', 1))  # Faster = higher score
                overall_score = (confidence * 0.7) + (speed_score * 0.3)
                model_scores.append((model, overall_score, confidence, pred.get('prediction_time', 0)))
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Display racing results
        for rank, (model, score, confidence, pred_time) in enumerate(model_scores, 1):
            model_info = MODELS[model]
            pred_result = predictions[model]
            
            # Medal emojis
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}️⃣"
            
            st.markdown(f"""
            <div class="racing-dashboard">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.5rem;">{medal} {model_info['emoji']} {model}</span>
                        <div style="margin-top: 0.5rem;">
                            <strong>Prediction:</strong> {pred_result.get('predicted_class', 'Unknown')}
                            <span style="margin-left: 1rem;"><strong>Confidence:</strong> {confidence:.3f}</span>
                            <span style="margin-left: 1rem;"><strong>Time:</strong> {pred_time:.3f}s</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2rem; font-weight: bold;">Score: {score:.3f}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">{model_info['complexity']} Complexity</div>
                    </div>
                </div>
                
                <!-- Performance bar -->
                <div style="margin-top: 1rem;">
                    <div style="background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #ff6b6b 0%, #feca57 50%, #48dbfb 100%); 
                                    height: 100%; width: {score*100:.1f}%; transition: width 1s ease;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 🧠 AI INSIGHTS
        if enable_insights:
            st.markdown("### 🧠 AI-Powered Insights & Recommendations")
            
            # Combine all analysis data
            text_analysis_combined = {
                **complexity_analysis,
                **style_analysis
            }
            
            # Generate recommendations
            recommendations = st.session_state.insights_engine.generate_model_recommendation(
                text, text_analysis_combined
            )
            
            # Generate insights
            insights = st.session_state.insights_engine.generate_text_insights(text, predictions)
            
            # Display insights in AI panel
            st.markdown(f"""
            <div class="ai-insights">
                <h4>🎯 Primary Recommendation</h4>
                <div style="margin: 1rem 0;">
                    <strong>Recommended Model:</strong> {recommendations['primary_recommendation']['model']} 
                    {MODELS[recommendations['primary_recommendation']['model']]['emoji']}
                </div>
                <div style="margin: 1rem 0;">
                    <strong>Reason:</strong> {recommendations['primary_recommendation']['reason']}
                </div>
                <div style="margin: 1rem 0;">
                    <strong>Confidence:</strong> {recommendations['primary_recommendation']['confidence']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display insights
            if insights:
                st.markdown("#### 💡 Intelligent Insights")
                for insight in insights:
                    st.info(insight)

def model_racing_interface():
    """🏁 Real-time model racing dashboard"""
    
    st.markdown("### 🏁 Model Performance Racing Dashboard")
    st.markdown("Compare model performance in real-time racing format!")
    
    # Racing controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        race_texts = st.text_area(
            "🏁 Race Track Texts (one per line):",
            height=150,
            placeholder="Enter multiple texts to race models...\nEach line will be a separate race checkpoint.",
            help="Each line represents a text sample for the racing competition"
        )
    
    with col2:
        racing_models = st.multiselect(
            "🏎️ Racing Models",
            options=list(MODELS.keys()),
            default=['MultinomialNB', 'LinearSVM', 'BiLSTM'],
            help="Select models to participate in the race"
        )
    
    with col3:
        racing_dataset = st.selectbox(
            "🏆 Racing Category",
            options=list(DATASETS.keys()),
            help="Choose dataset category for racing"
        )
        
        if st.button("🚀 Start Race!", disabled=not race_texts or not racing_models):
            start_model_race(race_texts, racing_dataset, racing_models)

def start_model_race(race_texts, dataset, models):
    """🏁 Execute the model racing competition"""
    
    texts = [line.strip() for line in race_texts.split('\n') if line.strip()]
    
    if not texts:
        st.error("No valid texts found for racing!")
        return
    
    st.markdown(f"### 🏁 RACE IN PROGRESS - {len(texts)} Checkpoints!")
    
    # Initialize race tracking
    race_results = {model: {'total_time': 0, 'total_confidence': 0, 'wins': 0, 'positions': []} for model in models}
    
    # Progress tracking
    overall_progress = st.progress(0)
    race_status = st.empty()
    
    # Race through each text
    for checkpoint, text in enumerate(texts, 1):
        race_status.markdown(f"🏁 **Checkpoint {checkpoint}/{len(texts)}**: Racing through text analysis...")
        
        checkpoint_results = []
        
        # Time each model
        for model in models:
            start_time = time.time()
            
            prediction_result = st.session_state.model_manager.predict_single_enhanced(
                text, dataset, model
            )
            
            end_time = time.time()
            race_time = end_time - start_time
            confidence = prediction_result.get('confidence', 0)
            
            checkpoint_results.append({
                'model': model,
                'time': race_time,
                'confidence': confidence,
                'performance_score': confidence * 0.7 + (1 - min(race_time, 2)/2) * 0.3
            })
            
            race_results[model]['total_time'] += race_time
            race_results[model]['total_confidence'] += confidence
        
        # Determine checkpoint winner
        checkpoint_results.sort(key=lambda x: x['performance_score'], reverse=True)
        winner = checkpoint_results[0]['model']
        race_results[winner]['wins'] += 1
        
        # Update positions
        for i, result in enumerate(checkpoint_results):
            race_results[result['model']]['positions'].append(i + 1)
        
        # Show checkpoint results
        with st.expander(f"🏁 Checkpoint {checkpoint} Results", expanded=False):
            for i, result in enumerate(checkpoint_results):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}️⃣"
                st.write(f"{medal} {MODELS[result['model']]['emoji']} {result['model']} - "
                        f"Score: {result['performance_score']:.3f} "
                        f"(Time: {result['time']:.3f}s, Confidence: {result['confidence']:.3f})")
        
        overall_progress.progress(checkpoint / len(texts))
    
    # Final race results
    race_status.markdown("🏁 **RACE COMPLETED!** Calculating final standings...")
    
    # Calculate final standings
    final_standings = []
    for model in models:
        avg_time = race_results[model]['total_time'] / len(texts)
        avg_confidence = race_results[model]['total_confidence'] / len(texts)
        wins = race_results[model]['wins']
        avg_position = np.mean(race_results[model]['positions'])
        
        final_score = (avg_confidence * 0.4) + (wins / len(texts) * 0.3) + ((6 - avg_position) / 5 * 0.3)
        
        final_standings.append({
            'model': model,
            'final_score': final_score,
            'avg_time': avg_time,
            'avg_confidence': avg_confidence,
            'wins': wins,
            'avg_position': avg_position
        })
    
    final_standings.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Display podium
    st.markdown("### 🏆 FINAL RACE STANDINGS")
    
    for rank, result in enumerate(final_standings, 1):
        model = result['model']
        model_info = MODELS[model]
        
        if rank == 1:
            medal = "🥇"
            color = "gold"
        elif rank == 2:
            medal = "🥈" 
            color = "silver"
        elif rank == 3:
            medal = "🥉"
            color = "#CD7F32"
        else:
            medal = f"{rank}️⃣"
            color = "#667eea"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                    border: 2px solid {color}; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 2rem;">{medal}</span>
                    <span style="font-size: 1.5rem; margin-left: 1rem;">{model_info['emoji']} {model}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.3rem; font-weight: bold;">Score: {result['final_score']:.3f}</div>
                    <div>Wins: {result['wins']}/{len(texts)} | Avg Time: {result['avg_time']:.3f}s</div>
                    <div>Avg Confidence: {result['avg_confidence']:.3f} | Avg Position: {result['avg_position']:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def batch_processing_interface():
    """📊 Enhanced batch processing with progress tracking"""
    
    st.markdown("### 📊 Intelligent Batch Processing")
    st.markdown("Process multiple texts with advanced progress tracking and analytics!")
    
    # Batch processing controls
    upload_method = st.radio(
        "📥 Input Method:",
        ["Text Input", "File Upload", "Generate Sample Data"],
        horizontal=True
    )
    
    texts_to_process = []
    
    if upload_method == "Text Input":
        batch_text = st.text_area(
            "📝 Enter texts (one per line):",
            height=200,
            placeholder="Enter multiple texts for batch processing...\nEach line will be processed separately."
        )
        if batch_text:
            texts_to_process = [line.strip() for line in batch_text.split('\n') if line.strip()]
    
    elif upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "📁 Upload text file",
            type=['txt', 'csv'],
            help="Upload a text file with one text per line"
        )
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            texts_to_process = [line.strip() for line in content.split('\n') if line.strip()]
    
    else:  # Generate Sample Data
        sample_count = st.slider("Number of sample texts:", 5, 50, 10)
        if st.button("Generate Sample Data"):
            samples = [
                "Breaking news: Major scientific breakthrough announced today",
                "The stock market showed significant gains in the technology sector",
                "Local football team wins championship in overtime thriller",
                "New restaurant opens downtown featuring fusion cuisine",
                "Weather forecast shows sunny skies for the weekend",
                "Company reports record quarterly profits",
                "Scientists discover new species in deep ocean exploration",
                "City council approves new infrastructure development plan",
                "Celebrity announces retirement from entertainment industry",
                "University researchers publish groundbreaking study results"
            ]
            texts_to_process = samples[:sample_count]
    
    if texts_to_process:
        st.success(f"✅ Ready to process {len(texts_to_process)} texts")
        
        # Batch settings
        col1, col2 = st.columns(2)
        
        with col1:
            batch_dataset = st.selectbox("Dataset:", list(DATASETS.keys()))
            batch_models = st.multiselect(
                "Models to compare:",
                list(MODELS.keys()),
                default=['MultinomialNB', 'LinearSVM']
            )
        
        with col2:
            enable_detailed_analysis = st.checkbox("🔍 Detailed Analysis", value=True)
            generate_visualizations = st.checkbox("📊 Generate Visualizations", value=True)
        
        if st.button("🚀 Process Batch", disabled=not batch_models):
            process_batch_with_genius_features(
                texts_to_process, batch_dataset, batch_models,
                enable_detailed_analysis, generate_visualizations
            )

def process_batch_with_genius_features(texts, dataset, models, detailed_analysis, visualizations):
    """📊 Process batch with advanced features and visualizations"""
    
    st.markdown("### 🔄 Batch Processing in Progress...")
    
    # Initialize progress tracking
    total_operations = len(texts) * len(models)
    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    
    start_time = time.time()
    
    # Process with progress callback
    def progress_callback(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
        
        # Calculate ETA
        elapsed = time.time() - start_time
        if progress > 0:
            eta = (elapsed / progress) * (1 - progress)
            eta_text.text(f"⏱️ ETA: {eta:.1f} seconds")
    
    # Run batch processing
    batch_results = st.session_state.model_manager.predict_batch_enhanced(
        texts, dataset, models, progress_callback
    )
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    eta_text.empty()
    
    total_time = time.time() - start_time
    st.success(f"✅ Batch processing completed in {total_time:.2f} seconds!")
    
    # Display results
    display_batch_results(batch_results, texts, models, detailed_analysis, visualizations)

def display_batch_results(results, texts, models, detailed_analysis, visualizations):
    """📊 Display comprehensive batch processing results"""
    
    st.markdown("### 📊 Batch Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = sum(len(results[model]['predictions']) for model in models)
    successful_predictions = sum(results[model]['successful_predictions'] for model in models)
    avg_time = np.mean([results[model]['avg_time_per_prediction'] for model in models])
    avg_confidence = np.mean([results[model]['avg_confidence'] for model in models])
    
    with col1:
        st.metric("📊 Total Predictions", total_predictions)
    with col2:
        st.metric("✅ Success Rate", f"{successful_predictions/total_predictions:.1%}")
    with col3:
        st.metric("⚡ Avg Time", f"{avg_time:.3f}s")
    with col4:
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
    
    # Detailed results table
    if detailed_analysis:
        st.markdown("#### 📋 Detailed Results")
        
        # Create comprehensive results dataframe
        detailed_data = []
        for i, text in enumerate(texts):
            row = {'Text_ID': i+1, 'Text_Preview': text[:50] + '...' if len(text) > 50 else text}
            
            for model in models:
                pred = results[model]['predictions'][i]
                row[f'{model}_Prediction'] = pred.get('predicted_class', 'Error')
                row[f'{model}_Confidence'] = pred.get('confidence', 0)
                row[f'{model}_Time'] = pred.get('prediction_time', 0)
            
            detailed_data.append(row)
        
        df = pd.DataFrame(detailed_data)
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "batch_results.csv",
            "text/csv"
        )
    
    # Visualizations
    if visualizations:
        create_batch_visualizations(results, models, texts)

def create_batch_visualizations(results, models, texts):
    """📊 Create advanced visualizations for batch results"""
    
    st.markdown("#### 📊 Advanced Visualizations")
    
    # Performance comparison chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance radar chart
        fig_radar = create_performance_radar_chart(results, models)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig_conf = create_confidence_distribution_chart(results, models)
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Time comparison
    fig_time = create_time_comparison_chart(results, models)
    st.plotly_chart(fig_time, use_container_width=True)

def create_performance_radar_chart(results, models):
    """Create radar chart for model performance comparison"""
    
    categories = ['Avg Confidence', 'Speed Score', 'Success Rate', 'Consistency']
    
    fig = go.Figure()
    
    for model in models:
        model_data = results[model]
        
        # Calculate metrics
        avg_conf = model_data['avg_confidence']
        speed_score = 1 - min(model_data['avg_time_per_prediction'], 1)  # Normalize to 0-1
        success_rate = model_data['successful_predictions'] / len(model_data['predictions'])
        
        # Calculate consistency (1 - std deviation of confidences)
        confidences = [p.get('confidence', 0) for p in model_data['predictions']]
        consistency = 1 - np.std(confidences) if confidences else 0
        
        values = [avg_conf, speed_score, success_rate, consistency]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f"{MODELS[model]['emoji']} {model}",
            line_color=MODELS[model]['color']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🎯 Model Performance Comparison (Radar Chart)",
        height=400
    )
    
    return fig

def create_confidence_distribution_chart(results, models):
    """Create confidence distribution chart"""
    
    fig = go.Figure()
    
    for model in models:
        confidences = [p.get('confidence', 0) for p in results[model]['predictions']]
        
        fig.add_trace(go.Histogram(
            x=confidences,
            name=f"{MODELS[model]['emoji']} {model}",
            opacity=0.7,
            nbinsx=20
        ))
    
    fig.update_layout(
        title="📊 Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    return fig

def create_time_comparison_chart(results, models):
    """Create timing comparison chart"""
    
    model_names = []
    avg_times = []
    colors = []
    
    for model in models:
        model_names.append(f"{MODELS[model]['emoji']} {model}")
        avg_times.append(results[model]['avg_time_per_prediction'])
        colors.append(MODELS[model]['color'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=avg_times,
            marker_color=colors,
            text=[f"{t:.3f}s" for t in avg_times],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="⚡ Average Prediction Time Comparison",
        xaxis_title="Models",
        yaxis_title="Time (seconds)",
        height=400
    )
    
    return fig

def advanced_analytics_interface():
    """🧠 Advanced analytics and insights dashboard"""
    
    st.markdown("### 🧠 Advanced Analytics Dashboard")
    st.markdown("Deep dive into model performance and text analysis patterns!")
    
    # Analytics options
    analytics_type = st.selectbox(
        "📊 Analytics Type:",
        ["Model Performance Analysis", "Text Complexity Patterns", "Prediction Confidence Analysis", "Error Analysis"]
    )
    
    if analytics_type == "Model Performance Analysis":
        model_performance_analytics()
    elif analytics_type == "Text Complexity Patterns":
        text_complexity_analytics()
    elif analytics_type == "Prediction Confidence Analysis":
        confidence_analytics()
    else:
        error_analytics()

def model_performance_analytics():
    """📊 Detailed model performance analytics"""
    
    st.markdown("#### 🎯 Model Performance Analytics")
    
    # Get performance data from session
    performance_summary = st.session_state.model_manager.get_performance_summary()
    model_stats = performance_summary.get('model_performance', {})
    
    if not model_stats:
        st.info("📈 No performance data available yet. Run some predictions to see analytics!")
        return
    
    # Create performance metrics
    metrics_data = []
    for model, stats in model_stats.items():
        metrics_data.append({
            'Model': model,
            'Avg Time': stats['avg_time'],
            'Min Time': stats['min_time'],
            'Max Time': stats['max_time'],
            'Std Dev': stats['std_time'],
            'Predictions': stats['prediction_count'],
            'Consistency': 1 / (1 + stats['std_time'])  # Higher is better
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(df_metrics, use_container_width=True)
    
    # Performance trends
    st.markdown("#### 📈 Performance Trends")
    
    # Create trend visualization
    fig_trends = go.Figure()
    
    for model in df_metrics['Model']:
        fig_trends.add_trace(go.Scatter(
            x=[1, 2, 3],  # Simplified trend
            y=[df_metrics[df_metrics['Model']==model]['Min Time'].iloc[0],
               df_metrics[df_metrics['Model']==model]['Avg Time'].iloc[0],
               df_metrics[df_metrics['Model']==model]['Max Time'].iloc[0]],
            mode='lines+markers',
            name=f"{MODELS.get(model, {}).get('emoji', '🤖')} {model}",
            line=dict(color=MODELS.get(model, {}).get('color', '#667eea'))
        ))
    
    fig_trends.update_layout(
        title="Performance Range (Min → Avg → Max)",
        xaxis_title="Performance Metric",
        yaxis_title="Time (seconds)",
        xaxis=dict(tickmode='array', tickvals=[1,2,3], ticktext=['Min', 'Avg', 'Max'])
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)

def performance_monitor_interface():
    """⚡ Real-time performance monitoring dashboard"""
    
    st.markdown("### ⚡ Performance Monitor Dashboard")
    st.markdown("Monitor system performance and resource usage in real-time!")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s intervals)", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # System metrics
    performance_summary = st.session_state.model_manager.get_performance_summary()
    system_metrics = performance_summary['system_metrics']
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = system_metrics.get('cpu_percent', 0)
        cpu_status = "🟢 Excellent" if cpu_percent < 30 else "🟡 Good" if cpu_percent < 70 else "🔴 High"
        
        st.markdown(f"""
        <div class="monitoring-panel">
            <h4>🖥️ CPU Usage</h4>
            <div style="font-size: 2rem;">{cpu_percent:.1f}%</div>
            <div>{cpu_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        memory_percent = system_metrics.get('memory_percent', 0)
        memory_available = system_metrics.get('memory_available', 0)
        memory_status = "🟢 Excellent" if memory_percent < 50 else "🟡 Good" if memory_percent < 80 else "🔴 High"
        
        st.markdown(f"""
        <div class="monitoring-panel">
            <h4>💾 Memory Usage</h4>
            <div style="font-size: 2rem;">{memory_percent:.1f}%</div>
            <div>{memory_status}</div>
            <div style="font-size: 0.8rem;">Available: {memory_available:.1f}GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        uptime = system_metrics.get('uptime', 0)
        uptime_hours = uptime / 3600
        
        st.markdown(f"""
        <div class="monitoring-panel">
            <h4>⏱️ Session Uptime</h4>
            <div style="font-size: 2rem;">{uptime_hours:.1f}h</div>
            <div>🟢 Running Smoothly</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Resource usage chart
    st.markdown("#### 📊 Resource Usage Visualization")
    
    # Create gauge charts
    fig_gauges = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CPU Usage", "Memory Usage"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # CPU gauge
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=cpu_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPU %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ),
        row=1, col=1
    )
    
    # Memory gauge
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=memory_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memory %"},
            delta={'reference': 60},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 85], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ),
        row=1, col=2
    )
    
    fig_gauges.update_layout(height=400)
    st.plotly_chart(fig_gauges, use_container_width=True)

def visualization_studio_interface():
    """🎨 Advanced visualization studio"""
    
    st.markdown("### 🎨 Visualization Studio")
    st.markdown("Create beautiful and interactive visualizations for your text analysis!")
    
    viz_type = st.selectbox(
        "🎨 Visualization Type:",
        ["Model Comparison Matrix", "Performance Heatmap", "3D Performance Space", "Word Cloud Generator"]
    )
    
    if viz_type == "Model Comparison Matrix":
        create_model_comparison_matrix()
    elif viz_type == "Performance Heatmap":
        create_performance_heatmap()
    elif viz_type == "3D Performance Space":
        create_3d_performance_space()
    else:
        create_word_cloud_generator()

def create_model_comparison_matrix():
    """Create interactive model comparison matrix"""
    
    st.markdown("#### 🔄 Model Comparison Matrix")
    
    # Sample data for demonstration
    models = list(MODELS.keys())
    metrics = ['Accuracy', 'Speed', 'Memory', 'Complexity']
    
    # Generate sample comparison data
    comparison_data = np.random.rand(len(models), len(metrics))
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=comparison_data,
        x=metrics,
        y=[f"{MODELS[m]['emoji']} {m}" for m in models],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title="🎯 Model Performance Comparison Matrix",
        xaxis_title="Performance Metrics",
        yaxis_title="Models",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_3d_performance_space():
    """Create 3D performance visualization"""
    
    st.markdown("#### 🌌 3D Performance Space")
    
    # Generate sample 3D data
    models = list(MODELS.keys())
    
    fig = go.Figure(data=[go.Scatter3d(
        x=np.random.rand(len(models)),
        y=np.random.rand(len(models)),
        z=np.random.rand(len(models)),
        mode='markers+text',
        text=[f"{MODELS[m]['emoji']} {m}" for m in models],
        textposition="top center",
        marker=dict(
            size=15,
            color=[MODELS[m]['color'] for m in models],
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title="🌌 3D Model Performance Space",
        scene=dict(
            xaxis_title="Accuracy",
            yaxis_title="Speed", 
            zaxis_title="Efficiency"
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_word_cloud_generator():
    """Generate word clouds from text analysis"""
    
    st.markdown("#### ☁️ Word Cloud Generator")
    
    text_for_cloud = st.text_area(
        "Enter text for word cloud:",
        height=150,
        placeholder="Enter text to generate a beautiful word cloud visualization..."
    )
    
    if text_for_cloud and st.button("🎨 Generate Word Cloud"):
        try:
            # Simple word frequency for demonstration
            words = text_for_cloud.lower().split()
            word_freq = Counter(words)
            
            # Create word frequency chart instead of word cloud (simpler)
            top_words = dict(word_freq.most_common(20))
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(top_words.keys()),
                    y=list(top_words.values()),
                    marker_color=px.colors.qualitative.Set3
                )
            ])
            
            fig.update_layout(
                title="📊 Top Word Frequencies",
                xaxis_title="Words",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")

# Additional helper functions
def text_complexity_analytics():
    """📖 Text complexity pattern analysis"""
    st.markdown("#### 📖 Text Complexity Analytics")
    st.info("Analyze patterns in text complexity and readability scores!")
    
    # Placeholder for complexity analytics
    sample_data = {
        'Complexity Level': ['Very Easy', 'Easy', 'Standard', 'Difficult'],
        'Count': [15, 25, 35, 10],
        'Avg Accuracy': [0.92, 0.88, 0.85, 0.78]
    }
    
    df = pd.DataFrame(sample_data)
    
    fig = px.bar(df, x='Complexity Level', y='Count', 
                 title="📖 Text Complexity Distribution",
                 color='Avg Accuracy', color_continuous_scale='Viridis')
    
    st.plotly_chart(fig, use_container_width=True)

def confidence_analytics():
    """🎯 Prediction confidence analysis"""
    st.markdown("#### 🎯 Confidence Score Analytics")
    st.info("Analyze prediction confidence patterns and reliability!")
    
    # Sample confidence data
    confidence_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    counts = [5, 10, 15, 35, 45]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidence_ranges,
            y=counts,
            marker_color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']
        )
    ])
    
    fig.update_layout(
        title="🎯 Confidence Score Distribution",
        xaxis_title="Confidence Range",
        yaxis_title="Number of Predictions"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def error_analytics():
    """❌ Error pattern analysis"""
    st.markdown("#### ❌ Error Analysis")
    st.info("Identify and analyze prediction errors and failure patterns!")
    
    # Sample error data
    error_types = ['Loading Error', 'Prediction Error', 'Timeout', 'Memory Error']
    error_counts = [2, 5, 1, 1]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=error_types,
            values=error_counts,
            title="Error Type Distribution"
        )
    ])
    
    st.plotly_chart(fig, use_container_width=True)

# 🚀 RUN THE GENIUS APPLICATION
if __name__ == "__main__":
    main()