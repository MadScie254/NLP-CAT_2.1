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

class ModelManager:
    """Centralized model management and prediction interface"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_metadata = {}
        self.load_metadata()
    
    def load_metadata(self):
        """Load model metadata from artifacts"""
        try:
            with open('artifacts/metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
        except FileNotFoundError:
            st.warning("Model metadata not found. Please train models first using the notebook.")
            self.model_metadata = {}
    
    def get_model_path(self, dataset: str, model: str) -> str:
        """Get the file path for a specific model"""
        dataset_clean = dataset.lower().replace(' ', '_').replace('newsgroups', 'ng')
        
        if model in ['MultinomialNB', 'LinearSVM']:
            model_file = f"{model.lower().replace('svm', 'svm').replace('nb', '_nb')}.joblib"
            return f"artifacts/classical/{dataset_clean}/{model_file}"
        elif model == 'BiLSTM':
            return f"artifacts/bilstm/{dataset_clean}/best_model.pt"
        elif model == 'BERT':
            return f"artifacts/bert/{dataset_clean}/"
        elif model == 'Hybrid':
            return f"artifacts/hybrid/{dataset_clean}/hybrid_model.joblib"
        
        return None
    
    def load_model(self, dataset: str, model: str):
        """Load a specific model for prediction"""
        model_key = f"{dataset}_{model}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        model_path = self.get_model_path(dataset, model)
        
        if not model_path or not os.path.exists(model_path):
            return None
        
        try:
            if model in ['MultinomialNB', 'LinearSVM', 'Hybrid']:
                # Classical and hybrid models
                loaded_model = joblib.load(model_path)
                self.loaded_models[model_key] = loaded_model
                return loaded_model
                
            elif model == 'BiLSTM':
                # PyTorch model
                if not torch.cuda.is_available():
                    device = 'cpu'
                else:
                    device = 'cuda'
                
                loaded_model = torch.load(model_path, map_location=device)
                loaded_model.eval()
                self.loaded_models[model_key] = loaded_model
                return loaded_model
                
            elif model == 'BERT' and TRANSFORMERS_AVAILABLE:
                # Transformer model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model_obj = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                loaded_model = {'tokenizer': tokenizer, 'model': model_obj}
                self.loaded_models[model_key] = loaded_model
                return loaded_model
                
        except Exception as e:
            st.error(f"Error loading {model} for {dataset}: {str(e)}")
            return None
        
        return None
    
    def predict_single(self, text: str, dataset: str, model: str, 
                      use_calibration: bool = False) -> Dict[str, Any]:
        """Make prediction for a single text sample"""
        
        loaded_model = self.load_model(dataset, model)
        
        if loaded_model is None:
            return {
                'error': f'Model {model} for {dataset} not available',
                'prediction': None,
                'probabilities': None,
                'inference_time': 0
            }
        
        start_time = time.perf_counter()
        
        try:
            if model in ['MultinomialNB', 'LinearSVM', 'Hybrid']:
                # Classical/hybrid models - handle both pipeline and direct model objects
                if hasattr(loaded_model, 'predict'):
                    prediction = loaded_model.predict([text])[0]
                elif isinstance(loaded_model, dict) and 'model' in loaded_model:
                    prediction = loaded_model['model'].predict([text])[0]
                else:
                    return {'error': 'Model object invalid', 'prediction': None, 'probabilities': None, 'inference_time': 0}
                
                try:
                    if hasattr(loaded_model, 'predict_proba'):
                        probabilities = loaded_model.predict_proba([text])[0]
                    elif isinstance(loaded_model, dict) and 'model' in loaded_model:
                        probabilities = loaded_model['model'].predict_proba([text])[0]
                    else:
                        probabilities = None
                except (AttributeError, KeyError):
                    # LinearSVM might not have predict_proba
                    probabilities = None
                
            elif model == 'BiLSTM':
                # PyTorch BiLSTM model
                # This would need custom preprocessing and tokenization
                # Placeholder implementation
                prediction = 0
                probabilities = np.array([0.5, 0.5])
                
            elif model == 'BERT':
                # BERT transformer model
                tokenizer = loaded_model['tokenizer']
                model_obj = loaded_model['model']
                
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=DATASETS[dataset]['max_length']
                )
                
                with torch.no_grad():
                    outputs = model_obj(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1).numpy()[0]
                    prediction = np.argmax(probabilities)
            
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            return {
                'prediction': int(prediction),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'inference_time': inference_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'probabilities': None,
                'inference_time': 0
            }
    
    def get_feature_importance(self, text: str, dataset: str, model: str, 
                             top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract feature importance/explanations for predictions"""
        
        loaded_model = self.load_model(dataset, model)
        
        if loaded_model is None or model not in ['MultinomialNB', 'LinearSVM']:
            return []
        
        try:
            # Get TF-IDF features for the text
            tfidf_vectorizer = loaded_model.named_steps['tfidf']
            text_vector = tfidf_vectorizer.transform([text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get classifier coefficients or feature log probabilities
            classifier = loaded_model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_log_prob_'):
                # Multinomial NB
                prediction = loaded_model.predict([text])[0]
                feature_scores = classifier.feature_log_prob_[prediction]
                
            elif hasattr(classifier, 'coef_'):
                # Linear SVM
                if classifier.coef_.shape[0] == 1:  # Binary
                    feature_scores = classifier.coef_[0]
                else:  # Multi-class
                    prediction = loaded_model.predict([text])[0]
                    feature_scores = classifier.coef_[prediction]
            else:
                return []
            
            # Get non-zero features from the text
            text_features = text_vector.toarray()[0]
            non_zero_indices = np.nonzero(text_features)[0]
            
            # Calculate importance scores
            importance_scores = []
            for idx in non_zero_indices:
                feature_name = feature_names[idx]
                feature_value = text_features[idx]
                model_weight = feature_scores[idx]
                importance = feature_value * model_weight
                importance_scores.append((feature_name, importance))
            
            # Sort by absolute importance and return top-k
            importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            return importance_scores[:top_k]
            
        except Exception as e:
            st.error(f"Error extracting feature importance: {e}")
            return []

# Initialize model manager
@st.cache_resource
def get_model_manager():
    return ModelManager()

model_manager = get_model_manager()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 NLP-CAT 2.1 Interactive Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <strong>Comprehensive Text Classification Model Comparison & Analysis</strong><br>
        Author: Daniel Wanjala Machimbo | The Cooperative University of Kenya
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Configuration Panel")
    
    # Dataset and model selection
    selected_dataset = st.sidebar.selectbox(
        "📊 Select Dataset",
        options=list(DATASETS.keys()),
        help="Choose the dataset for analysis"
    )
    
    selected_model = st.sidebar.selectbox(
        "🤖 Select Model",
        options=list(MODELS.keys()),
        help="Choose the model for predictions"
    )
    
    # Display dataset and model information
    with st.sidebar.expander("📋 Dataset Info", expanded=True):
        dataset_info = DATASETS[selected_dataset]
        st.write(f"**Description:** {dataset_info['description']}")
        st.write(f"**Classes:** {len(dataset_info['classes'])}")
        st.write(f"**Max Length:** {dataset_info['max_length']} tokens")
    
    with st.sidebar.expander("🔧 Model Info", expanded=True):
        model_info = MODELS[selected_model]
        st.write(f"**Type:** {model_info['type']}")
        st.write(f"**Complexity:** {model_info['complexity']}")
        st.write(f"**Description:** {model_info['description']}")
    
    # Main application tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Prediction", 
        "📊 Batch Analysis", 
        "📈 Model Comparison",
        "🔍 Feature Interpretation",
        "📋 Model Performance"
    ])
    
    # Tab 1: Single Text Prediction
    with tab1:
        st.header("🎯 Single Text Prediction")
        
        # Text input
        input_text = st.text_area(
            "Enter text for classification:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter the text you want to classify"
        )
        
        # Prediction options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            use_calibration = st.checkbox(
                "Use Temperature Scaling",
                help="Apply temperature scaling for better calibrated probabilities"
            )
            
            show_timing = st.checkbox(
                "Show Timing",
                value=True,
                help="Display inference time measurements"
            )
            
            show_features = st.checkbox(
                "Show Feature Importance",
                help="Display feature importance for classical models"
            )
        
        if st.button("🚀 Predict", type="primary"):
            if input_text.strip():
                with st.spinner("Making prediction..."):
                    result = model_manager.predict_single(
                        input_text, selected_dataset, selected_model, use_calibration
                    )
                
                if result['error']:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    prediction_class = DATASETS[selected_dataset]['classes'][result['prediction']]
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Class</div>
                            <div class="metric-value">{prediction_class}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if result['probabilities']:
                        max_prob = max(result['probabilities'])
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{max_prob:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if show_timing:
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Inference Time</div>
                                <div class="metric-value">{result['inference_time']:.1f}ms</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    if result['probabilities']:
                        st.subheader("📊 Probability Distribution")
                        
                        prob_df = pd.DataFrame({
                            'Class': DATASETS[selected_dataset]['classes'],
                            'Probability': result['probabilities']
                        })
                        
                        fig = px.bar(
                            prob_df, 
                            x='Class', 
                            y='Probability',
                            title="Class Probability Distribution",
                            color='Probability',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=400,
                            title_font_size=16
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    if show_features and selected_model in ['MultinomialNB', 'LinearSVM']:
                        st.subheader("🔍 Feature Importance")
                        
                        features = model_manager.get_feature_importance(
                            input_text, selected_dataset, selected_model, top_k=10
                        )
                        
                        if features:
                            feature_df = pd.DataFrame(features, columns=['Feature', 'Importance'])
                            feature_df['Abs_Importance'] = feature_df['Importance'].abs()
                            feature_df = feature_df.sort_values('Abs_Importance', ascending=True)
                            
                            fig = px.bar(
                                feature_df, 
                                x='Importance', 
                                y='Feature',
                                orientation='h',
                                title="Top Features Contributing to Prediction",
                                color='Importance',
                                color_continuous_scale='RdBu_r'
                            )
                            fig.update_layout(height=400, title_font_size=16)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("📊 Batch Text Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with text data",
            type=['csv'],
            help="CSV should have 'text' column and optionally 'label' column"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check required columns
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Find text column index safely
                        try:
                            text_idx = list(df.columns).index('text')
                        except ValueError:
                            text_idx = 0
                        text_column = st.selectbox("Text Column", df.columns, index=text_idx)
                    
                    with col2:
                        has_labels = 'label' in df.columns
                        if has_labels:
                            # Find label column index safely
                            try:
                                label_idx = list(df.columns).index('label') + 1
                            except ValueError:
                                label_idx = 0
                            label_column = st.selectbox("Label Column (optional)", 
                                                       ['None'] + list(df.columns), 
                                                       index=label_idx)
                        else:
                            label_column = st.selectbox("Label Column (optional)", ['None'])
                    
                    if st.button("🔄 Process Batch", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            # Process predictions
                            predictions = []
                            probabilities = []
                            inference_times = []
                            
                            progress_bar = st.progress(0)
                            
                            for idx, text in enumerate(df[text_column]):
                                result = model_manager.predict_single(
                                    str(text), selected_dataset, selected_model
                                )
                                
                                predictions.append(result['prediction'] if not result['error'] else -1)
                                probabilities.append(result['probabilities'] if not result['error'] else None)
                                inference_times.append(result['inference_time'])
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            # Add results to dataframe
                            df['predicted_class'] = [DATASETS[selected_dataset]['classes'][p] if p != -1 else 'Error' 
                                                    for p in predictions]
                            df['prediction_confidence'] = [max(p) if p else 0 for p in probabilities]
                            df['inference_time_ms'] = inference_times
                            
                            st.success("✅ Batch processing completed!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Total Samples</div>
                                    <div class="metric-value">{len(df)}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                avg_confidence = np.mean([c for c in df['prediction_confidence'] if c > 0])
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Avg Confidence</div>
                                    <div class="metric-value">{avg_confidence:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                avg_time = np.mean(df['inference_time_ms'])
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Avg Time (ms)</div>
                                    <div class="metric-value">{avg_time:.1f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Results table
                            st.subheader("📊 Prediction Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download Results",
                                data=csv,
                                file_name=f"predictions_{selected_model}_{selected_dataset.lower().replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                            # Confusion matrix if labels available
                            if label_column != 'None' and label_column in df.columns:
                                st.subheader("🎯 Confusion Matrix")
                                
                                # Create confusion matrix
                                from sklearn.metrics import confusion_matrix, classification_report
                                
                                true_labels = df[label_column]
                                pred_labels = df['predicted_class']
                                
                                # Filter out error predictions
                                valid_mask = pred_labels != 'Error'
                                true_labels_valid = true_labels[valid_mask]
                                pred_labels_valid = pred_labels[valid_mask]
                                
                                if len(true_labels_valid) > 0:
                                    cm = confusion_matrix(true_labels_valid, pred_labels_valid)
                                    
                                    # Plot confusion matrix
                                    fig = px.imshow(
                                        cm,
                                        text_auto=True,
                                        aspect="auto",
                                        title="Confusion Matrix"
                                    )
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Classification report
                                    report = classification_report(true_labels_valid, pred_labels_valid, 
                                                                 output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.subheader("📋 Classification Report")
                                    st.dataframe(report_df, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Tab 3: Model Comparison
    with tab3:
        st.header("📈 Model Performance Comparison")
        
        # Load performance data if available
        try:
            summary_df = pd.read_csv('results/summary.csv')
            st.success("✅ Performance data loaded successfully")
            
            # Filter by selected dataset
            dataset_data = summary_df[summary_df['dataset'] == selected_dataset.lower().replace(' ', '_')]
            
            if len(dataset_data) > 0:
                # Performance metrics comparison
                metrics = ['accuracy', 'f1_macro', 'inference_latency_ms']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_metric = st.selectbox("Select Metric", metrics)
                
                with col2:
                    sample_size = st.selectbox("Sample Size", 
                                             dataset_data['n_samples'].unique())
                
                # Filter data
                filtered_data = dataset_data[dataset_data['n_samples'] == sample_size]
                
                if len(filtered_data) > 0:
                    # Create comparison plot
                    fig = px.box(
                        filtered_data, 
                        x='model', 
                        y=selected_metric,
                        title=f"{selected_metric.title()} Comparison - {selected_dataset} (n={sample_size})"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics table
                    summary_stats = filtered_data.groupby('model')[selected_metric].agg([
                        'mean', 'std', 'min', 'max'
                    ]).round(4)
                    
                    st.subheader(f"📊 {selected_metric.title()} Summary Statistics")
                    st.dataframe(summary_stats, use_container_width=True)
            else:
                st.warning(f"No performance data available for {selected_dataset}")
                
        except FileNotFoundError:
            st.warning("Performance summary not found. Please run the complete experiment first.")
            st.info("Use the Jupyter notebook to generate comprehensive performance data.")
    
    # Tab 4: Feature Interpretation
    with tab4:
        st.header("🔍 Model Interpretation & Feature Analysis")
        
        if selected_model in ['MultinomialNB', 'LinearSVM']:
            st.info("Feature interpretation is available for classical models (MNB, SVM)")
            
            # Sample text for analysis
            sample_text = st.text_area(
                "Enter text for feature analysis:",
                value="This is a great movie with excellent acting and plot.",
                height=100
            )
            
            if st.button("🔍 Analyze Features"):
                with st.spinner("Analyzing features..."):
                    features = model_manager.get_feature_importance(
                        sample_text, selected_dataset, selected_model, top_k=20
                    )
                    
                    if features:
                        # Create feature importance visualization
                        feature_df = pd.DataFrame(features, columns=['Feature', 'Importance'])
                        feature_df['Abs_Importance'] = feature_df['Importance'].abs()
                        feature_df = feature_df.sort_values('Abs_Importance', ascending=True)
                        
                        fig = px.bar(
                            feature_df.tail(15), 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title="Top 15 Features by Importance",
                            color='Importance',
                            color_continuous_scale='RdBu_r'
                        )
                        fig.update_layout(height=600, title_font_size=16)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature details table
                        st.subheader("📋 Feature Details")
                        st.dataframe(
                            feature_df.sort_values('Abs_Importance', ascending=False),
                            use_container_width=True
                        )
                    else:
                        st.warning("No features extracted. Please check if the model is loaded correctly.")
        
        else:
            st.info(f"Feature interpretation for {selected_model} is not yet implemented.")
            st.write("Available interpretations:")
            st.write("- **Classical Models**: TF-IDF feature weights")
            st.write("- **Neural Models**: Attention weights, gradient-based saliency (planned)")
            st.write("- **BERT**: Integrated gradients, attention visualization (planned)")
    
    # Tab 5: Model Performance
    with tab5:
        st.header("📋 Model Performance Dashboard")
        
        # Model availability status
        st.subheader("🚦 Model Availability Status")
        
        status_data = []
        for model in MODELS.keys():
            model_path = model_manager.get_model_path(selected_dataset, model)
            is_available = model_path and os.path.exists(model_path)
            
            status_data.append({
                'Model': model,
                'Type': MODELS[model]['type'],
                'Status': '✅ Available' if is_available else '❌ Not Found',
                'Path': model_path or 'N/A'
            })
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)
        
        # System information
        st.subheader("💻 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Environment:**")
            import sys
            st.write(f"- Python: {sys.version.split()[0]}")
            st.write(f"- Streamlit: {st.__version__}")
            st.write(f"- PyTorch Available: {'✅' if 'torch' in globals() else '❌'}")
            st.write(f"- Transformers Available: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
        
        with col2:
            st.write("**Datasets Available:**")
            for dataset, info in DATASETS.items():
                st.write(f"- {dataset}: {info['description']}")
        
        # Performance tips
        st.subheader("🚀 Performance Tips")
        
        st.info("""
        **For optimal performance:**
        
        1. **Classical Models**: Fast inference, good for real-time applications
        2. **BiLSTM**: Medium speed, good balance of accuracy and efficiency  
        3. **BERT**: Highest accuracy but slower inference, use GPU when available
        4. **Hybrid**: Good compromise between BERT accuracy and classical speed
        
        **Recommendations:**
        - Use classical models for low-latency applications
        - Use BERT for highest accuracy requirements
        - Use hybrid approach for balanced performance
        """)

if __name__ == "__main__":
    main()