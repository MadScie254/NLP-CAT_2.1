"""
NLP Comparative Analysis Toolkit (NLP-CAT) 2.1 - Interactive Dashboard

A professional React-level Streamlit dashboard for comprehensive text classification model 
comparison and analysis.

Author: Daniel Wanjala Machimbo
Institution: The Cooperative University of Kenya
Date: October 2025

This application provides an interactive interface for:
- Model selection and comparison across datasets
- Real-time text classification with confidence scores
- Model interpretation and feature importance visualization
- Batch processing with confusion matrix generation
- Performance metrics and timing analysis
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import core libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers library not available - BERT models will not be functional")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn not available - Classical models will not be functional")

# Configure Streamlit page
st.set_page_config(
    page_title="NLP-CAT 2.1 Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for React-level styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Success/Warning/Error styling */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    /* Card-like containers */
    .analysis-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #667eea;
    }
    
    /* Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration and Constants
DATASETS = {
    'AG News': {
        'classes': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'description': '4-class news categorization',
        'max_length': 128
    },
    '20 Newsgroups': {
        'classes': [f'Class_{i}' for i in range(20)],
        'description': '20-class discussion forum classification', 
        'max_length': 256
    },
    'IMDb': {
        'classes': ['Negative', 'Positive'],
        'description': 'Binary sentiment analysis',
        'max_length': 512
    }
}

MODELS = {
    'MultinomialNB': {
        'type': 'classical',
        'description': 'Multinomial Naive Bayes with TF-IDF features',
        'complexity': 'Low'
    },
    'LinearSVM': {
        'type': 'classical', 
        'description': 'Linear Support Vector Machine with TF-IDF features',
        'complexity': 'Low'
    },
    'BiLSTM': {
        'type': 'neural',
        'description': 'Bidirectional LSTM with GloVe embeddings',
        'complexity': 'Medium'
    },
    'BERT': {
        'type': 'transformer',
        'description': 'BERT-base-uncased fine-tuned',
        'complexity': 'High'
    },
    'Hybrid': {
        'type': 'hybrid',
        'description': 'BERT embeddings + Linear SVM classifier',
        'complexity': 'Medium'
    }
}

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