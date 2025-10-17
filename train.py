#!/usr/bin/env python3
"""
NLP Comparative Analysis Toolkit (NLP-CAT) 2.1 - CLI Training Wrapper

A command-line interface for training individual model configurations across different
datasets and sample sizes with comprehensive logging and reproducibility controls.

Author: Daniel Wanjala Machimbo
Institution: The Cooperative University of Kenya
Date: October 2025

Usage:
    python train.py --dataset ag_news --model bert --n_samples 1000 --seed 42
    python train.py --dataset imdb --model svm --n_samples full --seed 101 --gpu
    python train.py --config experiments/config.json --verbose
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Setup logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for training runs"""
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup logger
    logger = logging.getLogger('NLP_CAT_Training')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ExperimentConfig:
    """Configuration class for training experiments"""
    
    def __init__(self, **kwargs):
        # Dataset configuration
        self.dataset = kwargs.get('dataset', 'ag_news')
        self.n_samples = kwargs.get('n_samples', 1000)
        
        # Model configuration
        self.model = kwargs.get('model', 'svm')
        
        # Training configuration
        self.seed = kwargs.get('seed', 42)
        self.test_size = kwargs.get('test_size', 0.2)
        self.cv_folds = kwargs.get('cv_folds', 3)
        
        # Hardware configuration
        self.use_gpu = kwargs.get('use_gpu', False)
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Output configuration
        self.output_dir = kwargs.get('output_dir', 'artifacts')
        self.save_model = kwargs.get('save_model', True)
        self.verbose = kwargs.get('verbose', False)
        
        # Hyperparameter configuration
        self.hyperparams = kwargs.get('hyperparams', {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'dataset': self.dataset,
            'n_samples': self.n_samples,
            'model': self.model,
            'seed': self.seed,
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'use_gpu': self.use_gpu,
            'device': self.device,
            'output_dir': self.output_dir,
            'save_model': self.save_model,
            'verbose': self.verbose,
            'hyperparams': self.hyperparams
        }
    
    @classmethod
    def from_json(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class DatasetLoader:
    """Centralized dataset loading functionality"""
    
    @staticmethod
    def set_random_seeds(seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def load_dataset(dataset_name: str, logger: logging.Logger) -> Dict[str, Any]:
        """Load specified dataset with error handling"""
        
        logger.info(f"Loading {dataset_name} dataset...")
        
        try:
            if dataset_name.lower() in ['ag_news', 'ag-news', 'agnews']:
                return DatasetLoader._load_ag_news(logger)
            elif dataset_name.lower() in ['20_newsgroups', '20newsgroups', 'newsgroups']:
                return DatasetLoader._load_20newsgroups(logger)
            elif dataset_name.lower() in ['imdb', 'imdb_reviews']:
                return DatasetLoader._load_imdb(logger)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    @staticmethod
    def _load_ag_news(logger: logging.Logger) -> Dict[str, Any]:
        """Load AG News dataset"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("ag_news", cache_dir="data/cache")
            
            train_texts = [item['text'] for item in dataset['train']]
            train_labels = [item['label'] for item in dataset['train']]
            test_texts = [item['text'] for item in dataset['test']]
            test_labels = [item['label'] for item in dataset['test']]
            
            class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            
            logger.info(f"AG News loaded: {len(train_texts)} train, {len(test_texts)} test")
            
            return {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts, 
                'test_labels': test_labels,
                'class_names': class_names,
                'n_classes': 4
            }
            
        except ImportError:
            logger.error("datasets library not available for AG News")
            raise
    
    @staticmethod
    def _load_20newsgroups(logger: logging.Logger) -> Dict[str, Any]:
        """Load 20 Newsgroups dataset"""
        try:
            from sklearn.datasets import fetch_20newsgroups
            
            train_data = fetch_20newsgroups(
                subset='train',
                remove=('headers', 'footers', 'quotes'),
                shuffle=True,
                random_state=42,
                data_home='data'
            )
            
            test_data = fetch_20newsgroups(
                subset='test',
                remove=('headers', 'footers', 'quotes'),
                shuffle=True,
                random_state=42,
                data_home='data'
            )
            
            logger.info(f"20 Newsgroups loaded: {len(train_data.data)} train, {len(test_data.data)} test")
            
            return {
                'train_texts': train_data.data,
                'train_labels': train_data.target.tolist(),
                'test_texts': test_data.data,
                'test_labels': test_data.target.tolist(),
                'class_names': train_data.target_names,
                'n_classes': 20
            }
            
        except ImportError:
            logger.error("sklearn not available for 20 Newsgroups")
            raise
    
    @staticmethod
    def _load_imdb(logger: logging.Logger) -> Dict[str, Any]:
        """Load IMDb dataset"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("imdb", cache_dir="data/cache")
            
            train_texts = [item['text'] for item in dataset['train']]
            train_labels = [item['label'] for item in dataset['train']]
            test_texts = [item['text'] for item in dataset['test']]
            test_labels = [item['label'] for item in dataset['test']]
            
            class_names = ['Negative', 'Positive']
            
            logger.info(f"IMDb loaded: {len(train_texts)} train, {len(test_texts)} test")
            
            return {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts,
                'test_labels': test_labels,
                'class_names': class_names,
                'n_classes': 2
            }
            
        except ImportError:
            logger.error("datasets library not available for IMDb")
            raise

class ModelTrainer:
    """Unified model training interface"""
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.training_history = {}
    
    def train(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Train model based on configuration"""
        
        self.logger.info(f"Starting training: {self.config.model} on {self.config.dataset}")
        
        # Set random seeds
        DatasetLoader.set_random_seeds(self.config.seed)
        
        # Prepare data
        X_train, y_train = dataset['train_texts'], dataset['train_labels']
        X_test, y_test = dataset['test_texts'], dataset['test_labels']
        
        # Sample data if requested
        if self.config.n_samples != 'full' and self.config.n_samples < len(X_train):
            indices = np.random.choice(len(X_train), self.config.n_samples, replace=False)
            X_train = [X_train[i] for i in indices]
            y_train = [y_train[i] for i in indices]
            self.logger.info(f"Sampled {self.config.n_samples} training examples")
        
        # Train model based on type
        if self.config.model.lower() in ['svm', 'linearsvm', 'linear_svm']:
            return self._train_svm(X_train, y_train, X_test, y_test, dataset['n_classes'])
        elif self.config.model.lower() in ['mnb', 'nb', 'multinomialnb']:
            return self._train_nb(X_train, y_train, X_test, y_test, dataset['n_classes'])
        elif self.config.model.lower() in ['bilstm', 'lstm']:
            return self._train_bilstm(X_train, y_train, X_test, y_test, dataset['n_classes'])
        elif self.config.model.lower() in ['bert', 'transformer']:
            return self._train_bert(X_train, y_train, X_test, y_test, dataset['n_classes'])
        elif self.config.model.lower() in ['hybrid']:
            return self._train_hybrid(X_train, y_train, X_test, y_test, dataset['n_classes'])
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")
    
    def _train_svm(self, X_train: List[str], y_train: List[int], 
                   X_test: List[str], y_test: List[int], n_classes: int) -> Dict[str, Any]:
        """Train Linear SVM with TF-IDF features"""
        
        self.logger.info("Training Linear SVM...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 1)
            )),
            ('svm', LinearSVC(
                class_weight='balanced',
                random_state=self.config.seed,
                max_iter=1000
            ))
        ])
        
        # Hyperparameter tuning if not disabled
        if self.config.hyperparams.get('tune', True):
            self.logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'tfidf__max_features': [5000, 10000],
                'svm__C': [0.1, 1.0, 10.0]
            }
            
            grid_search = GridSearchCV(
                pipeline, param_grid, 
                cv=self.config.cv_folds,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1 if self.config.verbose else 0
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.model = grid_search.best_estimator_
            self.logger.info(f"Best params: {grid_search.best_params_}")
            
        else:
            start_time = time.time()
            self.model = pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
        
        # Evaluation
        return self._evaluate_model(X_test, y_test, training_time)
    
    def _train_nb(self, X_train: List[str], y_train: List[int],
                  X_test: List[str], y_test: List[int], n_classes: int) -> Dict[str, Any]:
        """Train Multinomial Naive Bayes with TF-IDF features"""
        
        self.logger.info("Training Multinomial Naive Bayes...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=10000,
                ngram_range=(1, 1)
            )),
            ('nb', MultinomialNB())
        ])
        
        # Hyperparameter tuning if not disabled
        if self.config.hyperparams.get('tune', True):
            self.logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'tfidf__ngram_range': [(1, 1), (1, 2)], 
                'tfidf__max_features': [5000, 10000],
                'nb__alpha': [0.1, 1.0, 10.0]
            }
            
            grid_search = GridSearchCV(
                pipeline, param_grid,
                cv=self.config.cv_folds,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1 if self.config.verbose else 0
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.model = grid_search.best_estimator_
            self.logger.info(f"Best params: {grid_search.best_params_}")
            
        else:
            start_time = time.time()
            self.model = pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
        
        # Evaluation
        return self._evaluate_model(X_test, y_test, training_time)
    
    def _train_bilstm(self, X_train: List[str], y_train: List[int],
                      X_test: List[str], y_test: List[int], n_classes: int) -> Dict[str, Any]:
        """Train BiLSTM model (placeholder implementation)"""
        
        self.logger.warning("BiLSTM training not fully implemented in CLI wrapper")
        self.logger.info("Please use the Jupyter notebook for complete BiLSTM training")
        
        # Placeholder results
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'training_time': 0.0,
            'inference_time': 0.0,
            'model_size': 0.0,
            'error': 'BiLSTM training not implemented in CLI'
        }
    
    def _train_bert(self, X_train: List[str], y_train: List[int],
                    X_test: List[str], y_test: List[int], n_classes: int) -> Dict[str, Any]:
        """Train BERT model (placeholder implementation)"""
        
        self.logger.warning("BERT training not fully implemented in CLI wrapper")
        self.logger.info("Please use the Jupyter notebook for complete BERT training")
        
        # Placeholder results
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'training_time': 0.0,
            'inference_time': 0.0,
            'model_size': 0.0,
            'error': 'BERT training not implemented in CLI'
        }
    
    def _train_hybrid(self, X_train: List[str], y_train: List[int],
                      X_test: List[str], y_test: List[int], n_classes: int) -> Dict[str, Any]:
        """Train hybrid model (placeholder implementation)"""
        
        self.logger.warning("Hybrid training not fully implemented in CLI wrapper")
        self.logger.info("Please use the Jupyter notebook for complete hybrid training")
        
        # Placeholder results
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'training_time': 0.0,
            'inference_time': 0.0,
            'model_size': 0.0,
            'error': 'Hybrid training not implemented in CLI'
        }
    
    def _evaluate_model(self, X_test: List[str], y_test: List[int], 
                       training_time: float) -> Dict[str, Any]:
        """Evaluate trained model and return comprehensive metrics"""
        
        self.logger.info("Evaluating model...")
        
        # Predictions and timing
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Model size estimation
        model_size = 0.0
        if hasattr(self.model, 'named_steps'):
            try:
                temp_path = f"temp_model_{self.config.seed}.joblib"
                joblib.dump(self.model, temp_path)
                model_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
                os.remove(temp_path)
            except:
                pass
        
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}")
        self.logger.info(f"Training time: {training_time:.2f}s, Inference time: {inference_time:.2f}s")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'training_time': training_time,
            'inference_time': inference_time,
            'model_size': model_size,
            'n_test_samples': len(X_test)
        }
    
    def save_model(self, output_path: str) -> None:
        """Save trained model to specified path"""
        
        if self.model is None:
            self.logger.warning("No model to save")
            return
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if self.config.model.lower() in ['svm', 'mnb', 'linearsvm', 'multinomialnb']:
                joblib.dump(self.model, output_path)
            else:
                # For PyTorch models, would use torch.save
                self.logger.warning(f"Save not implemented for {self.config.model}")
                return
            
            self.logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="NLP-CAT 2.1 CLI Training Wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ag_news',
                       choices=['ag_news', '20_newsgroups', 'imdb'],
                       help='Dataset to use for training')
    
    parser.add_argument('--n_samples', type=str, default='1000',
                       help='Number of training samples (use "full" for complete dataset)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='svm',
                       choices=['svm', 'mnb', 'bilstm', 'bert', 'hybrid'],
                       help='Model type to train')
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    
    parser.add_argument('--cv_folds', type=int, default=3,
                       help='Number of cross-validation folds')
    
    # Hardware arguments
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='artifacts',
                       help='Output directory for models and results')
    
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save trained model')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, 
                       help='Path to JSON configuration file')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    parser.add_argument('--log_file', type=str,
                       help='Log file path')
    
    # Hyperparameter arguments
    parser.add_argument('--no_tune', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    logger.info("NLP-CAT 2.1 CLI Training Started")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        if args.config:
            config = ExperimentConfig.from_json(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            # Convert n_samples to int if not 'full'
            n_samples = args.n_samples if args.n_samples == 'full' else int(args.n_samples)
            
            config = ExperimentConfig(
                dataset=args.dataset,
                n_samples=n_samples,
                model=args.model,
                seed=args.seed,
                test_size=args.test_size,
                cv_folds=args.cv_folds,
                use_gpu=args.gpu,
                output_dir=args.output_dir,
                save_model=not args.no_save,
                verbose=args.verbose,
                hyperparams={'tune': not args.no_tune}
            )
        
        # Load dataset
        dataset = DatasetLoader.load_dataset(config.dataset, logger)
        
        # Initialize trainer
        trainer = ModelTrainer(config, logger)
        
        # Train model
        results = trainer.train(dataset)
        
        # Save model if requested
        if config.save_model and 'error' not in results:
            model_filename = f"{config.model}_{config.dataset}_n{config.n_samples}_seed{config.seed}.joblib"
            model_path = os.path.join(config.output_dir, 'cli_models', model_filename)
            trainer.save_model(model_path)
        
        # Save results
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'results': results
        }
        
        results_filename = f"results_{config.model}_{config.dataset}_n{config.n_samples}_seed{config.seed}.json"
        results_path = os.path.join(config.output_dir, 'cli_results', results_filename)
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY") 
        print("="*60)
        print(f"Dataset: {config.dataset}")
        print(f"Model: {config.model}")
        print(f"Samples: {config.n_samples}")
        print(f"Seed: {config.seed}")
        
        if 'error' not in results:
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1-macro: {results['f1_macro']:.4f}")
            print(f"Training time: {results['training_time']:.2f}s")
            print(f"Inference time: {results['inference_time']:.4f}s")
            if results['model_size'] > 0:
                print(f"Model size: {results['model_size']:.2f} MB")
        else:
            print(f"Error: {results['error']}")
        
        print("="*60)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()