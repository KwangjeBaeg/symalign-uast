"""
Experimental Configuration for UAST-LLM Alignment Research

This file contains all hyperparameters and paths for the experiments.
"""

import os
from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "Data_Extractor" / "extracted_data"
LIBRARY_CONCEPTS_PATH = ROOT_DIR / "extras" / "LLM interaction pipleline" / "libraries_concepts.json"
LIBRARY_USAGE_PATH = ROOT_DIR / "Data profiler" / "library_concept_usage_count.json"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Experimental Settings
# ============================================================================

# Library Selection
MIN_LIBRARY_COUNT = 50  # Minimum usage count to be included
TOP_K_LIBRARIES = 100   # Focus on top-k most frequent libraries

# Train/Test Split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Time-split for Zero-Shot (based on library release date)
# We'll manually annotate library release years for top libraries
TIME_SPLIT_YEAR = 2024  # Train on ≤2024, test on >2024

# ============================================================================
# Model Hyperparameters
# ============================================================================

# Baseline (TF-IDF + Logistic Regression)
BASELINE_CONFIG = {
    "max_features": 1000,
    "ngram_range": (1, 2),  # Unigrams and bigrams
    "min_df": 2,
    "max_df": 0.8,
}

# GNN Encoder (for Phase 2)
GNN_CONFIG = {
    "node_feature_dim": 50,  # Number of UAST node types
    "hidden_dim": 256,
    "output_dim": 768,  # Match LLM embedding size
    "num_layers": 3,
    "dropout": 0.1,
    "learning_rate": 1e-3,
}

# Contrastive Learning (for Phase 3)
CONTRASTIVE_CONFIG = {
    "temperature": 0.07,
    "batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 1e-4,
}

# ============================================================================
# Evaluation Metrics
# ============================================================================
EVAL_METRICS = [
    "top1_accuracy",
    "top5_accuracy",
    "top10_accuracy",
    "mrr",  # Mean Reciprocal Rank
    "ndcg@10",  # Normalized Discounted Cumulative Gain
]

# ============================================================================
# Information Bottleneck Experiment
# ============================================================================
IB_CONFIG = {
    "beta_values": [0.001, 0.01, 0.1, 1.0],  # Information bottleneck coefficient
    "compression_dims": [64, 128, 256, 512, 768],  # Latent dimensions to test
}

# ============================================================================
# Zero-Shot OOD Configuration
# ============================================================================
ZERO_SHOT_CONFIG = {
    "similarity_metrics": ["cosine", "euclidean", "manhattan"],
    "k_neighbors": [1, 3, 5, 10],  # For k-NN retrieval
}

# ============================================================================
# Concept Categories (from concept_map.txt)
# ============================================================================
CONCEPT_CATEGORIES = [
    "Web Development",
    "Data Processing",
    "Machine Learning",
    "Scientific Computing",
    "Systems Programming",
    "Database",
    "Testing",
    "Networking",
    "Security",
    "GUI",
    "Game Development",
    "Cloud Computing",
    "DevOps",
    "Quantum Computing",
]

print(f"Configuration loaded successfully!")
print(f"Data directory: {DATA_DIR}")
print(f"Total UAST files available: {len(list(DATA_DIR.glob('*.json')))}")
