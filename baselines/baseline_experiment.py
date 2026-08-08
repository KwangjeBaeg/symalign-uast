"""
Baseline Experiment: TF-IDF + Logistic Regression

This serves as the baseline for comparison with the UAST-LLM alignment model.

Experiment Design:
1. Represent each code file as a "bag of libraries" (similar to bag of words)
2. Use TF-IDF to weight library importance
3. Train a classifier to predict:
   a) Library domain (e.g., "Scientific Computing")
   b) Library functionality (e.g., "Linear Algebra")
   c) Next library to use (library recommendation)

This baseline answers: "Can we predict semantic concepts without
understanding code structure, using only library co-occurrence patterns?"
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import UASTDataLoader
from experiments.config import *


class BaselineExperiment:
    """
    Baseline experiment using traditional ML methods.
    """

    def __init__(self,
                 data_loader: UASTDataLoader,
                 task: str = "domain",
                 method: str = "tfidf"):
        """
        Args:
            data_loader: Initialized UASTDataLoader
            task: One of ["domain", "functionality", "library_recommendation"]
            method: One of ["tfidf", "count", "cooccurrence"]
        """
        self.data_loader = data_loader
        self.task = task
        self.method = method

        self.vectorizer = None
        self.classifier = None
        self.label_encoder = LabelEncoder()

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.results = {}

    def prepare_features(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert dataset to feature matrix and labels.

        Args:
            dataset: Output from UASTDataLoader.load_dataset()

        Returns:
            X: Feature matrix (sparse or dense)
            y: Labels (encoded)
            label_names: Original label names
        """
        print(f"\nPreparing features for task: {self.task}")

        # Create library "documents" - each file is a document of libraries
        documents = []
        labels = []
        label_names = []

        for item in tqdm(dataset, desc="Creating documents"):
            # Join libraries into a space-separated string
            lib_doc = " ".join(item["libraries"])
            documents.append(lib_doc)

            # Get label based on task
            label = self._get_label_for_item(item)
            if label is not None:
                labels.append(label)
                label_names.append(label)
            else:
                # If no label, use "Unknown"
                labels.append("Unknown")
                label_names.append("Unknown")

        # Vectorize documents
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=BASELINE_CONFIG["max_features"],
                ngram_range=BASELINE_CONFIG["ngram_range"],
                min_df=BASELINE_CONFIG["min_df"],
                max_df=BASELINE_CONFIG["max_df"],
                token_pattern=r'\S+',  # Don't split library names
            )
        elif self.method == "count":
            self.vectorizer = CountVectorizer(
                max_features=BASELINE_CONFIG["max_features"],
                ngram_range=BASELINE_CONFIG["ngram_range"],
                min_df=BASELINE_CONFIG["min_df"],
                max_df=BASELINE_CONFIG["max_df"],
                token_pattern=r'\S+',
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        X = self.vectorizer.fit_transform(documents)

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        print(f"Label distribution:")
        label_counts = Counter(labels)
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {label}: {count}")

        return X, y, label_names

    def _get_label_for_item(self, item: Dict) -> str:
        """
        Extract label for an item based on the task.

        For domain/functionality: Use the most common library's concept
        For library_recommendation: Use the most frequent library as target
        """
        if self.task in ["domain", "functionality"]:
            # Find the most frequent library in this file that has a concept
            lib_counts = Counter(item["libraries"])

            for lib, _ in lib_counts.most_common():
                if lib in self.data_loader.library_to_concept:
                    concept = self.data_loader.library_to_concept[lib]

                    if self.task == "domain":
                        domain = concept.get("domain")
                        if domain and domain != "None":
                            return domain

                    elif self.task == "functionality":
                        func = concept.get("functionality")
                        if func and func != "None":
                            return func

            return None

        elif self.task == "library_recommendation":
            # Predict the most frequent library
            lib_counts = Counter(item["libraries"])
            if lib_counts:
                most_common_lib = lib_counts.most_common(1)[0][0]
                return most_common_lib

        return None

    def create_splits(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      test_size: float = 0.15,
                      val_size: float = 0.15,
                      random_state: int = 42):
        """
        Create train/val/test splits.
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"\nDataset splits:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")

    def train(self, classifier_type: str = "logistic"):
        """
        Train the classifier.

        Args:
            classifier_type: "logistic" or "random_forest"
        """
        print(f"\nTraining {classifier_type} classifier...")

        if classifier_type == "logistic":
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        self.classifier.fit(self.X_train, self.y_train)
        print("Training completed!")

    def evaluate(self, split: str = "test") -> Dict:
        """
        Evaluate the model.

        Args:
            split: "train", "val", or "test"

        Returns:
            Dictionary of metrics
        """
        if split == "train":
            X, y = self.X_train, self.y_train
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_test, self.y_test

        print(f"\nEvaluating on {split} set...")

        # Predictions
        y_pred = self.classifier.predict(X)
        y_proba = self.classifier.predict_proba(X)

        # Metrics
        metrics = {}

        # Top-1 accuracy
        metrics["top1_accuracy"] = accuracy_score(y, y_pred)

        # Top-k accuracy
        for k in [5, 10]:
            if len(self.label_encoder.classes_) >= k:
                metrics[f"top{k}_accuracy"] = top_k_accuracy_score(
                    y, y_proba, k=k, labels=range(len(self.label_encoder.classes_))
                )

        # Mean Reciprocal Rank (MRR)
        mrr = self._compute_mrr(y, y_proba)
        metrics["mrr"] = mrr

        # Print results
        print(f"\n{split.upper()} Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        print("-" * 50)

        # Classification report
        print(f"\nClassification Report ({split}):")
        print(classification_report(
            y, y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))

        return metrics

    def _compute_mrr(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Compute Mean Reciprocal Rank.
        """
        reciprocal_ranks = []

        for i in range(len(y_true)):
            true_label = y_true[i]
            probs = y_proba[i]

            # Rank predictions by probability (descending)
            ranked_indices = np.argsort(probs)[::-1]

            # Find rank of true label (1-indexed)
            rank = np.where(ranked_indices == true_label)[0][0] + 1

            reciprocal_ranks.append(1.0 / rank)

        return np.mean(reciprocal_ranks)

    def visualize_confusion_matrix(self, split: str = "test", save_path: Path = None):
        """
        Visualize confusion matrix.
        """
        if split == "train":
            X, y = self.X_train, self.y_train
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_test, self.y_test

        y_pred = self.classifier.predict(X)

        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {split.upper()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        plt.show()

    def save_results(self, output_dir: Path):
        """
        Save experiment results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / "baseline_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "label_encoder": self.label_encoder,
                "task": self.task,
                "method": self.method,
            }, f)
        print(f"Saved model to {model_path}")

        # Save results
        results_path = output_dir / "baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved results to {results_path}")


def main():
    """
    Run the baseline experiment.
    """
    print("="*80)
    print("BASELINE EXPERIMENT: TF-IDF + Logistic Regression")
    print("="*80)

    # Initialize data loader
    loader = UASTDataLoader(
        data_dir=DATA_DIR,
        library_concepts_path=LIBRARY_CONCEPTS_PATH,
        library_usage_path=LIBRARY_USAGE_PATH,
        min_count=MIN_LIBRARY_COUNT,
        cache_dir=PROCESSED_DATA_DIR
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = loader.load_dataset(use_cache=True)

    # Run experiment for each task
    tasks = ["domain", "functionality", "library_recommendation"]

    all_results = {}

    for task in tasks:
        print("\n" + "="*80)
        print(f"TASK: {task.upper()}")
        print("="*80)

        # Initialize experiment
        experiment = BaselineExperiment(
            data_loader=loader,
            task=task,
            method="tfidf"
        )

        # Prepare features
        X, y, label_names = experiment.prepare_features(dataset)

        # Create splits
        experiment.create_splits(X, y)

        # Train
        experiment.train(classifier_type="logistic")

        # Evaluate
        train_metrics = experiment.evaluate("train")
        val_metrics = experiment.evaluate("val")
        test_metrics = experiment.evaluate("test")

        # Store results
        experiment.results = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

        all_results[task] = experiment.results

        # Save
        output_dir = RESULTS_DIR / "baseline" / task
        experiment.save_results(output_dir)

        # Visualize confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        experiment.visualize_confusion_matrix("test", cm_path)

    # Print summary
    print("\n" + "="*80)
    print("BASELINE EXPERIMENT SUMMARY")
    print("="*80)

    summary_df = pd.DataFrame({
        task: {
            "Train Top-1": results["train"]["top1_accuracy"],
            "Val Top-1": results["val"]["top1_accuracy"],
            "Test Top-1": results["test"]["top1_accuracy"],
            "Test MRR": results["test"]["mrr"],
        }
        for task, results in all_results.items()
    }).T

    print(summary_df)

    # Save summary
    summary_path = RESULTS_DIR / "baseline" / "summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")

    print("\n" + "="*80)
    print("Baseline experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()
