"""
Phase 2 - Step 3: GNN Training Pipeline

Trains the UASTClassifier on domain classification task.
Includes training loop, evaluation, and ablation studies.
"""

import json
import os
import sys
import time
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# PyTorch Geometric imports
from torch_geometric.data import Data, DataLoader

from experiments.phase2.subtree_extractor import UASTSubtreeExtractor, UAST_NODE_TYPES
from experiments.phase2.gnn_encoder import UASTClassifier


def prepare_labels(dataset: list, library_to_concept: dict) -> list:
    """
    Assign domain labels to each file based on most frequent library's domain.
    """
    labels = []
    for item in dataset:
        lib_counts = Counter(item["libraries"])
        label = "Unknown"
        for lib, _ in lib_counts.most_common():
            if lib in library_to_concept:
                domain = library_to_concept[lib].get("domain")
                if domain and domain not in ["None", None]:
                    label = domain
                    break
        labels.append(label)
    return labels


def graphs_to_pyg(graph_dicts: list, labels: list, label_encoder: LabelEncoder) -> list:
    """Convert graph data dicts to PyTorch Geometric Data objects."""
    pyg_data = []
    encoded_labels = label_encoder.transform(labels)

    for graph, label_idx in zip(graph_dicts, encoded_labels):
        if graph is None:
            continue

        x = torch.tensor(graph["x"], dtype=torch.float32)
        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        y = torch.tensor(label_idx, dtype=torch.long)

        # Handle empty edge case
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data.append(data)

    return pyg_data


class GNNTrainer:
    """Training pipeline for UAST GNN classifier with mixed-precision support."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        self.history = {"train_loss": [], "val_acc": [], "val_loss": []}
        # Mixed-precision for GPU (T4 Tensor Cores support FP16)
        self.use_amp = device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        total_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch.y.size(0)
            total_samples += batch.y.size(0)

        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        total_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            total_loss += loss.item() * batch.y.size(0)
            total_samples += batch.y.size(0)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        top1 = accuracy_score(all_labels, all_preds)

        # Top-5 accuracy
        k = min(5, all_probs.shape[1])
        top5_preds = np.argsort(all_probs, axis=1)[:, -k:]
        top5 = np.mean([l in t for l, t in zip(all_labels, top5_preds)])

        # MRR
        mrr_scores = []
        for i in range(len(all_labels)):
            ranked = np.argsort(all_probs[i])[::-1]
            rank = np.where(ranked == all_labels[i])[0][0] + 1
            mrr_scores.append(1.0 / rank)
        mrr = np.mean(mrr_scores)

        return {
            "top1": top1,
            "top5": top5,
            "mrr": mrr,
            "loss": total_loss / total_samples,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    @torch.no_grad()
    def get_embeddings(self, loader: DataLoader) -> tuple:
        """Extract graph embeddings for all data."""
        self.model.eval()
        all_embs = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                embs = self.model.get_embeddings(batch.x, batch.edge_index, batch.batch)
            all_embs.append(embs.float().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        return np.concatenate(all_embs), np.concatenate(all_labels)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        patience: int = 10,
        save_path: str = None,
    ) -> dict:
        """Full training loop with early stopping."""
        best_val_acc = 0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_acc"].append(val_metrics["top1"])
            self.history["val_loss"].append(val_metrics["loss"])

            if val_metrics["top1"] > best_val_acc:
                best_val_acc = val_metrics["top1"]
                epochs_without_improvement = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_metrics['top1']:.2%} | "
                    f"Val Top5: {val_metrics['top5']:.2%} | "
                    f"Val MRR: {val_metrics['mrr']:.4f} | "
                    f"Best: {best_val_acc:.2%}"
                )

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, weights_only=True))

        return {
            "best_val_acc": best_val_acc,
            "history": self.history,
            "epochs_trained": epoch + 1,
        }


def run_phase2_experiment(
    dataset: list,
    library_to_concept: dict,
    uast_dir: str,
    results_dir: str,
    device: str = "cuda",
    use_subtree: bool = True,
    k_hop: int = 3,
    num_epochs: int = 50,
    batch_size: int = 32,
    hidden_dim: int = 256,
    embedding_dim: int = 768,
    lr: float = 1e-3,
):
    """
    Run complete Phase 2 GNN experiment.

    Args:
        dataset: Output from UASTDataLoader.load_dataset()
        library_to_concept: Library -> concept mapping
        uast_dir: Path to Conversion/UAST_files/
        results_dir: Where to save results
        device: 'cuda' or 'cpu'
    """
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE 2: UAST GNN Domain Classification")
    print("=" * 80)

    # Step 1: Extract subtrees
    print("\n--- Step 1: UAST Subtree Extraction ---")
    extractor = UASTSubtreeExtractor(k_hop=k_hop)
    cache_path = os.path.join(results_dir, f"graph_cache_k{k_hop}_sub{use_subtree}.pkl")
    graphs = extractor.process_dataset(
        dataset, uast_dir, use_subtree=use_subtree, cache_path=cache_path
    )

    # Step 2: Prepare labels
    print("\n--- Step 2: Preparing Labels ---")
    labels = prepare_labels(dataset, library_to_concept)

    # Filter out None graphs and align labels
    valid_indices = []
    valid_graphs = []
    valid_labels = []

    graph_idx = 0
    for i, item in enumerate(dataset):
        file_id = item["file_id"]
        uast_path = os.path.join(uast_dir, f"UAST_{file_id}.json")
        if not os.path.exists(uast_path):
            continue
        if graph_idx < len(graphs) and graphs[graph_idx] is not None:
            if graphs[graph_idx]["file_id"] == file_id:
                valid_graphs.append(graphs[graph_idx])
                valid_labels.append(labels[i])
                valid_indices.append(i)
        graph_idx += 1

    # If alignment is tricky, use simpler approach
    if len(valid_graphs) != len(graphs):
        # Graphs already have file_id, match them
        file_to_label = {}
        for i, item in enumerate(dataset):
            file_to_label[item["file_id"]] = labels[i]

        valid_graphs = []
        valid_labels = []
        for g in graphs:
            fid = g.get("file_id")
            if fid and fid in file_to_label:
                valid_graphs.append(g)
                valid_labels.append(file_to_label[fid])

    print(f"Valid graphs: {len(valid_graphs)}")
    print(f"Label distribution: {Counter(valid_labels).most_common(10)}")

    # Filter out "Unknown" labels
    filtered_graphs = []
    filtered_labels = []
    for g, l in zip(valid_graphs, valid_labels):
        if l != "Unknown":
            filtered_graphs.append(g)
            filtered_labels.append(l)

    print(f"After filtering Unknown: {len(filtered_graphs)} graphs")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(filtered_labels)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Convert to PyG data
    print("\n--- Step 3: Converting to PyG Data ---")
    pyg_data = graphs_to_pyg(filtered_graphs, filtered_labels, label_encoder)
    print(f"PyG datasets: {len(pyg_data)}")

    # Split data
    indices = list(range(len(pyg_data)))
    encoded_labels = label_encoder.transform(filtered_labels)

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=42,
        stratify=encoded_labels[train_idx],
    )

    train_data = [pyg_data[i] for i in train_idx]
    val_data = [pyg_data[i] for i in val_idx]
    test_data = [pyg_data[i] for i in test_idx]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create data loaders (pin_memory + num_workers for GPU throughput)
    loader_kwargs = {"pin_memory": True, "num_workers": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **loader_kwargs)

    # Step 4: Train GNN
    print("\n--- Step 4: Training GNN ---")
    node_feature_dim = len(UAST_NODE_TYPES)

    model = UASTClassifier(
        num_classes=num_classes,
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=0.1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = GNNTrainer(model, device=device, lr=lr)
    save_path = os.path.join(results_dir, "best_gnn_model.pt")

    start_time = time.time()
    train_results = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        patience=10,
        save_path=save_path,
    )
    train_time = time.time() - start_time

    # Step 5: Evaluate on test set
    print("\n--- Step 5: Test Evaluation ---")
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Top-1: {test_metrics['top1']:.2%}")
    print(f"Test Top-5: {test_metrics['top5']:.2%}")
    print(f"Test MRR:   {test_metrics['mrr']:.4f}")

    # Classification report
    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    print("\nClassification Report:")
    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=label_encoder.classes_,
        )
    )

    # Extract embeddings for Phase 3
    print("\n--- Step 6: Extracting Embeddings ---")
    all_loader = DataLoader(pyg_data, batch_size=batch_size)
    embeddings, emb_labels = trainer.get_embeddings(all_loader)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save results
    results = {
        "phase": "phase2",
        "config": {
            "k_hop": k_hop,
            "use_subtree": use_subtree,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        "dataset": {
            "total_graphs": len(pyg_data),
            "num_classes": num_classes,
            "classes": list(label_encoder.classes_),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        },
        "training": {
            "train_time_seconds": train_time,
            "epochs_trained": train_results["epochs_trained"],
            "best_val_acc": train_results["best_val_acc"],
        },
        "test_results": {
            "top1": float(test_metrics["top1"]),
            "top5": float(test_metrics["top5"]),
            "mrr": float(test_metrics["mrr"]),
        },
        "per_class_report": report,
    }

    results_path = os.path.join(results_dir, "phase2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save embeddings for Phase 3
    emb_path = os.path.join(results_dir, "gnn_embeddings.npz")
    np.savez(emb_path, embeddings=embeddings, labels=emb_labels)
    print(f"Embeddings saved to {emb_path}")

    # Save label encoder
    le_path = os.path.join(results_dir, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)

    return results, embeddings, label_encoder
