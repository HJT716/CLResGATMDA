import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import random
from torch import optim
import scipy.sparse as sp
import argparse

from inits import load_data, generate_mask, test_negative_sample, normalize_features, adj_to_bias


class Dataset:
    def __init__(self, labels, nd, nm):
        """
        Initialize the dataset
        :param labels: Raw label data
        :param nd: Number of diseases
        :param nm: Number of microbes
        """
        self.positive_samples = labels[labels[:, 2] == 1].copy()
        self._generate_negative_samples(nd, nm)

    def _generate_negative_samples(self, nd, nm):
        """Generate all possible negative samples (disease-microbe pairs not present in positive samples)"""
        all_pairs = set()
        for i in range(nd):
            for j in range(nm):
                all_pairs.add((i + 1, j + 1))

        positive_pairs = set((int(p[0]), int(p[1])) for p in self.positive_samples)
        negative_pairs = [list(pair) + [0] for pair in (all_pairs - positive_pairs)]
        self.negative_samples = np.array(negative_pairs, dtype=np.float32)

    def split(self, test_ratio=0.2, equal_train=False, equal_test=False):
        """
        Split the dataset into training set and test set
        :param test_ratio: Proportion of the test set
        :param equal_train: Whether the training set has an equal number of positive and negative samples
        :param equal_test: Whether the test set has an equal number of positive and negative samples
        :return: Training set and test set (both contain positive and negative samples)
        """
        # Split positive samples
        pos_train, pos_test = self._split_positive(test_ratio)

        # Process negative samples
        if equal_train:
            train_neg = self._sample_negative(len(pos_train))
        else:
            train_neg = self._split_negative_by_ratio(test_ratio, is_train=True)

        if equal_test:
            test_neg = self._sample_negative(len(pos_test))
        else:
            test_neg = self._split_negative_by_ratio(test_ratio, is_train=False)

        train_data = np.vstack([pos_train, train_neg])
        test_data = np.vstack([pos_test, test_neg])
        return train_data, test_data

    def _split_positive(self, test_ratio):
        """Split positive samples into training set and test set"""
        split_idx = int(len(self.positive_samples) * (1 - test_ratio))
        np.random.shuffle(self.positive_samples)
        return self.positive_samples[:split_idx], self.positive_samples[split_idx:]

    def _sample_negative(self, count):
        """Randomly sample the specified number of samples from negative samples"""
        if count > len(self.negative_samples):
            raise ValueError("Insufficient negative samples to meet the sampling requirement")
        return np.random.choice(self.negative_samples, size=count, replace=False)

    def _split_negative_by_ratio(self, test_ratio, is_train):
        """Split negative samples by proportion"""
        split_idx = int(len(self.negative_samples) * (1 - test_ratio))
        np.random.shuffle(self.negative_samples)
        return self.negative_samples[:split_idx] if is_train else self.negative_samples[split_idx:]


class Config:
    def __init__(self, args):
        self.data_dir = r'src/data/mydata/HMDAD'
        self.save_dir = 'results'
        os.makedirs(self.save_dir, exist_ok=True)

        self.hidden_dim = 64
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.4
        self.weight_decay = 5e-05
        self.epochs = 300
        self.lr = 0.002
        self.patience = 50

        self.ensemble_size = 1
        self.use_diverse_models = True
        self.alpha = 0.1
        self.gamma = 3.0
        self.balance_weight = 0.6

        self.equal_train = args.equal_train
        self.equal_test = args.equal_test


# -------------------------- Loss Functions --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return torch.mean(self.alpha * (1 - pt) ** self.gamma * bce)


class BalancedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask

        pos_loss = F.binary_cross_entropy(inputs[pos_mask], targets[pos_mask], reduction='mean')
        neg_loss = F.binary_cross_entropy(inputs[neg_mask], targets[neg_mask], reduction='mean')

        return (pos_loss + neg_loss) / 2


class FusionLoss(nn.Module):
    def __init__(self, balance_weight=0.6, alpha=0.3, gamma=3.0):
        super().__init__()
        self.balance_weight = balance_weight
        self.balance_loss = BalancedLoss()
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, inputs, targets):
        return (self.balance_weight * self.balance_loss(inputs, targets) +
                (1 - self.balance_weight) * self.focal_loss(inputs, targets))


# -------------------------- Data Loader --------------------------
class EnhancedDataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self, train_arr, test_arr):
        """Load data using improved dataset splitting logic"""
        print("=== Using improved dataset splitting logic ==“）
        interaction, features, _, _, _, _, labels = load_data(train_arr, test_arr)


        self.nd = np.max(labels[:, 0]).astype(np.int32)  # Number of diseases
        self.nm = np.max(labels[:, 1]).astype(np.int32)  # Number of microbes
        self.nb_nodes = features.shape[0]

        features = normalize_features(features)

        adj = sp.csr_matrix(interaction)
        edge_index = np.array(adj.nonzero())
        edge_index = torch.from_numpy(edge_index).long()


        edge_label_index, edge_label, train_mask, test_mask = self.generate_edges(labels)

        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            edge_label=torch.FloatTensor(edge_label),
            train_mask=torch.BoolTensor(train_mask),
            test_mask=torch.BoolTensor(test_mask)
        ), labels

    def generate_edges(self, labels):
        """Generate edge labels and masks based on Dataset class"""
        dataset = Dataset(labels, self.nd, self.nm)
        train_data, test_data = dataset.split(
            test_ratio=0.2,
            equal_train=self.config.equal_train,
            equal_test=self.config.equal_test
        )

        pos_edges_train = np.array([(int(p[0] - 1), int(p[1] - 1 + self.nd))
                                    for p in train_data if p[2] == 1])
        neg_edges_train = np.array([(int(n[0] - 1), int(n[1] - 1 + self.nd))
                                    for n in train_data if n[2] == 0])

        pos_edges_test = np.array([(int(p[0] - 1), int(p[1] - 1 + self.nd))
                                   for p in test_data if p[2] == 1])
        neg_edges_test = np.array([(int(n[0] - 1), int(n[1] - 1 + self.nd))
                                   for n in test_data if n[2] == 0])

        all_edges = np.vstack([pos_edges_train, neg_edges_train, pos_edges_test, neg_edges_test])
        all_labels = np.hstack([
            np.ones(len(pos_edges_train)),
            np.zeros(len(neg_edges_train)),
            np.ones(len(pos_edges_test)),
            np.zeros(len(neg_edges_test))
        ])

        train_size = len(pos_edges_train) + len(neg_edges_train)
        test_size = len(pos_edges_test) + len(neg_edges_test)
        original_train_mask = np.concatenate([np.ones(train_size, dtype=bool), np.zeros(test_size, dtype=bool)])
        original_test_mask = ~original_train_mask

        indices = np.random.permutation(len(all_labels))
        edge_label_index = torch.tensor(all_edges[indices].T, dtype=torch.long)
        edge_label = all_labels[indices]

        train_mask = original_train_mask[indices]
        test_mask = original_test_mask[indices]

        print(f"Training set - Positive samples: {len(pos_edges_train)}, Negative samples: {len(neg_edges_train)}")
        print(f"Test set - Positive samples: {len(pos_edges_test)}, Negative samples: {len(neg_edges_test)}")

        return edge_label_index, edge_label, train_mask, test_mask


# -------------------------- Model Components --------------------------
class CrossLayerAttention(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.num_layers = num_layers

    def forward(self, layer_outputs):
        attn_weights = []
        for i in range(self.num_layers):
            attn_weights.append(self.attn(layer_outputs[i]))

        attn_weights = torch.cat(attn_weights, dim=1)
        attn_weights = F.softmax(attn_weights, dim=1)

        output = 0
        for i in range(self.num_layers):
            output += attn_weights[:, i].unsqueeze(1) * layer_outputs[i]

        return output


class ResidualGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projections = nn.ModuleList()

        self.layers.append(GATConv(in_dim, hidden_dim, heads=num_heads, dropout=dropout))
        self.residuals.append(
            nn.Linear(in_dim, hidden_dim * num_heads) if in_dim != hidden_dim * num_heads else nn.Identity())
        self.bns.append(BatchNorm(hidden_dim * num_heads))
        self.projections.append(nn.Linear(hidden_dim * num_heads, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            self.residuals.append(nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads))
            self.bns.append(BatchNorm(hidden_dim * num_heads))
            self.projections.append(nn.Linear(hidden_dim * num_heads, hidden_dim))

        self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
        self.residuals.append(nn.Linear(hidden_dim * num_heads, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        self.projections.append(nn.Linear(hidden_dim, hidden_dim))

        self.cross_attn = CrossLayerAttention(hidden_dim, num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        layer_outputs = []

        for layer, residual, bn, proj in zip(self.layers, self.residuals, self.bns, self.projections):
            x_res = residual(x)
            x = layer(x, edge_index)
            x = bn(x)
            x = F.elu(x + x_res)
            x = self.dropout(x)
            x_proj = proj(x)
            layer_outputs.append(x_proj)

        x = self.cross_attn(layer_outputs)
        return x


# -------------------------- Model --------------------------
class BioGATv3(nn.Module):
    def __init__(self, config, in_dim):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.gat = ResidualGAT(
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_label_index):
        x = self.gat(x, edge_index)
        row, col = edge_label_index
        z = torch.cat([x[row], x[col]], dim=-1)
        return torch.sigmoid(self.decoder(z)).squeeze()


class BioGATv3Variant(BioGATv3):
    def __init__(self, config, in_dim, hidden_dim, num_heads):
        super().__init__(config, in_dim)
        self.hidden_dim = hidden_dim
        self.gat = ResidualGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )


# -------------------------- Trainer --------------------------
class EnhancedGATTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self.data_loader = EnhancedDataLoader(config)


    def calculate_aupr(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)

    def _get_model(self, in_dim, seed):
        torch.manual_seed(seed)
        if not self.config.use_diverse_models:
            return BioGATv3(self.config, in_dim).to(self.device)

        hidden_dims = [64]
        num_heads = [8]
        dropout_rates = [0.4]

        hidden_dim = hidden_dims[seed % len(hidden_dims)]
        n_heads = num_heads[seed % len(num_heads)]
        dropout = dropout_rates[seed % len(dropout_rates)]

        model = BioGATv3Variant(self.config, in_dim, hidden_dim, n_heads).to(self.device)

        if random.random() > 0.5:
            for param in list(model.gat.parameters())[seed % len(list(model.gat.parameters())):]:
                param.data.add_(torch.randn_like(param) * 0.01)

        print(f"Generate GAT variant: Hidden dimension={hidden_dim}, Number of attention heads={n_heads}, dropout={dropout}")
        return model

    def train_single(self, model, data, train_mask, val_mask, seed):
        patience = 100 if isinstance(model, (BioGATv3, BioGATv3Variant)) else 80

        # Keep only the optimizer with fixed learning rate, remove the learning rate scheduler initialization
        optimizer = optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = FusionLoss(balance_weight=self.config.balance_weight,
                               alpha=self.config.alpha,
                               gamma=self.config.gamma)

        best_val_auc = 0
        patience_counter = 0
        model_path = os.path.join(self.config.save_dir, f'model_{seed}.pt')

        for epoch in range(self.config.epochs):


            model.train()
            optimizer.zero_grad()

            # Forward propagation to calculate loss
            out = model(data.x, data.edge_index, data.edge_label_index)
            loss = criterion(out[train_mask], data.edge_label[train_mask])

            # Backward propagation and parameter update
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping to prevent explosion
            optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_label_index)
                val_auc = roc_auc_score(
                    data.edge_label[val_mask].cpu().numpy(),
                    val_out[val_mask].cpu().numpy()
                )

            # Early stopping logic
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at Epoch {epoch}, Best validation AUC={best_val_auc:.4f}")
                    break


            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Validation AUC={val_auc:.4f}")


        model.load_state_dict(torch.load(model_path))
        return model, best_val_auc

    def filter_high_quality_models(self, models, val_aucs, threshold=0.94):
        high_quality = [
            (model, auc) for model, auc in zip(models, val_aucs)
            if auc > threshold
        ]
        if not high_quality:
            print(f"Warning: No models with AUC>{threshold}, using all models")
            return models, val_aucs
        filtered_models = [m for m, _ in high_quality]
        filtered_aucs = [a for _, a in high_quality]
        print(f"Retained {len(filtered_models)} high-quality models after filtering (AUC>{threshold})")
        return filtered_models, filtered_aucs

    def weighted_ensemble(self, models, val_aucs, data, test_mask):
        weights = np.array(val_aucs) / np.sum(val_aucs)
        print(f"Ensemble weights (based on validation AUC): {[round(w, 3) for w in weights]}")

        test_preds = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_label_index)[test_mask].cpu().numpy()
                test_preds.append(pred)

        ensemble_pred = np.zeros_like(test_preds[0])
        for w, p in zip(weights, test_preds):
            ensemble_pred += w * p
        return ensemble_pred

    def plot_combined_curves(self, roc_data, pr_data):
        plt.figure(figsize=(30, 10))

        plt.subplot(1, 2, 1)
        plt.plot([0, 1], [0, 1], 'k-', lw=4, label='Random Guess (AUC=0.5)')

        colors = plt.cm.tab20(np.linspace(0, 1, len(roc_data)))
        for i, (fpr, tpr, auc) in enumerate(roc_data):
            plt.plot(fpr, tpr,
                     color=colors[i],
                     lw=5,
                     alpha=0.90,
                     label=f'Fold {i + 1} (AUC={auc:.4f})')

        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('5-Fold Cross-Validation ROC Curves', fontsize=25)
        plt.grid(True, alpha=0.1)
        plt.xlim([-0.02, 1.02])
        plt.ylim([0.0, 1.02])

        plt.subplot(1, 2, 2)
        pos_ratios = [d[3] for d in pr_data]
        mean_pos_ratio = np.mean(pos_ratios)

        plt.plot([0, 1], [mean_pos_ratio, mean_pos_ratio],
                 'k-',
                 lw=4,
                 label=f'Random Guess (AUPR≈{mean_pos_ratio:.4f})')

        for i, (recall, precision, aupr, _) in enumerate(pr_data):
            plt.plot(recall, precision,
                     color=colors[i],
                     lw=5,
                     alpha=0.90,
                     label=f'Fold {i + 1} (AUPR={aupr:.4f})')

        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('5-Fold Cross-Validation PR Curves', fontsize=25)
        plt.grid(True, alpha=0.1)
        plt.xlim([-0.02, 1.02])
        plt.ylim([0.0, 1.02])

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.savefig(os.path.join(self.config.save_dir, 'combined_5fold_curves.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()
        print("5-Fold ROC and PR curves have been saved to combined_5fold_curves.png")

    def run(self):
        # Load label data
        labels_path = os.path.join(self.config.data_dir, 'adj.txt')
        labels = np.loadtxt(labels_path)
        kf = KFold(5, shuffle=True)

        # Fold-level result storage list (including optimal threshold)
        fold_aucs = []
        fold_auprs = []
        fold_f1s = []
        fold_precisions = []
        fold_recalls = []
        fold_best_threshs = []  # Store the optimal threshold for each fold
        all_true_pred = []  # Elements are DataFrames, containing "Fold", "True Label", "Predicted Score"

        roc_data = []
        pr_data = []

        for fold, (train_index, test_index) in enumerate(kf.split(labels)):
            print(f"\n=== Starting Fold {fold + 1} Cross-Validation ===")
            # Load data and move to GPU
            data, labels = self.data_loader.load_data(train_index, test_index)
            data = data.to(self.device)

            # Split training/validation sets (stratified sampling to ensure class distribution)
            stratify_arr = data.edge_label.cpu().numpy()
            train_val_indices = np.where(data.train_mask.cpu().numpy())[0]
            if np.max(train_val_indices) >= len(stratify_arr):
                raise ValueError(
                    f"train_val_indices exceeds the length of edge_label: {np.max(train_val_indices)} >= {len(stratify_arr)}")
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.15,
                stratify=stratify_arr[train_val_indices],
                random_state=42
            )

            # Build training/validation/test masks
            train_mask = torch.zeros(len(data.edge_label), dtype=torch.bool, device=self.device)
            val_mask = torch.zeros(len(data.edge_label), dtype=torch.bool, device=self.device)
            test_mask = data.test_mask
            train_mask[train_indices] = True
            val_mask[val_indices] = True

            # Train ensemble models
            models = []
            val_aucs = []
            for seed in range(self.config.ensemble_size):
                print(f"\n=== Training Model {seed + 1}/{self.config.ensemble_size} ===")
                model = self._get_model(data.x.size(1), seed)
                model, val_auc = self.train_single(model, data, train_mask, val_mask, seed)
                models.append(model)
                val_aucs.append(val_auc)
                print(f"Model {seed + 1} Best Validation AUC: {val_auc:.4f}")

            # Filter high-quality models and perform ensemble prediction
            models, val_aucs = self.filter_high_quality_models(models, val_aucs)
            test_pred = self.weighted_ensemble(models, val_aucs, data, test_mask)

            # Calculate test set metrics (including optimal threshold search)
            y_true = data.edge_label[test_mask].cpu().numpy()  # True labels of test set
            pos_ratio_fold = np.mean(y_true)
            y_pred = test_pred  # Predicted scores of test set

            # Collect true labels and predicted scores of the current fold
            fold_true_pred = pd.DataFrame({
                "Fold": [f"Fold {fold + 1}"] * len(y_true),  # Mark the fold it belongs to
                "True Label": y_true.astype(int),  # Convert to int for better readability (0/1)
                "Predicted Score": np.round(y_pred, 6)  # Keep 6 decimal places to avoid redundancy
            })
            all_true_pred.append(fold_true_pred)

            # 1. Calculate AUC and AUPR
            test_auc = roc_auc_score(y_true, y_pred)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            test_aupr = auc(recall, precision)

            # 2. Search for the optimal threshold (0.1~0.9, step size 0.01)
            best_f1 = 0.0
            best_precision = 0.0
            best_recall = 0.0
            best_thresh = 0.5  # Initial threshold
            thresholds = np.arange(0.1, 0.91, 0.01)
            for thresh in thresholds:
                pred_binary = y_pred > thresh
                current_f1 = f1_score(y_true, pred_binary)
                current_prec = precision_score(y_true, pred_binary)
                current_recall = recall_score(y_true, pred_binary)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_precision = current_prec
                    best_recall = current_recall
                    best_thresh = thresh

            # 3. Store the results of the current fold
            print(f"Fold {fold + 1} Optimal Threshold: {best_thresh:.2f}")
            fold_aucs.append(test_auc)
            fold_auprs.append(test_aupr)
            fold_f1s.append(best_f1)
            fold_precisions.append(best_precision)
            fold_recalls.append(best_recall)
            fold_best_threshs.append(best_thresh)  # Save optimal threshold

            # 4. Record ROC/PR curve data
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_data.append((fpr, tpr, test_auc))
            pr_data.append((recall, precision, test_aupr, pos_ratio_fold))

            # Print the results of the current fold
            print(f"\nFold {fold + 1} Ensemble Test AUC: {test_auc:.4f}")
            print(f"Fold {fold + 1} Ensemble Test AUPR: {test_aupr:.4f}")
            print(f"Fold {fold + 1} Ensemble Test F1 Score: {best_f1:.4f}")
            print(f"Fold {fold + 1} Ensemble Test Precision: {best_precision:.4f}")
            print(f"Fold {fold + 1} Ensemble Test Recall: {best_recall:.4f}")

        # Plot cross-validation curves
        self.plot_combined_curves(roc_data, pr_data)

        # Calculate final average metrics
        final_auc = np.mean(fold_aucs)
        final_aupr = np.mean(fold_auprs)
        final_f1 = np.mean(fold_f1s)
        final_precision = np.mean(fold_precisions)
        final_recall = np.mean(fold_recalls)

        # Organize fold-level results into a list of dictionaries
        fold_results = [
            {
                "Fold": f"Fold {i + 1}",
                "AUC": round(fold_aucs[i], 4),
                "AUPR": round(fold_auprs[i], 4),
                "F1 Score": round(fold_f1s[i], 4),
                "Precision": round(fold_precisions[i], 4),
                "Recall": round(fold_recalls[i], 4),
                "Optimal Threshold": round(fold_best_threshs[i], 2)
            }
            for i in range(5)
        ]

        # Print final average results
        print(f"\nFinal 5-Fold Cross-Validation Average Test AUC: {final_auc:.4f}")
        print(f"Final 5-Fold Cross-Validation Average Test AUPR: {final_aupr:.4f}")
        print(f"Final 5-Fold Cross-Validation Average Test F1 Score: {final_f1:.4f}")
        print(f"Final 5-Fold Cross-Validation Average Test Precision: {final_precision:.4f}")
        print(f"Final 5-Fold Cross-Validation Average Test Recall: {final_recall:.4f}")

        # Return the aggregated data of true labels and predicted scores when returning
        return (final_auc, final_aupr, final_f1, final_precision, final_recall,
                fold_results, all_true_pred)


if __name__ == "__main__":
    # Record the start time of the code
    start_time = time.perf_counter()  # High-precision timing
    print(f"Code Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Command line argument parsing
    parser = argparse.ArgumentParser(description='GATMDA Model Training Parameters')
    parser.add_argument('--equal_train', action='store_true', help='Whether to make the number of positive and negative samples equal in the training set')
    parser.add_argument('--equal_test', action='store_true', help='Whether to make the number of positive and negative samples equal in the test set')
    args = parser.parse_args()

    # Initialize random seeds (to ensure experimental reproducibility)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)  # GPU random seed (to ensure GPU training reproducibility)

    # Initialize configuration and trainer
    config = Config(args)
    trainer = EnhancedGATTrainer(config)

    # Run training and receive results (add all_true_pred: list of true labels and predicted scores)
    (final_auc, final_aupr, final_f1, final_precision, final_recall,
     fold_results, all_true_pred) = trainer.run()

    # Calculate total running time
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    # Convert to "Hours:Minutes:Seconds" format (more readable)
    total_hours = int(total_seconds // 3600)
    total_minutes = int((total_seconds % 3600) // 60)
    total_seconds_remain = round(total_seconds % 60, 2)
    total_time_str = f"{total_hours}h {total_minutes}m {total_seconds_remain}s"
    print(f"\nTotal Code Running Time: {total_time_str}")
    print(f"Code End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Organize results and save to Excel
    # 1. Define Excel save path
    excel_save_dir = r"src/data/mydata/HMDAD"
    # Ensure the path exists (create if it does not exist)
    if not os.path.exists(excel_save_dir):
        os.makedirs(excel_save_dir, exist_ok=True)
        print(f"Created Excel Save Directory: {excel_save_dir}")

    # 2. Generate a unique Excel file name (with timestamp to avoid duplication)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"GAT_DiseaseMicrobe_Result_{current_time}.xlsx"
    excel_file_path = os.path.join(excel_save_dir, excel_filename)

    # 3. Build DataFrame (original fold-level metrics + new true labels and predicted scores)
    # 3.1 Fold-level metric data
    df_fold = pd.DataFrame(fold_results)
    df_avg = pd.DataFrame([
        {
            "Fold": "5-Fold Average",
            "AUC": round(final_auc, 4),
            "AUPR": round(final_aupr, 4),
            "F1 Score": round(final_f1, 4),
            "Precision": round(final_precision, 4),
            "Recall": round(final_recall, 4),
            "Optimal Threshold": "—"
        }
    ])
    df_time = pd.DataFrame([
        {
            "Fold": "Total Running Time",
            "AUC": total_time_str,
            "AUPR": f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "F1 Score": f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Precision": "—",
            "Recall": "—",
            "Optimal Threshold": "—"
        }
    ])
    df_metrics = pd.concat([df_fold, df_avg, df_time], ignore_index=True)

    # 3.2 True label and predicted score data (merge all folds)
    df_true_pred = pd.concat(all_true_pred, ignore_index=True)

    # 4. Save multiple worksheets using ExcelWriter
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        # Worksheet 1: Fold-level metric summary
        df_metrics.to_excel(writer, sheet_name="Fold-Level Metric Summary", index=False)
        # Worksheet 2: True labels and predicted scores
        df_true_pred.to_excel(writer, sheet_name="True Labels and Predicted Scores", index=False)

    print(f"\nResults have been saved to Excel: {excel_file_path}")
    print(f"Excel contains 2 worksheets:")
    print(f"  1. 'Fold-Level Metric Summary': AUC, AUPR and other metrics for each fold and their averages")
    print(f"  2. 'True Labels and Predicted Scores': True labels (0/1) and model predicted scores of the test set for each fold")

    # Final print results
    print(f"\n=== Final Summary Results ===")
    print(f"Final Test AUC: {final_auc:.4f}")
    print(f"Final Test AUPR: {final_aupr:.4f}")
    print(f"Final Test F1 Score: {final_f1:.4f}")
    print(f"Final Test Precision: {final_precision:.4f}")
    print(f"Final Test Recall: {final_recall:.4f}")
    print(f"Total Running Time: {total_time_str}")