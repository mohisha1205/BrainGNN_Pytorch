import os
import copy
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch_geometric.loader import DataLoader
from net.braingnn import Network
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--weightdecay', type=float, default=0.02)

parser.add_argument('--lamb0', type=float, default=1)
parser.add_argument('--lamb1', type=float, default=0.1)
parser.add_argument('--lamb2', type=float, default=0.1)
parser.add_argument('--lamb3', type=float, default=0.1)
parser.add_argument('--lamb4', type=float, default=0.1)
parser.add_argument('--lamb5', type=float, default=0.1)

parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--indim', type=int, default=200)
parser.add_argument('--nclass', type=int, default=2)

parser.add_argument('--save_path', type=str, default='./cv_models/')
parser.add_argument('--plot_path', type=str, default='./plots/')
parser.add_argument('--csv_path', type=str, default='./plots/fold_metrics.csv')
parser.add_argument('--val_size', type=float, default=0.2)

opt = parser.parse_args()

os.makedirs(opt.save_path, exist_ok=True)
os.makedirs(opt.plot_path, exist_ok=True)

dataset = torch.load("./data/processed/abide_graph_dataset.pt", weights_only=False)
labels = np.array([d.y.item() for d in dataset])

print("Dataset size:", len(dataset))

def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    k = max(1, int(s.size(1) * ratio))
    return -torch.log(s[:, -k:] + EPS).mean() - torch.log(1 - s[:, :k] + EPS).mean()

def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0]).to(device)
    D = torch.eye(s.shape[0]).to(device) * torch.sum(W, dim=1)
    L = D - W
    return torch.trace(s.t() @ L @ s) / (s.shape[0] * s.shape[0])

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    loss_all = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        output, w1, w2, s1, s2 = model(
            data.x, data.edge_index, data.batch, data.edge_attr, data.pos
        )

        loss_c = F.nll_loss(output, data.y)
        loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
        loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
        loss_tpk1 = topk_loss(s1, opt.ratio)
        loss_tpk2 = topk_loss(s2, opt.ratio)

        loss_consist = 0
        for c in range(opt.nclass):
            cls_mask = (data.y == c)
            if cls_mask.sum() > 0:
                loss_consist += consist_loss(s1[cls_mask])

        loss = (
            opt.lamb0 * loss_c
            + opt.lamb1 * loss_p1
            + opt.lamb2 * loss_p2
            + opt.lamb3 * loss_tpk1
            + opt.lamb4 * loss_tpk2
            + opt.lamb5 * loss_consist
        )

        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs

    scheduler.step()
    return loss_all / len(loader.dataset)

def eval_metrics(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)[0]
            pred = output.max(1)[1]
            y_true.extend(data.y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    if cm.shape != (2, 2):
        tn = fp = fn = tp = 0
        if len(y_true) > 0:
            unique = sorted(set(y_true + y_pred))
            if unique == [0]:
                tn = len(y_true)
            elif unique == [1]:
                tp = len(y_true)
    else:
        tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn + EPS)
    sensitivity = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)

    return acc, sensitivity, specificity

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n========== FOLD {fold + 1} ==========")

    train_val_labels = labels[train_val_idx]

    inner_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=opt.val_size,
        random_state=42 + fold
    )

    inner_train_rel, inner_val_rel = next(inner_split.split(np.zeros(len(train_val_idx)), train_val_labels))
    train_idx = train_val_idx[inner_train_rel]
    val_idx = train_val_idx[inner_val_rel]

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=opt.batchSize, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=opt.batchSize, shuffle=False)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=opt.batchSize, shuffle=False)

    model = Network(opt.indim, opt.ratio, opt.nclass).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    best_val = -1.0
    patience = 12
    counter = 0
    best_wts = copy.deepcopy(model.state_dict())

    train_acc_hist = []
    val_acc_hist = []

    for epoch in range(opt.n_epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        tr_acc, _, _ = eval_metrics(model, train_loader)
        val_acc, _, _ = eval_metrics(model, val_loader)

        train_acc_hist.append(tr_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Train {tr_acc:.4f} | Val {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            counter = 0
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, f"{opt.save_path}/fold{fold + 1}.pth")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_wts)
    test_acc, test_sens, test_spec = eval_metrics(model, test_loader)

    print("Fold Test Accuracy:", test_acc)
    print("Fold Sensitivity:", test_sens)
    print("Fold Specificity:", test_spec)

    fold_results.append({
        "fold": fold + 1,
        "accuracy": test_acc,
        "sensitivity": test_sens,
        "specificity": test_spec
    })

    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.title(f"Fold {fold + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{opt.plot_path}/fold{fold + 1}.png")
    plt.close()

with open(opt.csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fold", "accuracy", "sensitivity", "specificity"])
    writer.writeheader()
    writer.writerows(fold_results)

accs = [x["accuracy"] for x in fold_results]
sens = [x["sensitivity"] for x in fold_results]
spec = [x["specificity"] for x in fold_results]

print("\n================ FINAL RESULTS ================")
print("Fold Accuracies:", accs)
print("Mean Accuracy:", np.mean(accs))
print("Std Accuracy:", np.std(accs))
print("Mean Sensitivity:", np.mean(sens))
print("Std Sensitivity:", np.std(sens))
print("Mean Specificity:", np.mean(spec))
print("Std Specificity:", np.std(spec))
print("Saved CSV to:", opt.csv_path)
