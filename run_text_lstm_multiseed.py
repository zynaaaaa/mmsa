import argparse
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists


DEFAULT_SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
METRIC_NAMES = [
    "Has0_acc_2", "Has0_F1_score",
    "Non0_acc_2", "Non0_F1_score",
    "Mult_acc_5", "Mult_acc_7",
    "MAE", "Corr",
]


def _multiclass_acc(y_pred, y_true):
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


class TextLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        pooled = outputs.mean(dim=1)
        return self.regressor(pooled)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data, split):
    x = data[split]["text"].astype(np.float32)
    y = np.array(data[split]["regression_labels"]).astype(np.float32)
    return torch.tensor(x), torch.tensor(y).view(-1, 1)


def evaluate(model, loader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            preds.append(pred.numpy())
            labels.append(y.numpy())

    preds = np.concatenate(preds).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    has0_pred_binary = (preds >= 0).astype(int)
    has0_label_binary = (labels >= 0).astype(int)

    nonzero = labels != 0
    non0_pred_binary = (preds[nonzero] > 0).astype(int)
    non0_label_binary = (labels[nonzero] > 0).astype(int)

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    labels_a7 = np.clip(labels, a_min=-3., a_max=3.)
    preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    labels_a5 = np.clip(labels, a_min=-2., a_max=2.)

    return {
        "Has0_acc_2": accuracy_score(has0_label_binary, has0_pred_binary),
        "Has0_F1_score": f1_score(has0_label_binary, has0_pred_binary, average='weighted'),
        "Non0_acc_2": accuracy_score(non0_label_binary, non0_pred_binary),
        "Non0_F1_score": f1_score(non0_label_binary, non0_pred_binary, average='weighted'),
        "Mult_acc_5": _multiclass_acc(preds_a5, labels_a5),
        "Mult_acc_7": _multiclass_acc(preds_a7, labels_a7),
        "MAE": mean_absolute_error(labels, preds),
        "Corr": pearsonr(labels, preds)[0],
    }


def run_one_seed(data, seed):
    set_seed(seed)

    x_train, y_train = load_split(data, "train")
    x_valid, y_valid = load_split(data, "valid")
    x_test, y_test = load_split(data, "test")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = TextLSTMRegressor(input_dim=x_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    best_valid_mae = float("inf")
    best_state = None

    print(f"\nSeed {seed} | Input shape: {tuple(x_train.shape[1:])}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        valid_result = evaluate(model, valid_loader)
        if valid_result["MAE"] < best_valid_mae:
            best_valid_mae = valid_result["MAE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} | "
            f"valid_MAE={valid_result['MAE']:.4f}"
        )

    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader)
    print(f"Seed {seed} result: {test_result}")
    return test_result


def summarize(results):
    summary = {}
    for metric in METRIC_NAMES:
        values = np.array([result[metric] for result in results], dtype=np.float64)
        summary[metric] = {
            "mean": values.mean(),
            "std": values.std(),
        }
    return summary


def print_summary(summary):
    print("\nFinal mean ± std:")
    for metric in METRIC_NAMES:
        mean = summary[metric]["mean"]
        std = summary[metric]["std"]
        print(f"{metric}: {mean:.4f} ± {std:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Text-LSTM strong unimodal baseline on MOSI.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)

    print("Experiment: Text-LSTM strong unimodal baseline")
    print("Feature file:", feature_path)
    print("Seeds:", args.seeds)
    print("Model: BiLSTM + mean pooling + MLP regressor")

    with feature_path.open("rb") as f:
        data = pickle.load(f)

    results = [run_one_seed(data, seed) for seed in args.seeds]
    print_summary(summarize(results))


if __name__ == "__main__":
    main()
