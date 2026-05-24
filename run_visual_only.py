import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists, get_device


MODALITY = "vision"
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 1111


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data, split):
    x = data[split][MODALITY].astype(np.float32)
    y = np.array(data[split]["regression_labels"]).astype(np.float32)
    x = x.mean(axis=1)
    return torch.tensor(x), torch.tensor(y).view(-1, 1)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    nonzero = labels != 0
    pred_binary = preds[nonzero] > 0
    label_binary = labels[nonzero] > 0

    return {
        "Non0_acc_2": accuracy_score(label_binary, pred_binary),
        "Non0_F1_score": f1_score(label_binary, pred_binary),
        "MAE": mean_absolute_error(labels, preds),
        "Corr": pearsonr(labels, preds)[0],
    }


def main():
    set_seed(SEED)
    device = get_device()
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)

    with feature_path.open("rb") as f:
        data = pickle.load(f)

    x_train, y_train = load_split(data, "train")
    x_valid, y_valid = load_split(data, "valid")
    x_test, y_test = load_split(data, "test")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    best_valid_mae = float("inf")
    best_state = None

    print("Experiment: Visual-only")
    print("Device:", device)
    print("Feature file:", feature_path)
    print("Input dim:", x_train.shape[1])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        valid_result = evaluate(model, valid_loader, device)
        if valid_result["MAE"] < best_valid_mae:
            best_valid_mae = valid_result["MAE"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:02d} | train_loss={np.mean(train_losses):.4f} | valid_MAE={valid_result['MAE']:.4f}")

    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, device)
    print("Test result:", test_result)


if __name__ == "__main__":
    main()
