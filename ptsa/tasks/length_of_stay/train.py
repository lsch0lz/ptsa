import os
import numpy as np
import optuna
import torch
import wandb
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ptsa.models.porbabilistic.lstm import BayesianLSTM


def calculate_mse_rmse(x, y, model, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x).float().to(device).unsqueeze(1)
        y_pred, _ = model.predict(x_tensor)
        y_pred = y_pred.cpu().numpy().squeeze() * y_std + y_mean
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
    return mse, rmse

# Set global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df_clean = pd.read_csv("/vol/tmp/scholuka/ptsa/data/los_prediction.csv")
LOS = df_clean['LOS'].values
features = df_clean.drop(columns=['LOS'])

# Split the data into train+val and test sets
x_train_val, x_test, y_train_val, y_test = train_test_split(features, LOS, test_size=0.2, random_state=GLOBAL_SEED)

# Further split train+val into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=GLOBAL_SEED)

# Normalize the target variable
y_train_mean, y_train_std = y_train.mean(), y_train.std()
y_train_normalized = (y_train - y_train_mean) / y_train_std
y_val_normalized = (y_val - y_train_mean) / y_train_std

x_train = x_train.to_numpy().astype(np.float32)
x_val = x_val.to_numpy().astype(np.float32)
x_test = x_test.to_numpy().astype(np.float32)


class DataFeed(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]
        if self.transform:
            sample_x = self.transform(sample_x)
        return sample_x, sample_y


trainset = DataFeed(x_train, y_train_normalized)
valset = DataFeed(x_val, y_val_normalized)


def objective(trial):
    # Hyperparameter definitions remain the same
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    hidden_size_2 = trial.suggest_int('hidden_size_2', 64, 512)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    depth = trial.suggest_int('depth', 1, 3)
    num_epochs = trial.suggest_int('num_epochs', 50, 200)

    input_dim = x_train.shape[1]
    output_dim = 1

    with wandb.init(project="probabilistic_lstm", reinit=True):
        wandb.config.update({
            "batch_size": batch_size,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "depth": depth,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        })

        model = BayesianLSTM(input_dim, batch_size, output_length=1,
                             hidden_dim=hidden_dim, num_layers=depth,
                             dropout=dropout, hidden_size_2=hidden_size_2, device=device).to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate)

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_batch.unsqueeze(1))
                loss = model.loss(y_pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_val_pred = model(x_val.unsqueeze(1))
                    val_loss += model.loss(y_val_pred.squeeze(), y_val).item()

            avg_val_loss = val_loss / len(val_loader)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            trial.report(avg_val_loss, epoch)

            # if trial.should_prune():
            # raise optuna.exceptions.TrialPruned()

        mse, rmse = calculate_mse_rmse(x_test, y_test, model, y_train_mean, y_train_std)

        wandb.log({
            "test_mse": mse,
            "test_rmse": rmse
        })

    return avg_val_loss


# Optuna study creation and optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters: ", study.best_params)

# Re-train the model with the best hyperparameters
best_params = study.best_params
batch_size = best_params['batch_size']
hidden_dim = best_params['hidden_dim']
hidden_size_2 = best_params['hidden_size_2']
learning_rate = best_params['learning_rate']
dropout = best_params['dropout']
depth = best_params['depth']
num_epochs = best_params['num_epochs']

input_dim = x_train.shape[1]

model = BayesianLSTM(input_dim, batch_size, output_length=1,
                     hidden_dim=hidden_dim, num_layers=depth,
                     dropout=dropout, hidden_size_2=hidden_size_2, device=device).to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

best_val_loss = float('inf')
best_model_state = None

with wandb.init(project="probabilistic_lstm", name="final_training"):
    wandb.config.update(best_params)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch.unsqueeze(1))
            loss = model.loss(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_val_pred = model(x_val.unsqueeze(1))
                val_loss += model.loss(y_val_pred.squeeze(), y_val).item()

        avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    model_save_path = os.path.join(wandb.run.dir, "best_model.pth")
    torch.save(best_model_state, model_save_path)
    wandb.save(model_save_path)


    def calculate_mse_rmse(x, y, model, y_mean, y_std):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x).float().to(device).unsqueeze(1)
            y_pred, _ = model.predict(x_tensor)
            y_pred = y_pred.cpu().numpy().squeeze() * y_std + y_mean
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
        return mse, rmse


    # Evaluate on test data
    model.load_state_dict(best_model_state)
    mse, rmse = calculate_mse_rmse(x_test, y_test, model, y_train_mean, y_train_std)

    wandb.log({
        "test_mse": mse,
        "test_rmse": rmse
    })

    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")

print(f"Best model saved to {model_save_path}")
