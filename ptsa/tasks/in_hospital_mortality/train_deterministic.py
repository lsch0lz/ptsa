import os
import re
import argparse
import numpy as np
import torch
from torch import nn

import wandb
from tqdm import tqdm
import optuna

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, roc_auc_score

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data
from ptsa.utils import utils
from ptsa.utils import metrics

from ptsa.models.deterministic.lstm_classification import LSTM 
from ptsa.models.deterministic.rnn import RNN
from ptsa.models.deterministic.gru import GRU

def objective(trial):
    # Hyperparameters to tune
    config = {
        "input_size": 76,  # Fixed based on input data
        "hidden_size": trial.suggest_int('hidden_size', 32, 256),
        "num_layers": trial.suggest_int('num_layers', 1, 4),
        "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        "batch_size": trial.suggest_categorical('batch_size', [32, 64, 128]),
        "dropout": trial.suggest_uniform('dropout', 0.1, 0.5),
        "num_epochs": 10  # Could also be tuned, but keeping constant for now
    }

    # Initialize wandb run for this trial
    wandb.init(
        project="ihm_lstm_optuna", 
        config=config,
        group=f"trial_{trial.number}"
    )

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loading and preprocessing
    train_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'train'), 
        listfile=os.path.join(args.data, 'train/listfile.csv'), 
        period_length=48.0
    )
    val_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, 'train'), 
        listfile=os.path.join(args.data, 'train/listfile.csv'), 
        period_length=48.0
    )

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'), listfile=os.path.join(args.data, 'test/listfile.csv'), period_length=48.0)
    
    # Discretizer and Normalizer setup (similar to original script)
    discretizer = Discretizer(
        timestep=float(args.timestep), 
        store_masks=True, 
        impute_strategy='previous', 
        start_time='zero'
    )
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = f'ihm_ts{args.timestep}.input_str_{args.imputation}.start_time_zero.normalizer'
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    # Load data
    train_raw = load_data(train_reader, discretizer, normalizer, args.small_part)
    val_raw = load_data(val_reader, discretizer, normalizer, args.small_part)
    test_raw = load_data(test_reader, discretizer, normalizer, args.small_part)

    # Build the model
    model = nn.Module()
    if args.model == "lstm":
        model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
    elif args.model == "rnn":
        model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
    elif args.model == "gru":
        model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        for i in range(len(train_raw[0])):
            x, y = train_raw[0][i], train_raw[1]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)

            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            optimizer.zero_grad()
            outputs = model(x).view(-1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_raw[0])

        
        model.eval()
        val_loss = 0
        all_correct = 0
        total_samples = 0
        
        for i in range(len(val_raw[0])):
            x, y = val_raw[0][i], val_raw[1]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)
            
            # adding batch dim
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            outputs = model(x)

            outputs = outputs.view(-1)
            loss = criterion(outputs, y)
            val_loss += loss.item()

            # Compute accuracy
            predicted = (outputs > 0.5).float()
            all_correct += (predicted == y).float().sum().item()
            total_samples += y.size(0)

        val_loss /= len(val_raw[0])

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}.epoch{epoch}.pth"))
       
        accuracy = all_correct / total_samples
    
        # Report for pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    # TESING
    model.eval()
    all_predictions = []
    all_targets = []
        
    all_correct_testing = 0
    total_samples_testing = 0
    with torch.no_grad():
        for i in range(len(test_raw[0])):
            x, y = test_raw[0][i], test_raw[1]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(device)

            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            outputs = model(x).view(-1)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
            # Compute accuracy
            predicted_testing = (outputs > 0.5).float()
            all_correct_testing += (predicted_testing == y).float().sum().item()
            total_samples_testing += y.size(0)
    
    # Compute metrics
    predictions = [pred[0] for pred in all_predictions]
    targets = [target[0] for target in all_targets]
    
    # Compute AUC-ROC as the objective metric
    auc_roc = roc_auc_score(targets, predictions)
    
    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(targets, predictions)
    avg_precision = average_precision_score(targets, predictions)

    # Accuracy
    accuracy = all_correct_testing / total_samples_testing

    # Create Precision-Recall Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Trial {trial.number}')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "average_precision": avg_precision,
        "precision_recall_curve": wandb.Image(plt)
    })

    plt.close()
    
    # Close wandb run for this trial
    wandb.finish()

    return auc_roc

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    utils.add_common_arguments(parser)
    parser.add_argument('--data', type=str, 
        help='Path to the data of in-hospital mortality task', 
        default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/')
    )
    parser.add_argument('--output_dir', type=str, 
        help='Directory relative which all output files are stored', 
        default='.'
    )
    parser.add_argument('--num_trials', type=int, default=10, 
        help='Number of Optuna trials to run')
    
    parser.add_argument('--model', type=str, default="lstm", help="lstm, rnn, gru, transformer")
    parser.add_argument("--model_name", type=str, help="Name for the model file")


    parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")

    global args
    args = parser.parse_args()

    # Create a study object and specify the direction is 'maximize'
    study = optuna.create_study(
        direction='maximize', 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    # Run the hyperparameter optimization
    study.optimize(objective, n_trials=args.num_trials)

    # Print the best trial
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value (AUC-ROC): ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Optional: Save the best hyperparameters
    with open(os.path.join(args.output_dir, 'best_hyperparams.txt'), 'w') as f:
        f.write(f"Best AUC-ROC: {trial.value}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    # Optional: Visualize hyperparameter importance
    optuna.visualization.plot_optimization_history(study)
    plt.savefig(os.path.join(args.output_dir, 'optimization_history.png'))
    optuna.visualization.plot_param_importances(study)
    plt.savefig(os.path.join(args.output_dir, 'param_importances.png'))

if __name__ == "__main__":
    main()

