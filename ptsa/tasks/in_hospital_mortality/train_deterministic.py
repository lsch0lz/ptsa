import os
import re
import argparse
import numpy as np
import torch
from torch import nn

import wandb
from tqdm import tqdm

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data, save_results
from ptsa.utils import utils
from ptsa.utils import metrics

from ptsa.models.deterministic.lstm_classification import LSTM 
from ptsa.models.deterministic.rnn import RNN
from ptsa.models.deterministic.gru import GRU

parser = argparse.ArgumentParser()
utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task', default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored', default='.')
parser.add_argument('--model', type=str, default="lstm", help="lstm, rnn, gru, transformer")
parser.add_argument("--model_name", type=str, help="Name for the model file")
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")


args = parser.parse_args()

config = {
    "input_size": 76,
    "hidden_size": 64,
    "num_layers": 2,
    "learning_rate": args.lr,
    "num_epochs": args.epochs,
    "batch_size": args.batch_size,
    "dropout": 0.2
}

wandb.init(project=f"ihm_{args.model}", config=config)

device = "cuda" if torch.cuda.is_available() else "cpu" 

# Build the model
model = nn.Module()
if args.model == "lstm":
    model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
elif args.model == "rnn":
    model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
elif args.model == "gru":
    model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'), listfile=os.path.join(args.data, 'train/listfile.csv'), period_length=48.0)
val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'), listfile=os.path.join(args.data, 'train/listfile.csv'), period_length=48.0)
test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'), listfile=os.path.join(args.data, 'test/listfile.csv'), period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep), store_masks=True, impute_strategy='previous', start_time='zero')
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())

args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = args.target_repl_coef > 0.0


# Compile the model
optimizer_config = {'class_name': args.optimizer, 'config': {'lr': config["learning_rate"], 'beta_1': args.beta_1}}
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config['config']['lr'])

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_state_dict(torch.load(args.load_state))
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))

# Read data
train_raw = load_data(train_reader, discretizer, normalizer, args.small_part)
val_raw = load_data(val_reader, discretizer, normalizer, args.small_part)
test_raw = load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)

# TODO: Use train_test_split() for validation data split

if args.target_repl_coef > 0.0:
    T = train_raw[0][0].shape[0]
    train_raw[1] = [train_raw[1], np.expand_dims(train_raw[1], axis=-1).repeat(T, axis=1)]
    val_raw[1] = [val_raw[1], np.expand_dims(val_raw[1], axis=-1).repeat(T, axis=1)]

if args.mode == 'train':
    for epoch in tqdm(range(n_trained_chunks + config["num_epochs"])):
        model.train()
        train_loss = 0
        for i in range(len(train_raw[0])):
            x, y = train_raw[0][i], train_raw[1]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            
            # adding batch dim
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            print(f"TRAINING:")
            print(f"X Shape: {x.shape}")
            print(f"Y Shape: {y.shape}")

            print(f"X Value: {x}")
            print(f"Y Value: {y}")

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_raw[0])

        model.eval()
        val_loss = 0
        for i in range(len(val_raw[0])):
            x, y = val_raw[0][i], val_raw[1]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            
            # adding batch dim
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            print(f"VALIDATION:")
            print(f"X Shape: {x.shape}")
            print(f"Y Shape: {y.shape}")

            print(f"X Value: {x}")
            print(f"Y Value: {y}")

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

        val_loss /= len(val_raw[0])

        print(f'Epoch [{epoch+1}/{n_trained_chunks + config["num_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"{model.final_name}.epoch{epoch}.pth"))

elif args.mode == 'test':
    model.eval()
    with torch.no_grad():
        test_data, test_labels, test_names = test_raw
        test_outputs = model(torch.FloatTensor(test_data).to(device))
        test_preds = test_outputs[:, 0].cpu().numpy()

    metrics.print_metrics_binary(test_labels, test_preds)
    save_results(test_names, test_preds, test_labels, os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv")

    mse = np.mean((test_preds - test_labels) ** 2)
    rmse = np.sqrt(mse)

    wandb.log({
        "MSE": mse,
        "RMSE": rmse
    })

else:
    raise ValueError("Wrong value for args.mode")

wandb.finish()
