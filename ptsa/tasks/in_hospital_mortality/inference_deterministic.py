import os
import logging
import argparse
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn

import wandb
import optuna
import random

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from ptsa.tasks.readers import InHospitalMortalityReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.in_hospital_mortality.utils import load_data
from ptsa.utils import utils

from ptsa.models.deterministic.lstm_classification import LSTM 
from ptsa.models.deterministic.rnn_classification import RNN
from ptsa.models.deterministic.gru_classification import GRU

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "input_size": 76,
    "hidden_size": 64,
    "num_layers": 4,
    "learning_rate": 0.0000498373896802271,
    "dropout": 0.2,
    "batch_size": 64,
    "num_epochs": 28,
    "weight_decay": 0.0016314086500572671,
}

DATA_PATH = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality/"
MODEL_PATH = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/rnn/final_model_trial_2.pth"

# model = LSTM(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
model = RNN(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)
# model = GRU(config["input_size"], config["hidden_size"], config["num_layers"], config["dropout"]).to(device)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

def even_out_number_of_data_points(data):
    data_points, labels = data[0], data[1]
    logger.info("Number of Samples: %s", len(labels))
    # Separate indices by class
    positive_indices = [i for i, label in enumerate(labels) if label == 1]
    negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    logger.info("Number of Positive Samples: %s", len(positive_indices))
    logger.info("Number of Negative Samples: %s", len(negative_indices))
    # Determine the target number of samples (equal to the minority class size)
    target_size = min(len(positive_indices), len(negative_indices))

    # Downsample the majority class
    sampled_positive_indices = random.sample(positive_indices, target_size)
    sampled_negative_indices = random.sample(negative_indices, target_size)

    # Combine indices and shuffle
    balanced_indices = sampled_positive_indices + sampled_negative_indices
    random.shuffle(balanced_indices)

    # Create the new balanced dataset
    balanced_data_points = [data_points[i] for i in balanced_indices]
    balanced_labels = [labels[i] for i in balanced_indices]
    logger.info("Number of Balanced Samples: %s", len(balanced_labels))
    return (balanced_data_points, balanced_labels)

# Data loading and preprocessing
all_reader = InHospitalMortalityReader(
    dataset_dir=os.path.join(DATA_PATH, 'train'), 
    listfile=os.path.join(DATA_PATH, 'train/listfile.csv'), 
    period_length=48.0
)

train_data, val_data = train_test_split(all_reader._data, test_size=0.2, random_state=43)

train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(DATA_PATH, 'train'), listfile=os.path.join(DATA_PATH, 'train/listfile.csv'), period_length=48.0)
train_reader._data = train_data

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(DATA_PATH, 'train'), listfile=os.path.join(DATA_PATH, 'train/listfile.csv'), period_length=48.0)
val_reader._data = val_data

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(DATA_PATH, 'test'), listfile=os.path.join(DATA_PATH, 'test/listfile.csv'), period_length=48.0)

# Discretizer and Normalizer setup (similar to original script)
discretizer = Discretizer(
    timestep=float(1.0), 
    store_masks=True, 
    impute_strategy='previous',
    start_time='zero'
)
discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = None
if normalizer_state is None:
    normalizer_state = f'ihm_ts1.0.input_str_previous.start_time_zero.normalizer'
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

# Load data
train_raw_data = load_data(train_reader, discretizer, normalizer, True)
val_raw_data = load_data(val_reader, discretizer, normalizer, True)
test_raw_data = load_data(test_reader, discretizer, normalizer, True)

train_raw = even_out_number_of_data_points(train_raw_data)
val_raw = even_out_number_of_data_points(val_raw_data)
test_raw = even_out_number_of_data_points(test_raw_data)

# Loss and Optimizer
pos_weight_tensor = torch.tensor([1.0], device=device)
criterion = nn.BCELoss(weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# Final testing and metrics logging
model.eval()
all_predictions = []
all_targets = []

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


# Compute metrics
predictions = [pred[0] for pred in all_predictions]
targets = [target[0] for target in all_targets]

# Convert to numpy arrays if needed
targets = np.array(targets)
predictions = np.array(predictions)

binary_predictions = (predictions >= 0.5).astype(int)
f1 = f1_score(targets, binary_predictions)

print(f"F1 Score: {f1}")
