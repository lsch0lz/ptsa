import os
import logging
import argparse
from typing import Dict, List

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

from ptsa.models.probabilistic.lstm_classification import LSTM 
from ptsa.models.probabilistic.rnn_classification import RNN
from ptsa.models.probabilistic.gru_classification import GRU

logging.basicConfig(level=logging.INFO)

class IHMProbabilisticInference:
    def __init__(self, config: Dict, data_path: str, model_path: str, model_name: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config 
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name

    def _load_model(self):
        if self.model_name == "LSTM":
            model = LSTM(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "RNN":
            model = RNN(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)
        elif self.model_name == "GRU":
            model = GRU(self.config["input_size"], self.config["hidden_size"], self.config["num_layers"], self.config["dropout"]).to(self.device)

        model.load_state_dict(torch.load(self.model_path, weights_only=True))

        return model

    def even_out_number_of_data_points(self, data):
        data_points, labels = data[0], data[1]
        self.logger.info("Number of Samples: %s", len(labels))
        # Separate indices by class
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]
        
        self.logger.info("Number of Positive Samples: %s", len(positive_indices))
        self.logger.info("Number of Negative Samples: %s", len(negative_indices))
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
        self.logger.info("Number of Balanced Samples: %s", len(balanced_labels))
        
        return (balanced_data_points, balanced_labels)

    
    def load_test_data(self):
        # Data loading and preprocessing
        all_reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(self.data_path, 'train'), 
            listfile=os.path.join(self.data_path, 'train/listfile.csv'), 
            period_length=48.0
        )

        train_data, val_data = train_test_split(all_reader._data, test_size=0.2, random_state=43)

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'), listfile=os.path.join(self.data_path, 'train/listfile.csv'), period_length=48.0)
        train_reader._data = train_data

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'train'), listfile=os.path.join(self.data_path, 'train/listfile.csv'), period_length=48.0)
        val_reader._data = val_data

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(self.data_path, 'test'), listfile=os.path.join(self.data_path, 'test/listfile.csv'), period_length=48.0)

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
        train_raw_data = load_data(train_reader, discretizer, normalizer, False)
        val_raw_data = load_data(val_reader, discretizer, normalizer, False)
        test_raw_data = load_data(test_reader, discretizer, normalizer, False)

        train_raw = self.even_out_number_of_data_points(train_raw_data)
        val_raw = self.even_out_number_of_data_points(val_raw_data)
        test_raw = self.even_out_number_of_data_points(test_raw_data)

        return train_raw, val_raw, test_raw

    def infer_on_data_points(self, test_data):
        model = self._load_model()
        # Loss and Optimizer

        # Final testing and metrics logging
        model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []

        with torch.no_grad():
            for i in range(len(test_data[0])):
                x, y = test_data[0][i], test_data[1]
                x = torch.FloatTensor(x).to(self.device)
                y = torch.FloatTensor([y[i] if isinstance(y, (list, np.ndarray)) else y]).to(self.device)

                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                # outputs = model(x).view(-1)

                mean, variance = model.predict_with_uncertainty(x, num_samples=self.config["num_mc_samples"])
                
                all_predictions.append(mean.cpu().numpy())
                all_uncertainties.append(variance.cpu().numpy())
                all_targets.append(y.cpu().numpy())


        return all_predictions, all_uncertainties 
