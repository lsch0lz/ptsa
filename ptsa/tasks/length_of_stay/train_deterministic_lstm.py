import os
import argparse

import torch
from torch import nn
from sklearn.model_selection import train_test_split


from ptsa.tasks.readers import LengthOfStayReader
from ptsa.tasks.length_of_stay.utils import BatchGen
from ptsa.utils.preprocessing import Normalizer, Discretizer
from ptsa.tasks.length_of_stay.utils import utils

from ptsa.models.deterministic.lstm import LSTM 

parser = argparse.ArgumentParser()
utils.add_common_arguments(parser)

parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')

parser.set_defaults(deep_supervision=False)

parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")

parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))

parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')

parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples to use')

args = parser.parse_args()

# Hyperparameters
input_size = 76  # Number of features in your input data
hidden_size = 64
num_layers = 2
learning_rate = 0.001
num_epochs = 50
batch_size = 128

device = "cuda" if torch.cuda.is_available() else "cpu" 

model = LSTM(input_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# data loading
all_data = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                listfile=os.path.join(args.data, 'train/listfile.csv'))

# Split data into train, validation, and test sets
train_val_data, test_data = train_test_split(all_data._data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# Limit training samples if specified
if args.num_train_samples is not None:
    train_data = train_data[:args.num_train_samples]

# Create readers for each split
train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
train_reader._data = train_data

val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
val_reader._data = val_data

test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'))
test_reader._data = test_data

discretizer = Discretizer(timestep=args.timestep,
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'los_ts{}.input_str_previous.start_time_zero.n5e4.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

train_data_gen = BatchGen(reader=train_reader,
                            discretizer=discretizer,
                            normalizer=normalizer,
                            batch_size=batch_size,
                            steps=None,
                            shuffle=True,
                            partition=args.partition)

val_data_gen = BatchGen(reader=val_reader,
                        discretizer=discretizer,
                        normalizer=normalizer,
                        batch_size=batch_size,
                        steps=None,
                        shuffle=False,
                        partition=args.partition)

test_data_gen = BatchGen(reader=test_reader,
                            discretizer=discretizer,
                            normalizer=normalizer,
                            batch_size=batch_size,
                            steps=None,
                            shuffle=False,
                            partition=args.partition)


# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i in range(train_data_gen.steps):
        batch = next(train_data_gen)
        x, y = batch
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / train_data_gen.steps
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    # Assuming you have a separate validation generator
    val_loss = 0
    for i in range(val_data_gen.steps):
        batch = next(val_data_gen)
        x, y = batch
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        
        outputs = model(x)
        loss = criterion(outputs, y)
        val_loss += loss.item()
    
    avg_val_loss = val_loss / val_data_gen.steps
    print(f'Validation Loss: {avg_val_loss:.4f}')
