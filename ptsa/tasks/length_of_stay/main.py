import argparse
import os
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from sklearn.model_selection import train_test_split

import numpy as np

from ptsa.tasks.length_of_stay.utils import BatchGen
from ptsa.tasks.readers import LengthOfStayReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.length_of_stay.utils import utils

from ptsa.models.porbabilistic.lstm import BayesianLSTM


def parse_arguments():
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


    return parser.parse_args()


def prepare_data(args, batch_size: int):
    # Read all data
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
    test_reader._data = test_data:

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

    return train_data_gen, val_data_gen, test_data_gen, discretizer_header

def train(model, train_data_gen, val_data_gen, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = 0.0002055432739893699
    num_epochs = 20
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = MSELoss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for i in range(train_data_gen.steps):
            batch = next(train_data_gen)
            x, y = batch
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(val_data_gen.steps):
                batch = next(val_data_gen)
                x, y = batch
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor(y).to(device)

                outputs = model(x)
                val_loss += criterion(outputs, y).item()

        val_loss /= val_data_gen.steps
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model.pth")
        
    print("Training completed. Best model saved as 'best_model.pth'")

def evaluate(model, test_data_gen):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion_mse = MSELoss()
    total_mse = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(test_data_gen.steps):
            batch = next(test_data_gen)
            x, y = batch
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).unsqueeze(1).to(device)

            outputs = model(x)
            mse = criterion_mse(outputs, y).item()
            total_mse += mse * x.size(0)
            total_samples += x.size(0)

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_mse = total_mse / total_samples
    rmse = np.sqrt(avg_mse)

    print(f"Test MSE: {avg_mse}")
    print(f"Test RMSE: {rmse}")

    return avg_mse, rmse

def main():
    args = parse_arguments()
    
    batch_size = 64

    train_data_gen, val_data_gen, test_data_gen, discretizer_header = prepare_data(args, batch_size)
    
    input_dim = len(discretizer_header)
    """model = BayesianLSTM(input_dim=input_dim,
                         hidden_dim=args.dim,
                         output_length=1,
                         batch_size=args.batch_size,
                         dropout=args.dropout,
                         )
    """

    hidden_dim = 92
    hidden_size_2 = 92
    depth = 1
    dropout = 0.4778582594500034
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    model = BayesianLSTM(input_dim, batch_size, output_length=1,
                 hidden_dim=hidden_dim, num_layers=depth,
                 dropout=dropout, hidden_size_2=hidden_size_2, device=device)

    train(model, train_data_gen, val_data_gen, args)

    model.load_state_dict(torch.load("best_model.pth"))
    mse, rmse = evaluate(model, test_data_gen)

if __name__ == "__main__":
    main()
