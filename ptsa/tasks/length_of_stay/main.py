import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from ptsa.tasks.length_of_stay.utils import BatchGen
from ptsa.tasks.readers import LengthOfStayReader
from ptsa.utils.preprocessing import Discretizer, Normalizer
from ptsa.tasks.length_of_stay.utils import metrics, utils

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
    return parser.parse_args()


def prepare_data(args):
    train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'train/listfile.csv'))
    val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                    listfile=os.path.join(args.data, 'train/listfile.csv'))

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
                              batch_size=args.batch_size,
                              steps=None,
                              shuffle=True,
                              partition=args.partition)
    val_data_gen = BatchGen(reader=val_reader,
                            discretizer=discretizer,
                            normalizer=normalizer,
                            batch_size=args.batch_size,
                            steps=None,
                            shuffle=False,
                            partition=args.partition)

    return train_data_gen, val_data_gen, discretizer_header


def train(model, train_data_gen, val_data_gen, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = MSELoss()

    for epoch in range(args.epochs):
        model.train()
        for i in range(train_data_gen.steps):
            batch = next(train_data_gen)
            x, y = batch
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)

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
                x, y = batch['data']
                x = torch.FloatTensor(x).to(device)
                y = torch.FloatTensor(y).to(device)

                outputs = model(x)
                val_loss += criterion(outputs, y).item()

        val_loss /= val_data_gen.steps
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

        # Save model
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


def main():
    args = parse_arguments()
    train_data_gen, val_data_gen, discretizer_header = prepare_data(args)

    input_dim = len(discretizer_header)
    model = BayesianLSTM(input_dim=input_dim,
                         hidden_dim=args.dim,
                         output_dim=1,
                         batch_size=args.batch_size,
                         dropout=args.dropout,
                         rec_dropout=args.rec_dropout)

    train(model, train_data_gen, val_data_gen, args)


if __name__ == "__main__":
    main()
