import threading
import os
import numpy as np
import random

from ptsa.utils import metrics, utils

def drop_columns(data, header, columns_to_drop):
    indices_to_drop = []
    for col in columns_to_drop:
        for i, header_item in enumerate(header):
            if col in header_item:
                indices_to_drop.append(i)
    
    new_header = [h for i, h in enumerate(header) if i not in indices_to_drop]
    
    if isinstance(data, list):
        return [np.delete(x, indices_to_drop, axis=1) for x in data], indices_to_drop, new_header
    else:
        return np.delete(data, indices_to_drop, axis=1), indices_to_drop, new_header


class NormalizerWrapper:
    def __init__(self, normalizer, dropped_indices):
        self.normalizer = normalizer
        self.dropped_indices = dropped_indices
        
        self.index_mapping = {}
        j = 0
        for i in range(len(normalizer._means)):
            if i not in dropped_indices:
                self.index_mapping[j] = i
                j += 1
    
    def transform(self, X):
        ret = X.copy()
        for i in range(X.shape[1]):
            if i in self.index_mapping:
                original_col = self.index_mapping[i]
                if original_col in self.normalizer._fields:
                    ret[:, i] = (X[:, i] - self.normalizer._means[original_col]) / self.normalizer._stds[original_col]
        return ret


def preprocess_chunk(data, ts, discretizer, normalizer=None, columns_to_drop=None):
    transformed_data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]

    if columns_to_drop and len(data) > 0:
        header = discretizer.transform(data[0], end=ts[0])[1].split(',')
        transformed_data, dropped_indices, new_header = drop_columns(transformed_data, header, columns_to_drop)

        if normalizer is not None:
            normalizer_wrapper = NormalizerWrapper(normalizer, dropped_indices)
            transformed_data = [normalizer_wrapper.transform(X) for X in transformed_data]
            return transformed_data

    if normalizer is not None:
        transformed_data = [normalizer.transform(X) for X in transformed_data]
    
    return transformed_data


class BatchGen(object):
    def __init__(self, reader, partition, discretizer, normalizer,
                 batch_size, steps, shuffle, return_names=False, columns_to_drop=None):
        self.reader = reader
        self.partition = partition
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names
        self.columns_to_drop = columns_to_drop

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(1024, self.steps) * batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size

                ret = utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]

                Xs = preprocess_chunk(Xs, ts, self.discretizer, self.normalizer, self.columns_to_drop)
                (Xs, ys, ts, names) = utils.sort_and_shuffle([Xs, ys, ts, names], B)

                for i in range(0, current_size, B):
                    X = utils.pad_zeros(Xs[i:i + B])
                    y = ys[i:i+B]
                    y_true = np.array(y)
                    batch_names = names[i:i+B]
                    batch_ts = ts[i:i+B]

                    if self.partition == 'log':
                        y = [metrics.get_bin_log(x, 10) for x in y]
                    if self.partition == 'custom':
                        y = [metrics.get_bin_custom(x, 10) for x in y]

                    y = np.array(y)

                    if self.return_y_true:
                        batch_data = (X, y, y_true)
                    else:
                        batch_data = (X, y)

                    if not self.return_names:
                        yield batch_data
                    else:
                        yield {"data": batch_data, "names": batch_names, "ts": batch_ts}

    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return next(self.generator)

    def __next__(self):
        return self.next()


class BatchGenDeepSupervision(object):

    def __init__(self, dataloader, partition, discretizer, normalizer,
                 batch_size, shuffle, return_names=False):
        self.partition = partition
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        self._load_per_patient_data(dataloader, discretizer, normalizer)

        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader, discretizer, normalizer):
        timestep = discretizer._timestep

        def get_bin(t):
            eps = 1e-6
            return int(t / timestep - eps)

        N = len(dataloader._data["X"])
        Xs = []
        ts = []
        masks = []
        ys = []
        names = []

        for i in range(N):
            X = dataloader._data["X"][i]
            cur_ts = dataloader._data["ts"][i]
            cur_ys = dataloader._data["ys"][i]
            name = dataloader._data["name"][i]

            cur_ys = [float(x) for x in cur_ys]

            T = max(cur_ts)
            nsteps = get_bin(T) + 1
            mask = [0] * nsteps
            y = [0] * nsteps

            for pos, z in zip(cur_ts, cur_ys):
                mask[get_bin(pos)] = 1
                y[get_bin(pos)] = z

            X = discretizer.transform(X, end=T)[0]
            if normalizer is not None:
                X = normalizer.transform(X)

            Xs.append(X)
            masks.append(np.array(mask))
            ys.append(np.array(y))
            names.append(name)
            ts.append(cur_ts)

            assert np.sum(mask) > 0
            assert len(X) == len(mask) and len(X) == len(y)

        self.data = [[Xs, masks], ys]
        self.names = names
        self.ts = ts

    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[[None]*N, [None]*N], [None]*N]
                tmp_names = [None] * N
                tmp_ts = [None] * N
                for i in range(N):
                    tmp_data[0][0][i] = self.data[0][0][order[i]]
                    tmp_data[0][1][i] = self.data[0][1][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_names[i] = self.names[order[i]]
                    tmp_ts[i] = self.ts[order[i]]
                self.data = tmp_data
                self.names = tmp_names
                self.ts = tmp_ts
            else:
                # sort entirely
                Xs = self.data[0][0]
                masks = self.data[0][1]
                ys = self.data[1]
                (Xs, masks, ys, self.names, self.ts) = utils.sort_and_shuffle([Xs, masks, ys,
                                                                                      self.names, self.ts], B)
                self.data = [[Xs, masks], ys]

            for i in range(0, len(self.data[1]), B):
                X = self.data[0][0][i:i+B]
                mask = self.data[0][1][i:i+B]
                y = self.data[1][i:i+B]
                names = self.names[i:i+B]
                ts = self.ts[i:i+B]

                y_true = [np.array(x) for x in y]
                y_true = utils.pad_zeros(y_true)
                y_true = np.expand_dims(y_true, axis=-1)

                if self.partition == 'log':
                    y = [np.array([metrics.get_bin_log(x, 10) for x in z]) for z in y]
                if self.partition == 'custom':
                    y = [np.array([metrics.get_bin_custom(x, 10) for x in z]) for z in y]

                X = utils.pad_zeros(X)  # (B, T, D)
                mask = utils.pad_zeros(mask)  # (B, T)
                y = utils.pad_zeros(y)
                y = np.expand_dims(y, axis=-1)

                if self.return_y_true:
                    batch_data = ([X, mask], y, y_true)
                else:
                    batch_data = ([X, mask], y)

                if not self.return_names:
                    yield batch_data
                else:
                    yield {"data": batch_data, "names": names, "ts": ts}

    def __iter__(self):
        return self.generator

    def next(self, return_y_true=False):
        with self.lock:
            self.return_y_true = return_y_true
            return next(self.generator)

    def __next__(self):
        return self.next()


def save_results(names, ts, pred, y_true, path):
    utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,period_length,prediction,y_true\n")
        for (name, t, x, y) in zip(names, ts, pred, y_true):
            f.write("{},{:.6f},{:.6f},{:.6f}\n".format(name, t, x, y))
