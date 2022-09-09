import numpy as np
from utils import onehot, unhot, is_sparse

class Dataset:

    """
    Class for Storing Datasets
    """

    def __init__(self, inputs, labels, batch_size=32, drop_last=True):
        # set inputs
        self.inputs = np.array(inputs, dtype=np.float64)
        self.labels = np.array(labels, dtype=np.int64)
        # force errors for invalid
        if self.inputs.ndim < 2:
            raise ValueError("inputs must be at least a 2D array.")
        if self.labels.ndim > 2:
            raise ValueError("y must be a 1D or 2D array.")
        # make sparse labels 1d array
        if is_sparse(self.labels) and self.labels.ndim != 1:
            self.labels = np.squeeze(self.labels, axis=-1)
        # set dataset info
        self.classes = np.unique(self.labels) if is_sparse(self.labels) \
                        else np.unique(unhot(self.labels))
        self.size = len(self.inputs)
        # create the dataset
        self.dataset = self.create_dataset(batch_size, drop_last)

    def create_dataset(self, batch_size, drop_last):
        # num of batches & sample shape
        m = self.size // batch_size
        inshape = list(self.inputs[0].shape)
        lshape = list(self.labels[0].shape) if self.labels.ndim > 1 else None

        # evenly divisible batch size
        if self.size % batch_size == 0:
            inputs = self.inputs.reshape(m, batch_size, *inshape)
            # one-hot labels
            if lshape:
                labels = self.labels.reshape(m, batch_size, *lshape)
            # sparse labels
            else:
                labels = self.labels.reshape(m, batch_size, )
            dataset = [(data, label) for data, label in zip(inputs, labels)]
        # non evenly divisble batch size
        else:
            inputs = self.inputs[:m * batch_size].reshape(m, batch_size, *inshape)
            # one-hot labels
            if lshape:
                labels = self.labels[:m * batch_size].reshape(m, batch_size, *lshape)
            # sparse labels
            else:
                labels = self.labels[:m * batch_size].reshape(m, batch_size, )
            dataset = [(data, label) for data, label in zip(inputs, labels)]
            # add what's left over
            if not drop_last:
                inputs = self.inputs[m * batch_size:].reshape(1, -1, *inshape)
                # one-hot labels
                if lshape:
                    labels = self.labels[m * batch_size:].reshape(1, -1, *lshape)
                # sparse labels
                else:
                    labels = self.labels[m * batch_size:].reshape(1, -1, )
                dataset.extend([(data, label) for data, label in zip(inputs, labels)])

        return dataset

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return f"Inputs:\n{self.inputs}\nLabels:\n{self.labels}"

    def __getitem__(self, index):
        return self.dataset[index]

if __name__ == "__main__":

    # TESTING DATASET
    inputs = np.random.randint(0, 10, (1000, 27, 27, 3))
    labels = np.random.choice(5, (1000, 1))

    # (DROP LAST TRUE)
    dataset = Dataset(inputs, labels, batch_size=32, drop_last=True)
    first_batch = dataset[0]
    x, y = first_batch
    print(x.shape, y.shape)
    print(y[0])
    last_batch = dataset[-1]
    x, y = last_batch
    print(x.shape, y.shape)

    print()

    # (DROP LAST FALSE)
    dataset = Dataset(inputs, labels, batch_size=32, drop_last=False)
    first_batch = dataset[0]
    x, y = first_batch
    print(x.shape, y.shape)
    print(y[0])
    last_batch = dataset[-1]
    x, y = last_batch
    print(x.shape, y.shape)

    print()

    # (ONEHOT ENCODED)
    dataset = Dataset(inputs, onehot(labels), drop_last=False)
    first_batch = dataset[0]
    x, y = first_batch
    print(x.shape, y.shape)
    print(y[0])
    last_batch = dataset[-1]
    x, y = last_batch
    print(x.shape, y.shape)
    
    

    