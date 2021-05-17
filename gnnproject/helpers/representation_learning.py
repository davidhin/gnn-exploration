import copy
import json
import pickle as pkl
import sys

import numpy
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as pr
from sklearn.metrics import recall_score as rc
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class DataEntry:
    def __init__(self, dataset, feature_repr, label, meta_data=None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet)
        self.features = copy.deepcopy(feature_repr)
        self.label = label
        self.meta_data = meta_data
        pass

    def __repr__(self):
        return str(self.features) + "\t" + str(self.label)

    def __hash__(self):
        return str(self.features).__hash__

    def is_positive(self):
        return self.label == 1


class DataSet:
    def __init__(self, batch_size, hdim):
        self.train_entries = []
        self.valid_entries = []
        self.test_entries = []
        self.train_batch_indices = []
        self.valid_batch_indices = []
        self.test_batch_indices = []
        self.batch_size = batch_size
        self.hdim = hdim
        self.positive_indices_in_train = []
        self.negative_indices_in_train = []

    def initialize_dataset(self, balance=True, output_buffer=sys.stderr):
        if isinstance(balance, bool) and balance:
            entries = []
            train_features = []
            train_targets = []
            for entry in self.train_entries:
                train_features.append(entry.features)
                train_targets.append(entry.label)
            train_features = np.array(train_features)
            train_targets = np.array(train_targets)
            smote = SMOTE(random_state=1000)
            features, targets = smote.fit_resample(train_features, train_targets)
            for feature, target in zip(features, targets):
                entries.append(DataEntry(self, feature.tolist(), target.item()))
            self.train_entries = entries
        elif isinstance(balance, list) and len(balance) == 2:
            entries = []
            for entry in self.train_entries:
                if entry.is_positive():
                    for _ in range(balance[0]):
                        entries.append(
                            DataEntry(
                                self, entry.features, entry.label, entry.meta_data
                            )
                        )
                else:
                    if np.random.uniform() <= balance[1]:
                        entries.append(
                            DataEntry(
                                self, entry.features, entry.label, entry.meta_data
                            )
                        )
            self.train_entries = entries
            pass
        for tidx, entry in enumerate(self.train_entries):
            if entry.label == 1:
                self.positive_indices_in_train.append(tidx)
            else:
                self.negative_indices_in_train.append(tidx)
        self.initialize_train_batches()
        if output_buffer is not None:
            print(
                "Number of Train Entries %d #Batches %d"
                % (len(self.train_entries), len(self.train_batch_indices)),
                file=output_buffer,
            )
        self.initialize_valid_batches()
        if output_buffer is not None:
            print(
                "Number of Valid Entries %d #Batches %d"
                % (len(self.valid_entries), len(self.valid_batch_indices)),
                file=output_buffer,
            )
        self.initialize_test_batches()
        if output_buffer is not None:
            print(
                "Number of Test  Entries %d #Batches %d"
                % (len(self.test_entries), len(self.test_batch_indices)),
                file=output_buffer,
            )

    def add_data_entry(self, feature, label, part="train"):
        assert part in ["train", "valid", "test"]
        entry = DataEntry(self, feature, label)
        if part == "train":
            self.train_entries.append(entry)
        elif part == "valid":
            self.valid_entries.append(entry)
        else:
            self.test_entries.append(entry)

    def initialize_train_batches(self):
        self.train_batch_indices = self.create_batches(
            self.batch_size, self.train_entries
        )
        return len(self.train_batch_indices)
        pass

    def clear_test_set(self):
        self.test_entries = []

    def initialize_valid_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batch_indices = self.create_batches(batch_size, self.valid_entries)
        return len(self.valid_batch_indices)
        pass

    def initialize_test_batches(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batch_indices = self.create_batches(batch_size, self.test_entries)
        return len(self.test_batch_indices)
        pass

    def get_next_train_batch(self):
        if len(self.train_batch_indices) > 0:
            indices = self.train_batch_indices.pop()
            features, targets = self.prepare_data(self.train_entries, indices)
            same_class_features = self.find_same_class_data(ignore_indices=indices)
            different_class_features = self.find_different_class_data(
                ignore_indices=indices
            )
            return features, targets, same_class_features, different_class_features
        raise ValueError(
            "Initialize Train Batch First by calling dataset.initialize_train_batches()"
        )
        pass

    def get_next_valid_batch(self):
        if len(self.valid_batch_indices) > 0:
            indices = self.valid_batch_indices.pop()
            return self.prepare_data(self.valid_entries, indices)
        raise ValueError(
            "Initialize Valid Batch First by calling dataset.initialize_valid_batches()"
        )
        pass

    def get_next_test_batch(self):
        if len(self.test_batch_indices) > 0:
            indices = self.test_batch_indices.pop()
            return self.prepare_data(self.test_entries, indices)
        raise ValueError(
            "Initialize Test Batch First by calling dataset.initialize_test_batches()"
        )
        pass

    def create_batches(self, batch_size, entries):
        _batches = []
        if batch_size == -1:
            batch_size = self.batch_size
        total = len(entries)
        indices = np.arange(0, total - 1, 1)
        np.random.shuffle(indices)
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + batch_size
            if c_end > end:
                c_end = end
            _batches.append(indices[curr:c_end])
            curr = c_end
        return _batches

    def prepare_data(self, _entries, indices):
        batch_size = len(indices)
        features = np.zeros(shape=(batch_size, self.hdim))
        targets = np.zeros(shape=(batch_size))
        for tidx, idx in enumerate(indices):
            entry = _entries[idx]
            assert isinstance(entry, DataEntry)
            targets[tidx] = entry.label
            for feature_idx in range(self.hdim):
                features[tidx, feature_idx] = entry.features[feature_idx]
        return torch.FloatTensor(features), torch.LongTensor(targets)
        pass

    def find_same_class_data(self, ignore_indices):
        positive_indices_pool = set(self.positive_indices_in_train).difference(
            ignore_indices
        )
        negative_indices_pool = set(self.negative_indices_in_train).difference(
            ignore_indices
        )
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool
        )

    def find_different_class_data(self, ignore_indices):
        positive_indices_pool = set(self.negative_indices_in_train).difference(
            ignore_indices
        )
        negative_indices_pool = set(self.positive_indices_in_train).difference(
            ignore_indices
        )
        return self.find_triplet_loss_data(
            ignore_indices, negative_indices_pool, positive_indices_pool
        )

    def find_triplet_loss_data(
        self, ignore_indices, negative_indices_pool, positive_indices_pool
    ):
        indices = []
        for eidx in ignore_indices:
            if self.train_entries[eidx].is_positive():
                indices_pool = positive_indices_pool
            else:
                indices_pool = negative_indices_pool
            indices_pool = list(indices_pool)
            indices.append(np.random.choice(indices_pool))
        features, _ = self.prepare_data(self.train_entries, indices)
        return features


def create_dataset(
    train_file, valid_file=None, test_file=None, batch_size=32, output_buffer=sys.stderr
):
    if output_buffer is not None:
        print("Reading Train data from %s" % train_file, file=output_buffer)
    train_data = json.load(open(train_file))
    # "target": 1, "graph_feature"
    hdim = len(train_data[0]["graph_feature"])
    dataset = DataSet(batch_size=batch_size, hdim=hdim)
    for data in train_data:
        dataset.add_data_entry(
            data["graph_feature"], min(int(data["target"]), 1), part="train"
        )
    if valid_file is not None:
        if output_buffer is not None:
            print("Reading Valid data from %s" % valid_file, file=output_buffer)
        valid_data = json.load(open(valid_file))
        for data in valid_data:
            dataset.add_data_entry(
                data["graph_feature"], min(int(data["target"]), 1), part="valid"
            )
    if test_file is not None:
        if output_buffer is not None:
            print("Reading Test data from %s" % test_file, file=output_buffer)
        test_data = json.load(open(test_file))
        for data in test_data:
            dataset.add_data_entry(
                data["graph_feature"], min(int(data["target"]), 1), part="test"
            )
    # dataset.initialize_dataset()
    return dataset


class MetricLearningModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout_p=0.2,
        aplha=0.5,
        lambda1=0.5,
        lambda2=0.001,
        num_layers=1,
    ):
        super(MetricLearningModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.dropout_p = dropout_p
        self.alpha = aplha
        self.layer1 = nn.Sequential(
            nn.Linear(
                in_features=self.input_dim, out_features=self.hidden_dim, bias=True
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        )
        self.feature = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=self.hidden_dim,
                        out_features=self.internal_dim,
                        bias=True,
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_p),
                    nn.Linear(
                        in_features=self.internal_dim,
                        out_features=self.hidden_dim,
                        bias=True,
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_p),
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
            nn.LogSoftmax(dim=-1),
        )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss_function = nn.NLLLoss(reduction="none")
        # print(self.alpha, self.lambda1, self.lambda2, sep='\t', end='\t')

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def forward(
        self, example_batch, targets=None, positive_batch=None, negative_batch=None
    ):
        train_mode = (
            positive_batch is not None
            and negative_batch is not None
            and targets is not None
        )
        h_a = self.extract_feature(example_batch)
        y_a = self.classifier(h_a)
        probs = torch.exp(y_a)
        batch_loss = None
        if targets is not None:
            ce_loss = self.loss_function(input=y_a, target=targets)
            batch_loss = ce_loss.sum(dim=-1)
        if train_mode:
            h_p = self.extract_feature(positive_batch)
            h_n = self.extract_feature(negative_batch)
            dot_p = (
                h_a.unsqueeze(dim=1).bmm(h_p.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            )
            dot_n = (
                h_a.unsqueeze(dim=1).bmm(h_n.unsqueeze(dim=-1)).squeeze(-1).squeeze(-1)
            )
            mag_a = torch.norm(h_a, dim=-1)
            mag_p = torch.norm(h_p, dim=-1)
            mag_n = torch.norm(h_n, dim=-1)
            D_plus = 1 - (dot_p / (mag_a * mag_p))
            D_minus = 1 - (dot_n / (mag_a * mag_n))
            trip_loss = self.lambda1 * torch.abs((D_plus - D_minus + self.alpha))
            ce_loss = self.loss_function(input=y_a, target=targets)
            l2_loss = self.lambda2 * (mag_a + mag_p + mag_n)
            total_loss = ce_loss + trip_loss + l2_loss
            batch_loss = (total_loss).sum(dim=-1)
        return probs, h_a, batch_loss
        pass


class RepresentationLearningModel(BaseEstimator):
    def __init__(
        self,
        alpha=0.5,
        lambda1=0.5,
        lambda2=0.001,
        hidden_dim=256,  # Model Parameters
        dropout=0.2,
        batch_size=64,
        balance=True,  # Model Parameters
        num_epoch=100,
        max_patience=20,  # Training Parameters
        print=False,
        num_layers=1,
    ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dropout = dropout
        self.num_epoch = num_epoch
        self.max_patience = max_patience
        self.batch_size = batch_size
        self.balance = balance
        self.cuda = torch.cuda.is_available()
        self.print = print
        self.num_layers = num_layers
        if print:
            self.output_buffer = sys.stderr
        else:
            self.output_buffer = None
        pass

    def fit(self, train_x, train_y):
        self.train(train_x, train_y)

    def train(self, train_x, train_y):
        input_dim = train_x.shape[1]
        self.model = MetricLearningModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            aplha=self.alpha,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            dropout_p=self.dropout,
            num_layers=self.num_layers,
        )
        self.optimizer = Adam(self.model.parameters())
        if self.cuda:
            self.model.cuda(device=0)
        self.dataset = DataSet(self.batch_size, train_x.shape[1])
        for _x, _y in zip(train_x, train_y):
            if numpy.random.uniform() <= 0.1:
                self.dataset.add_data_entry(_x.tolist(), _y.item(), "valid")
            else:
                self.dataset.add_data_entry(_x.tolist(), _y.item(), "train")
        self.dataset.initialize_dataset(
            balance=self.balance, output_buffer=self.output_buffer
        )
        train(
            model=self.model,
            dataset=self.dataset,
            optimizer=self.optimizer,
            num_epochs=self.num_epoch,
            max_patience=self.max_patience,
            cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer,
        )
        if self.output_buffer is not None:
            print("Training Complete", file=self.output_buffer)

    def predict(self, text_x):
        if not hasattr(self, "dataset"):
            raise ValueError(
                "Cannnot call predict or evaluate in untrained model. Train First!"
            )
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part="test")
        return predict(
            model=self.model,
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(),
            cuda_device=0 if self.cuda else -1,
        )

    def predict_proba(self, text_x):
        if not hasattr(self, "dataset"):
            raise ValueError(
                "Cannnot call predict or evaluate in untrained model. Train First!"
            )
        self.dataset.clear_test_set()
        for _x in text_x:
            self.dataset.add_data_entry(_x.tolist(), 0, part="test")
        return predict_proba(
            model=self.model,
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(),
            cuda_device=0 if self.cuda else -1,
        )

    def evaluate(self, text_x, test_y):
        if not hasattr(self, "dataset"):
            raise ValueError(
                "Cannnot call predict or evaluate in untrained model. Train First!"
            )
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part="test")
        acc, pr, rc, f1 = evaluate_from_model(
            model=self.model,
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(),
            cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer,
        )
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

    def score(self, text_x, test_y):
        if not hasattr(self, "dataset"):
            raise ValueError(
                "Cannnot call predict or evaluate in untrained model. Train First!"
            )
        self.dataset.clear_test_set()
        for _x, _y in zip(text_x, test_y):
            self.dataset.add_data_entry(_x.tolist(), _y.item(), part="test")
        _, _, _, f1 = evaluate_from_model(
            model=self.model,
            iterator_function=self.dataset.get_next_test_batch,
            _batch_count=self.dataset.initialize_test_batches(),
            cuda_device=0 if self.cuda else -1,
            output_buffer=self.output_buffer,
        )
        return f1


def train(
    model,
    dataset,
    optimizer,
    num_epochs,
    max_patience=5,
    valid_every=1,
    cuda_device=-1,
    output_buffer=sys.stderr,
):
    if output_buffer is not None:
        print("Start Training", file=output_buffer)
    assert isinstance(model, MetricLearningModel) and isinstance(dataset, DataSet)
    best_f1 = 0
    best_model = None
    patience_counter = 0
    train_losses = []
    try:
        for epoch_count in range(num_epochs):
            batch_losses = []
            num_batches = dataset.initialize_train_batches()
            output_batches_generator = range(num_batches)
            if output_buffer is not None:
                output_batches_generator = tqdm(output_batches_generator)
            for _ in output_batches_generator:
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                (
                    features,
                    targets,
                    same_class_features,
                    diff_class_features,
                ) = dataset.get_next_train_batch()
                if cuda_device != -1:
                    features = features.cuda(device=cuda_device)
                    targets = targets.cuda(device=cuda_device)
                    same_class_features = same_class_features.cuda(device=cuda_device)
                    diff_class_features = diff_class_features.cuda(device=cuda_device)
                probabilities, representation, batch_loss = model(
                    example_batch=features,
                    targets=targets,
                    positive_batch=same_class_features,
                    negative_batch=diff_class_features,
                )
                batch_losses.append(batch_loss.detach().cpu().item())
                batch_loss.backward()
                optimizer.step()
            epoch_loss = np.sum(batch_losses).item()
            train_losses.append(epoch_loss)
            if output_buffer is not None:
                print("=" * 100, file=output_buffer)
                print(
                    "After epoch %2d Train loss : %10.4f" % (epoch_count, epoch_loss),
                    file=output_buffer,
                )
                print("=" * 100, file=output_buffer)
            if epoch_count % valid_every == 0:
                valid_batch_count = dataset.initialize_valid_batches()
                vacc, vpr, vrc, vf1 = evaluate_from_model(
                    model,
                    dataset.get_next_valid_batch,
                    valid_batch_count,
                    cuda_device,
                    output_buffer,
                )
                if vf1 > best_f1:
                    best_f1 = vf1
                    patience_counter = 0
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if dataset.initialize_test_batches() != 0:
                    tacc, tpr, trc, tf1 = evaluate_from_model(
                        model,
                        dataset.get_next_test_batch,
                        dataset.initialize_test_batches(),
                        cuda_device,
                        output_buffer=output_buffer,
                    )
                    if output_buffer is not None:
                        print(
                            "Test Set:       Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f"
                            % (tacc, tpr, trc, tf1),
                            file=output_buffer,
                        )
                        print("=" * 100, file=output_buffer)
                if output_buffer is not None:
                    print(
                        "Validation Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f\tPatience: %2d"
                        % (vacc, vpr, vrc, vf1, patience_counter),
                        file=output_buffer,
                    )
                    print("-" * 100, file=output_buffer)
                if patience_counter == max_patience:
                    if best_model is not None:
                        model.load_state_dict(best_model)
                        if cuda_device != -1:
                            model.cuda(device=cuda_device)
                    break
    except KeyboardInterrupt:
        if output_buffer is not None:
            print("Training Interrupted by User!")
        if best_model is not None:
            model.load_state_dict(best_model)
            if cuda_device != -1:
                model.cuda(device=cuda_device)
    if dataset.initialize_test_batches() != 0:
        tacc, tpr, trc, tf1 = evaluate_from_model(
            model,
            dataset.get_next_test_batch,
            dataset.initialize_test_batches(),
            cuda_device,
        )
        if output_buffer is not None:
            print("*" * 100, file=output_buffer)
            print("*" * 100, file=output_buffer)
            print(
                "Test Set: Acc: %6.3f\tPr: %6.3f\tRc %6.3f\tF1: %6.3f"
                % (tacc, tpr, trc, tf1),
                file=output_buffer,
            )
            print("%f\t%f\t%f\t%f" % (tacc, tpr, trc, tf1))
            print("*" * 100, file=output_buffer)
            print("*" * 100, file=output_buffer)


def predict(model, iterator_function, _batch_count, cuda_device):
    probs = predict_proba(model, iterator_function, _batch_count, cuda_device)
    return np.argmax(probs, axis=-1)


def predict_proba(model, iterator_function, _batch_count, cuda_device):
    model.eval()
    with torch.no_grad():
        predictions = []
        for _ in tqdm(range(_batch_count)):
            features, targets = iterator_function()
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            probs, _, _ = model(example_batch=features)
            predictions.extend(probs)
        model.train()
    return np.array(predictions)


def evaluate_from_model(
    model, iterator_function, _batch_count, cuda_device, output_buffer=sys.stderr
):
    if output_buffer is not None:
        print(_batch_count, file=output_buffer)
    model.eval()
    with torch.no_grad():
        predictions = []
        expectations = []
        batch_generator = range(_batch_count)
        if output_buffer is not None:
            batch_generator = tqdm(batch_generator)
        for _ in batch_generator:
            features, targets = iterator_function()
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            probs, _, _ = model(example_batch=features)
            batch_pred = np.argmax(probs.detach().cpu().numpy(), axis=-1).tolist()
            batch_tgt = targets.detach().cpu().numpy().tolist()
            predictions.extend(batch_pred)
            expectations.extend(batch_tgt)
        model.train()
        return (
            acc(expectations, predictions) * 100,
            pr(expectations, predictions) * 100,
            rc(expectations, predictions) * 100,
            f1(expectations, predictions) * 100,
        )


def show_representation(
    model, iterator_function, _batch_count, cuda_device, name, output_buffer=sys.stderr
):
    model.eval()
    with torch.no_grad():
        representations = []
        expected_targets = []
        batch_generator = range(_batch_count)
        if output_buffer is not None:
            batch_generator = tqdm(batch_generator)
        for _ in batch_generator:
            iterator_values = iterator_function()
            features, targets = iterator_values[0], iterator_values[1]
            if cuda_device != -1:
                features = features.cuda(device=cuda_device)
            _, repr, _ = model(example_batch=features)
            repr = repr.detach().cpu().numpy()
            print(repr.shape)
            representations.extend(repr.tolist())
            expected_targets.extend(targets.numpy().tolist())
        model.train()
        print(np.array(representations).shape)
        print(np.array(expected_targets).shape)


def representation_learning(train_pickle_path, test_pickle_path, no_ggnn=False):
    """Run Representation Learning module from ReVeal."""
    with open(
        train_pickle_path,
        "rb",
    ) as f:
        X_train, y_train = zip(*pkl.load(f))
        train_X = np.array([i[0] for i in X_train])
        train_Y = np.array(y_train)

    with open(
        test_pickle_path,
        "rb",
    ) as f:
        X_test, y_test = zip(*pkl.load(f))
        test_X = np.array([i[0] for i in X_test])
        test_Y = np.array(y_test)

    if no_ggnn:
        train_X = np.array(X_train)
        test_X = np.array(X_test)

    print(
        train_X.shape,
        train_Y.shape,
        test_X.shape,
        test_Y.shape,
        sep="\t",
        file=sys.stderr,
        flush=True,
    )

    model = RepresentationLearningModel(
        lambda1=0.5,
        lambda2=0.001,
        batch_size=64,
        print=True,
        max_patience=5,
        balance=True,
        num_layers=1,
    )

    model.train(train_X, train_Y)
    results_train = model.evaluate(train_X, train_Y)
    results_test = model.evaluate(test_X, test_Y)
    return results_train, results_test
