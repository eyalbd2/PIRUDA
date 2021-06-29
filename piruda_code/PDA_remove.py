import torch
import logging
from tqdm import tqdm
from data_utils import load_domain_data
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from argparse import ArgumentParser
import sys
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np


def get_top_neurons_per_tag(weights, p):
    top_neurons_per_label = {}
    theta = weights
    num_of_neurons_per_label = torch.ceil(torch.tensor(theta.shape[1] * p / 100)).int().item()
    for l in range(theta.shape[0]):
        weights_for_label = theta[l]
        sorted_neurons_for_label = np.argsort(-np.abs(weights_for_label))
        top_neurons_per_label[l] = sorted_neurons_for_label[:num_of_neurons_per_label]
    return top_neurons_per_label


def get_top_neurons(weights):
    ordered_neurons = []
    for p in np.logspace(-1, np.log10(100), 100):
        tnpt = get_top_neurons_per_tag(weights, p)
        top_neurons = [neuron.item() for label, neurons in tnpt.items() for neuron in neurons]
        new_neurons = np.setdiff1d(top_neurons, ordered_neurons)
        ordered_neurons += new_neurons.tolist()
    return ordered_neurons


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-task', type=str)
    parser.add_argument('-domain', type=str)
    args = parser.parse_args()
    task = args.task
    test_domain = args.domain
    ft = True
    binary = True
    pkl_path = Path('piruda-models', task, test_domain, 'ft') if ft else Path('piruda-models', task)
    X_train, Y_train, X_dev, Y_dev, majority = load_domain_data(task, test_domain, ft, binary)
    res_file_dir = Path('results', task, test_domain, 'ft') if ft else Path('piruda-models', task)
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    binary_str = '_one_vs_all' if binary else ''
    res_file_name = f'domainAblation{binary_str}.txt'
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        print('test domain: ', test_domain)
        print('creating train and test datasets')
        cls = LogisticRegression(max_iter=1000)
        cls.fit(X_train, Y_train)
        # cls.predict(X_test)
        print('train acc:')
        print(cls.score(X_train, Y_train))
        print('test acc:')
        print(cls.score(X_dev, Y_dev))
        weights = cls.coef_
        ranking = get_top_neurons(weights)
        print(f'ranking: {ranking}')
        neurons_to_ablate = []
        step = 5
        for i in tqdm(range(0, len(ranking), step)):
            neurons_to_ablate.extend(ranking[i - step:i] if i > 0 else [])
            print(f'ablating {i} neurons:')
            print(neurons_to_ablate)
            X_train[:, neurons_to_ablate] = 0.
            cls = LogisticRegression(max_iter=1000)
            cls.fit(X_train, Y_train)
            print('train acc:')
            print(cls.score(X_train, Y_train))
            print('test acc:')
            test_acc = cls.score(X_dev, Y_dev)
            print(test_acc)
            if test_acc < majority + 0.01:
                print('below majority')
                with open(Path(pkl_path, test_domain, f'neurons_to_ablate{binary_str}'), 'wb+') as f:
                    pickle.dump(neurons_to_ablate, f)
                break
