import pickle
from argparse import ArgumentParser
from pathlib import Path
from inlp.src.inlp_oop import inlp, inlp_dataset_handler, inlp_linear_model
import torch
from sklearn.linear_model import LogisticRegression
import sys
from data_utils import load_domain_data

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    parser.add_argument('-task', type=str)
    parser.add_argument('-domain', type=str)
    parser.add_argument('-iter', type=int, help='num of iterations inlp was run')
    args = parser.parse_args()
    task = args.task
    test_domain = args.domain
    num_iter = args.iter
    ft = True
    binary = True
    pkl_path = Path('piruda-models', task, test_domain, 'ft') if ft else Path('piruda-models', task)
    X_train, Y_train, X_dev, Y_dev, majority = load_domain_data(task, test_domain, ft, binary)
    res_file_dir = Path('results', task, test_domain, 'ft') if ft else Path('piruda-models', task)
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    binary_str = 'one_vs_all_' if binary else ''
    res_file_name = f'inlp_{binary_str}{num_iter}_iter.txt'
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        inlp_dataset = inlp_dataset_handler.ClassificationDatasetHandler(X_train, Y_train, X_dev, Y_dev)
        inlp_model_handler = inlp_linear_model.SKlearnClassifier(LogisticRegression, {'max_iter': 1000})
        P, rowspace_projections, Ws = inlp.run_INLP(num_classifiers=num_iter, input_dim=X_train.shape[1],
                                                    is_autoregressive=True,
                                                    min_accuracy=majority + 0.01, dataset_handler=inlp_dataset,
                                                    model=inlp_model_handler)
        with open(Path(pkl_path, test_domain, f'projection_{binary_str}{num_iter}_iter.pkl'), 'wb+') as f:
            pickle.dump(P, f)
