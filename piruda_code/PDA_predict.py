import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from data_utils import load_task_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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
    X_train_domain, X_train_task, Y_train_domain, Y_train_task, X_dev_domain, X_dev_task, Y_dev_domain, Y_dev_task, \
    X_test, Y_test = load_task_data(task, test_domain, ft, binary)
    binary_str = '_one_vs_all' if binary else ''
    with open(Path(pkl_path, test_domain, f'neurons_to_ablate{binary_str}'), 'rb') as f:
        neurons_to_ablate = pickle.load(f)
    res_file_dir = Path('results', task, test_domain, 'ft') if ft else Path('piruda-models', task)
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    res_file_name = f'ablation_task_results{binary_str}.txt'
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f
        print('test domain: ', test_domain)
        cls = LogisticRegression(max_iter=1000)
        cls.fit(X_train_domain, Y_train_domain)
        print('before ablation - domain train acc:')
        print(cls.score(X_train_domain, Y_train_domain))
        print('before ablation - domain dev acc:')
        print(cls.score(X_dev_domain, Y_dev_domain))
        X_train_domain[:, neurons_to_ablate] = 0.
        X_dev_domain[:, neurons_to_ablate] = 0.
        cls = LogisticRegression(max_iter=1000)
        cls.fit(X_train_domain, Y_train_domain)
        print('after ablation - domain train acc:')
        print(cls.score(X_train_domain, Y_train_domain))
        print('after ablation - domain dev acc:')
        print(cls.score(X_dev_domain, Y_dev_domain))
        cls = LogisticRegression(max_iter=1000)
        cls.fit(X_train_task, Y_train_task)
        if task == 'sentiment':
            print('before ablation - task train acc:')
            print(cls.score(X_train_task, Y_train_task))
            print('before ablation - task in-domain dev acc:')
            print(cls.score(X_dev_task, Y_dev_task))
            print('before ablation - task o.o.d test acc:')
            print(cls.score(X_test, Y_test))
            X_train_task[:, neurons_to_ablate] = 0.
            X_dev_task[:, neurons_to_ablate] = 0.
            X_test[:, neurons_to_ablate] = 0.
            cls = LogisticRegression(max_iter=1000)
            cls.fit(X_train_task, Y_train_task)
            print('after ablation - task train acc:')
            print(cls.score(X_train_task, Y_train_task))
            print('after ablation - task in-domain dev acc:')
            print(cls.score(X_dev_task, Y_dev_task))
            print('after ablation - task o.o.d test acc:')
            print(cls.score(X_test, Y_test))
        else:
            print('before ablation - task train f1:')
            train_preds = cls.predict(X_train_task)
            print(f1_score(Y_train_task, train_preds, average='macro' if task == 'mnli' else 'binary'))
            print('before ablation - task dev f1:')
            dev_preds = cls.predict(X_dev_task)
            print(f1_score(Y_dev_task, dev_preds, average='macro' if task == 'mnli' else 'binary'))
            print('before ablation - task o.o.d test f1:')
            test_preds = cls.predict(X_test)
            print(f1_score(Y_test, test_preds, average='macro' if task == 'mnli' else 'binary'))
            X_train_task[:, neurons_to_ablate] = 0.
            X_dev_task[:, neurons_to_ablate] = 0.
            cls = LogisticRegression(max_iter=1000)
            cls.fit(X_train_task, Y_train_task)
            print('after ablation - task train f1:')
            train_preds = cls.predict(X_train_task)
            print(f1_score(Y_train_task, train_preds, average='macro' if task == 'mnli' else 'binary'))
            print('after ablation - task in-domain dev f1:')
            dev_preds = cls.predict(X_dev_task)
            print(f1_score(Y_dev_task, dev_preds, average='macro' if task == 'mnli' else 'binary'))
            print('after ablation - task o.o.d f1:')
            test_preds = cls.predict(X_test)
            print(f1_score(Y_test, test_preds, average='macro' if task == 'mnli' else 'binary'))
