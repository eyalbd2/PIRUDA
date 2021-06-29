from argparse import ArgumentParser
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification
import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from data_utils import data_dict, data_dirs, aspect_label_mapping


class customDataset(Dataset):
    def __init__(self, encodings, labels, task):
        self.encodings = encodings
        self.label = labels
        self.task = task

    def __getitem__(self, idx):
        item = self.encodings[idx]
        if self.task != 'aspect':
            item['label'] = torch.tensor(self.label[idx])
        else:
            item['label_ids'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if predictions.ndim > 1:
        predictions = np.where(predictions.flatten() == 2, 1, predictions.flatten())[labels.flatten() != -100]
        labels = np.where(labels.flatten() == 2, 1, labels.flatten())[labels.flatten() != -100]
    acc = accuracy_score(y_true=labels, y_pred=predictions)
    binary = True if len(set(labels)) == 2 else False
    if binary:
        f1 = f1_score(y_true=labels, y_pred=predictions)
    else:
        f1 = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-task', type=str)
    argparser.add_argument('-domain', type=str)
    args = argparser.parse_args()
    task = args.task
    data = data_dict[task]
    test_domain = args.domain
    print(f'domain: {test_domain}')
    pkl_path = Path('piruda-models', task)
    data_dir = data_dirs[task]
    all_domains = [d.name for d in Path('data', data_dir).glob('*') if d.is_dir()]
    train_domains = [d for d in all_domains if d != test_domain]
    train_data, dev_data = ([], []), ([], [])
    for d in train_domains:
        train_path = data['train_paths'][d]
        dev_path = data['dev_paths'][d]
        test_path = data['test_paths'][d]
        with open(train_path, 'rb') as f:
            curr_train_data = pickle.load(f)
        with open(dev_path, 'rb') as f:
            curr_dev_data = pickle.load(f)
        train_data[0].extend(curr_train_data[0])
        train_data[1].extend(curr_train_data[1])
        dev_data[0].extend(curr_dev_data[0])
        dev_data[1].extend(curr_dev_data[1])
    num_labels = len(set(train_data[1])) if task != 'aspect' else \
        len(set([label for labels in train_data[1] for label in labels]))
    pickles_root = Path('piruda-models', task, test_domain, 'ft_models')
    if not pickles_root.exists():
        pickles_root.mkdir(parents=True)
    bert_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    datas = [train_data, dev_data]
    for i, data in enumerate(datas):
        filtered_sents, filtered_labels = {}, {}
        example_idx = 0
        for sentence, label in zip(data[0], data[1]):
            if type(sentence) == tuple:
                if sentence[0][-1] == '.':
                    sentence = ' '.join(sentence)
                else:
                    sentence = '. '.join(sentence)
            elif type(sentence) == list:
                new_labels = [-100]  # ignore index for CLS token
                for word, l in zip(sentence, label):
                    word_labels = [aspect_label_mapping[l]]
                    subtokens = tokenizer(word).data['input_ids']
                    if len(subtokens) > 3:
                        # word got split - label sub-tokens with ignore index
                        word_labels.extend([-100] * (len(subtokens) - 3))
                    new_labels.extend(word_labels)
                sentence = ' '.join(sentence)
                new_labels.extend([-100] * (128 - len(new_labels)))
            filtered_sents[example_idx] = tokenizer(sentence, padding='max_length', truncation=True,
                                                    max_length=128).data
            filtered_labels[example_idx] = label if task != 'aspect' else new_labels
            example_idx += 1
        datas[i] = filtered_sents, filtered_labels
    train_data, dev_data = tuple(datas)
    train_data = customDataset(train_data[0], train_data[1], task)
    dev_data = customDataset(dev_data[0], dev_data[1], task)

    # Setup BERT
    if task == 'aspect':
        model = AutoModelForTokenClassification.from_pretrained(bert_name, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels)
    train_args = TrainingArguments(pickles_root.__str__(), evaluation_strategy='epoch', per_device_train_batch_size=50,
                                   save_strategy='epoch')
    trainer = Trainer(model=model, args=train_args, train_dataset=train_data,
                      eval_dataset=dev_data, compute_metrics=compute_metrics)
    trainer.train()
    print('finished')
    trainer.evaluate()
