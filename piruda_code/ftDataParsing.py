from argparse import ArgumentParser
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from tqdm import tqdm
import pickle
import torch
from pathlib import Path
import h5py
import numpy as np
from data_utils import data_dict, aspect_label_mapping

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-task', type=str)
    argparser.add_argument('-domain', type=str)
    args = argparser.parse_args()
    task = args.task
    ft = True
    data = data_dict[task]
    test_domain = args.domain
    print(f'task: {task}')
    print(f'domain: {test_domain}')
    print(f'ft: {ft}')
    pickles_root = Path('piruda-models', task, test_domain, 'ft' if ft else '')
    if not pickles_root.exists():
        pickles_root.mkdir(parents=True)
    all_domains = [d.name for d in Path('piruda-models', task).glob('*') if d.is_dir()]
    bert_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    for domain in all_domains:
        print(f'curr domain: {domain}')
        train_path = data['train_paths'][domain]
        dev_path = data['dev_paths'][domain]
        test_path = data['test_paths'][domain]
        curr_pkl_root = Path(pickles_root, domain)
        if not curr_pkl_root.exists():
            curr_pkl_root.mkdir(parents=True)
        train_dump_path = Path(curr_pkl_root, 'train_parsed.pkl')
        dev_dump_path = Path(curr_pkl_root, 'dev_parsed.pkl')
        test_dump_path = Path(curr_pkl_root, 'test_parsed.pkl')
        # Setup BERT
        saved_models_path = Path('piruda-models', task, test_domain, 'ft_models')
        models_idx = [int(d.name[len('checkpoint-'):]) for d in saved_models_path.glob('*')]
        max_idx = max(models_idx)
        model_path = Path(saved_models_path, f'checkpoint-{str(max_idx)}')

        for file_path, dump_path in zip(
                [test_path, dev_path, train_path],
                [test_dump_path, dev_dump_path, train_dump_path]):

            print(f"Processing {file_path}...")
            with open(file_path, 'rb') as f:
                curr_data = pickle.load(f)
            num_labels = len(set(curr_data[1])) if task != 'aspect' else \
                len(set([label for labels in curr_data[1] for label in labels]))
            if task == 'aspect':
                model = BertForTokenClassification.from_pretrained(model_path.__str__(), num_labels=num_labels).to(
                    device)
            else:
                model = BertForSequenceClassification.from_pretrained(model_path.__str__(), num_labels=num_labels).to(
                    device)
            # Prepare to compute BERT embeddings
            representations = []
            labels = []
            model.eval()

            for sentence, label in tqdm(zip(curr_data[0], curr_data[1])):
                with torch.no_grad():
                    if task == 'aspect':
                        new_labels = [-100]  # ignore index for CLS token
                        for word, l in zip(sentence, label):
                            word_labels = [aspect_label_mapping[l]]
                            subtokens = tokenizer(word).data['input_ids']
                            if len(subtokens) > 3:
                                # word got split - label sub-tokens with ignore index
                                word_labels.extend([-100] * (len(subtokens) - 3))
                            new_labels.extend(word_labels)
                        sentence = ' '.join(sentence)
                        padding_num = 128 - len(new_labels)
                        new_labels.extend([-100] * padding_num)
                        label = new_labels
                    elif task == 'mnli':
                        if sentence[0][-1] == '.':
                            sentence = ' '.join(sentence)
                        else:
                            sentence = '. '.join(sentence)
                    tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=128,
                                       return_tensors="pt").to(device).data
                    outputs = \
                    model(tokens['input_ids'], attention_mask=tokens['attention_mask'], output_hidden_states=True)[1][
                        -1][0]
                    if task == 'aspect':
                        # omit CLS and PAD reps and labels
                        relevant_outputs = outputs[1:-padding_num]
                        relevant_labels = label[1:-padding_num]
                        new_outputs, new_labels = [], []
                        for i, (r, l) in enumerate(zip(relevant_outputs, relevant_labels)):
                            if l == -100:
                                # ignore the token
                                continue
                            if i + 1 < len(relevant_labels) and relevant_labels[i + 1] != -100:
                                # word did not split
                                new_outputs.append(r.cpu().numpy())
                                new_labels.append(l if l == 0 else 1)
                                continue
                            # word was sub-tokenized - compute mean representation for all its sub-tokens
                            start_idx, end_idx = i, i + 1
                            while len(relevant_labels) > end_idx and relevant_labels[end_idx] == -100:
                                end_idx += 1
                            word_rep = relevant_outputs[start_idx:end_idx].mean(dim=0).cpu().numpy()
                            new_outputs.append(word_rep)
                            new_labels.append(l if l == 0 else 1)
                        representations.append(new_outputs)
                        labels.extend(new_labels)
                    else:
                        representations.append(outputs[0].cpu().numpy())
                        labels.append(label)
            representations = np.concatenate(representations) if task == 'aspect' else np.array(representations)
            labels = np.array(labels, dtype=float)
            # Save final results
            if task in ['mnli', 'aspect']:
                hf = h5py.File(dump_path, 'w')
                hf.create_dataset('representations', data=representations)
                hf.create_dataset('labels', data=labels)
                hf.close()
            else:
                with open(dump_path, "wb+") as h:
                    pickle.dump((representations, labels), h)
