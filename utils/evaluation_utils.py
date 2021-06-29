
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from scipy.special import softmax
import collections
import os


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    try:
        f1 = f1_score(y_true=labels, y_pred=preds)
    except:
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return (acc + f1) / 2


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    if type(preds[0]) == list:
        preds = flatten_list(preds)
        labels = flatten_list(labels)

    concatenated_preds = np.array(preds)
    concatenated_labels = np.array(labels)
    try:
        f1 = f1_score(y_true=concatenated_labels, y_pred=concatenated_preds)
    except:
        f1 = f1_score(y_true=concatenated_labels, y_pred=concatenated_preds, average='macro')
    return \
        {
            "acc": simple_accuracy(concatenated_preds, concatenated_labels),
            "f1": f1,
            "acc_and_f1": acc_and_f1(concatenated_preds, concatenated_labels)
        }


def accum_preds_and_label_ids(accum_preds, accum_label_ids, preds, label_ids, is_tok_cls=False):
    if is_tok_cls:
        if accum_preds is None:
            accum_preds = preds
            accum_label_ids = label_ids
        else:
            for i in range(len(preds)):
                accum_preds.append(preds[i])
                accum_label_ids.append(label_ids[i])
    elif len(preds) != len(label_ids):
        if accum_preds is None:
            accum_preds = preds[0][0]
            accum_label_ids = label_ids[0].detach().cpu().numpy()
        else:
            accum_preds = np.append(accum_preds, preds[0][0])
            accum_label_ids = np.append(accum_label_ids, label_ids[0].detach().cpu().numpy())
    else:
        if accum_preds is None:
            accum_preds = preds
            try:
                accum_label_ids = label_ids.detach().cpu().numpy()
            except:
                accum_label_ids = label_ids
        else:
            accum_preds = np.append(accum_preds, preds)
            try:
                accum_label_ids = np.append(accum_label_ids, label_ids.detach().cpu().numpy(), axis=0)
            except:
                accum_label_ids = np.append(accum_label_ids, label_ids, axis=0)
    return accum_preds, accum_label_ids


def calc_loss_and_acc(args, logits, label_ids, num_labels, domain_logits, domain_label_ids, num_domain_labels, tr_acc,
                      n_gpu, use_domain_loss=True):
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    task_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    if use_domain_loss:
        domain_loss = loss_fct(domain_logits.view(-1, num_domain_labels), domain_label_ids.view(-1))
        loss = task_loss + domain_loss
    else:
        loss = task_loss
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    label_ids = label_ids.detach().cpu().numpy()
    preds = np.array([pred for idx, pred in enumerate(preds) if label_ids[idx] >= 0])
    label_ids = np.array([label_id for label_id in label_ids if label_id >= 0])
    tr_acc += compute_metrics(preds, label_ids)[args.metric_to_use]

    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.

    return tr_acc, loss, preds


def flatten_list(t):
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def calc_loss_and_acc_for_tok_cls(args, logits, label_ids, num_labels, tr_acc, n_gpu):
    loss_fct = CrossEntropyLoss(ignore_index=-100)

    task_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1, ))

    loss = task_loss
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=2)
    label_ids = label_ids.detach().cpu().numpy()

    preds = [[tok_pred for tok_idx, tok_pred in enumerate(sent_pred) if label_ids[sent_idx][tok_idx] >= 0] for
             sent_idx, sent_pred in enumerate(preds)]
    label_ids = [[tok_lbl for tok_idx, tok_lbl in enumerate(sent_lbls) if label_ids[sent_idx][tok_idx] >= 0] for
                 sent_idx, sent_lbls in enumerate(label_ids)]
    preds = [[1 if tok_pred > 0 else 0 for tok_pred in sent_pred] for sent_pred in preds]
    label_ids = [[1 if tok_lbl > 0 else 0 for tok_lbl in sent_lbls] for sent_lbls in label_ids]
    tr_acc += compute_metrics(preds, label_ids)[args.metric_to_use]

    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.

    return tr_acc, loss, preds, label_ids


def evaluate(args, eval_dataloader, model, device, num_labels, num_domain_labels, is_tok_cls=False, is_ner=False,
             processor=None):
    model.eval()
    eval_loss, nb_eval_steps = 0, 0
    accum_preds, accum_label_ids, all_logits = None, None, None

    for eval_element in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, segment_ids, label_ids, domain_label_ids = eval_element[:5]
        input_ids, input_mask, segment_ids, label_ids, domain_label_ids = \
            input_ids.to(device), input_mask.to(device), segment_ids.to(device), label_ids.to(device), \
            domain_label_ids.to(device)

        with torch.no_grad():
            logits, domain_logits = model(input_ids, segment_ids, input_mask)
            use_domain_loss = True

        if is_ner:
            _, loss, preds, label_ids = calc_loss_and_acc_for_ner(args, logits, label_ids, num_labels, tr_acc=0,
                                                                  n_gpu=1, processor=processor)
        elif is_tok_cls:
            _, loss, preds, label_ids = calc_loss_and_acc_for_tok_cls(args, logits, label_ids, num_labels, tr_acc=0,
                                                                      n_gpu=1)
        else:
            _, loss, preds = calc_loss_and_acc(args, logits, label_ids, num_labels, domain_logits, domain_label_ids,
                                               num_domain_labels, tr_acc=0, n_gpu=1, use_domain_loss=use_domain_loss)

        if all_logits is None:
            all_logits = logits
        else:
            all_logits = torch.cat((all_logits, logits), 0)
        accum_preds, accum_label_ids = accum_preds_and_label_ids(accum_preds, accum_label_ids, preds, label_ids,
                                                                 is_tok_cls=is_tok_cls)
        eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_acc = compute_metrics(accum_preds, accum_label_ids)[args.metric_to_use]
    model.train()
    return eval_acc, eval_loss, accum_preds, accum_label_ids

