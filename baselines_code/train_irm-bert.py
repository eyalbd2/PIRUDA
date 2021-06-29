
from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
from tqdm import tqdm, trange
import os
import numpy as np

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import grad


from utils.processor_utils import BinaryClassificationProcessor, MnliClassificationProcessor
from utils.data_load_utils import get_examples_from_raw_data, print_data_params, \
    create_tensor_dataset, get_sampler
from utils.absa_processor import AbsaProcessor, token_classification_batch_encode_plus
from utils.train_utils import get_domain_labels_dict, print_and_save_results, init_all_seeds, \
    update_global_epoch_params, run_initial_setups, calc_num_train_optimization_steps, \
    check_for_arg_failures
from utils.evaluation_utils import accum_preds_and_label_ids, compute_metrics
from utils.processor_utils import BinaryClassificationProcessor, MnliClassificationProcessor
from model_utils import count_parameters

logger = logging.getLogger(__name__)

DATA_SPECIFICATIONS = \
    {
        "rumour": ("rumour_data", "f1", BinaryClassificationProcessor),
        "blitzer": ("blitzer_data", "acc", BinaryClassificationProcessor),
        "mnli": ("mnli_data", "f1", MnliClassificationProcessor),
        "absa": ("absa_data", "f1", AbsaProcessor),
    }


class BertForClassificationIrm(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, is_tok_level_task=False):
        super(BertForClassificationIrm, self).__init__(config)
        self.num_labels = num_labels
        self.is_tok_level_task = is_tok_level_task
        self.bert = BertModel(config)
        self.task_classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        enc_sequence, pooler_output = self.bert(input_ids, attention_mask=attention_mask,
                                                output_all_encoded_layers=False)
        bert_output = enc_sequence if self.is_tok_level_task else pooler_output
        cls_logits = self.task_classifier(bert_output)
        return cls_logits


def calc_loss_and_preds(args, logits, label_ids, num_labels, n_gpu):
    is_tok_level_task = True if args.data_type == 'absa' else False
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    task_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    loss = task_loss

    # Calc task accuracy/F1
    preds = np.argmax(logits.view(-1, num_labels).detach().cpu().numpy(), axis=1)
    label_ids = label_ids.view(-1).detach().cpu().numpy()
    preds = np.array([pred for idx, pred in enumerate(preds) if label_ids[idx] >= 0])
    label_ids = np.array([label_id for label_id in label_ids if label_id >= 0])

    if is_tok_level_task:
        preds[preds > 1] = 1
        label_ids[label_ids > 1] = 1

    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.

    return loss, preds, label_ids


def evaluate(args, eval_dataloader, model, device, num_labels, is_tok_cls=False):
    model.eval()
    eval_loss, nb_eval_steps = 0, 0
    accum_preds, accum_label_ids, all_logits = None, None, None

    for eval_element in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, segment_ids, label_ids, domain_label_ids = eval_element[:5]
        input_ids, input_mask, segment_ids, label_ids, domain_label_ids = \
            input_ids.to(device), input_mask.to(device), segment_ids.to(device), label_ids.to(device), \
            domain_label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask)

        loss, preds, label_ids = calc_loss_and_preds(args, logits, label_ids, num_labels, n_gpu=1)
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
    return eval_acc, eval_loss


def calc_num_batches_in_epoch(dataloaders):
    n_batches_per_domain = []
    for dataloader in dataloaders:
        n_batches_per_domain.append(len(dataloader))
    return min(n_batches_per_domain)


def load_model_and_tokenizer(args, num_labels, device, n_gpu):
    model = BertForClassificationIrm.from_pretrained('bert-base-uncased',
                                                     num_labels=num_labels,
                                                     is_tok_level_task=True if args.data_type == 'absa' else False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model.to(device)
    print("----------------------------------------------------------------------")
    print("Num Parameters -", count_parameters(model))
    # print_parameters(model)
    print("----------------------------------------------------------------------")
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model, tokenizer


def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, domain_label_id, output_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.domain_label_id = domain_label_id
        if output_ids is not None:
            self.output_ids = output_ids


def sanity_checks(input_ids, input_mask, segment_ids, max_seq_length):
    for i in range(len(input_ids)):
        assert len(input_ids[i]) == max_seq_length
        assert len(input_mask[i]) == max_seq_length
        assert len(segment_ids[i]) == max_seq_length


def zero_pad(input_ids, input_mask, segment_ids, max_seq_length):
    # Zero-pad up to the sequence length.
    for i in range(len(input_ids)):
        padding = [0] * (max_seq_length - len(input_ids[i]))
        input_ids[i] += padding
        input_mask[i] += padding
        segment_ids[i] += padding
    return input_ids, input_mask, segment_ids


def convert_example_to_ids(example, max_seq_length, tokenizer):
    # Prepare 'input_ids', 'input_mask', and 'segment_ids' before padding all to 'max_seq_length'
    # text_b is always None - so we delete code that supports truncation of token_a with tokens_b
    tokens_a = tokenizer.tokenize(example.text_a)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # Add [CLS] and [SEP] tokens
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)
    return [input_ids], [input_mask], [segment_ids]


def convert_examples_to_features(examples, label_list, domains_list, max_seq_length, tokenizer, logger,
                                 is_tok_level_task, do_print_data_sample=False):
    if is_tok_level_task:
        features = convert_examples_to_tok_level_features(examples, label_list, domains_list, max_seq_length,
                                                          tokenizer, logger, do_print_data_sample=do_print_data_sample)
    else:
        features = convert_examples_to_text_level_features(examples, label_list, domains_list, max_seq_length,
                                                           tokenizer, logger, do_print_data_sample=do_print_data_sample)
    return features


def convert_examples_to_text_level_features(examples, label_list, domains_list, max_seq_length, tokenizer, logger,
                                            do_print_data_sample=False):
    """Loads a data file into a list of `InputBatch`s."""
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    label_to_domain_map = {i: label for i, label in enumerate(domains_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        input_ids, input_mask, segment_ids = convert_example_to_ids(example, max_seq_length, tokenizer)
        input_ids, input_mask, segment_ids = zero_pad(input_ids, input_mask, segment_ids, max_seq_length)
        sanity_checks(input_ids, input_mask, segment_ids, max_seq_length)
        if ex_index < 1 and do_print_data_sample:
            print_data_sample(logger, example, input_ids[0], input_mask[0], segment_ids[0], label_map[example.label],
                              example.domain_label)
        for i in range(len(input_ids)):
            features.append(
                InputFeatures(input_ids=input_ids[i],
                              input_mask=input_mask[i],
                              segment_ids=segment_ids[i],
                              label_id=label_map[example.label],
                              domain_label_id=example.domain_label))
    return features


def convert_examples_to_tok_level_features(examples, label_list, domains_list, max_seq_length, tokenizer, logger,
                                           do_print_data_sample=False):
    """Loads a data file into a list of `InputBatch`s."""
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    label_to_domain_map = {i: label for i, label in enumerate(domains_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        input_ids, input_mask, segment_ids, output_ids = token_classification_batch_encode_plus(example.text_a,
                                                                                                example.labels,
                                                                                                label_map, tokenizer,
                                                                                                max_seq_length)
        sanity_checks([input_ids], [input_mask], [segment_ids], max_seq_length)
        if ex_index < 1 and do_print_data_sample:
            print_data_sample(logger, example, input_ids, input_mask, segment_ids, output_ids,
                              example.domain_label)
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=output_ids,
                          domain_label_id=example.domain_label))
    return features


def load_all_data_loaders(args, logger, processor, tokenizer, label_list):
    src_domains_list = args.src_domains.split(',')
    trg_domains_list = args.trg_domains.split(',')
    is_tok_level_task = True if args.data_type == 'absa' else False
    # load train data
    examples = get_examples_from_raw_data(args.root_data_dir, src_domains_list, trg_domains_list, 'train', processor)
    features = convert_examples_to_features(examples, label_list, src_domains_list, args.max_seq_length, tokenizer,
                                            logger, is_tok_level_task)
    print_data_params(logger, 'train', len(features), args.train_batch_size)
    data = create_tensor_dataset(features)
    sampler = get_sampler('train', args.local_rank, data)
    train_dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)

    # load dev data
    examples = get_examples_from_raw_data(args.root_data_dir, src_domains_list, trg_domains_list, 'dev', processor)
    features = convert_examples_to_features(examples, label_list, src_domains_list, args.max_seq_length, tokenizer,
                                            logger, is_tok_level_task)
    print_data_params(logger, 'dev', len(features), args.eval_batch_size)
    data = create_tensor_dataset(features)
    sampler = get_sampler('dev', args.local_rank, data)
    dev_dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    # load test data
    examples = get_examples_from_raw_data(args.root_data_dir, src_domains_list, trg_domains_list, 'dev_cross', processor)
    features = convert_examples_to_features(examples, label_list, src_domains_list, args.max_seq_length, tokenizer,
                                            logger, is_tok_level_task)
    print_data_params(logger, 'dev_cross', len(features), args.eval_batch_size)
    data = create_tensor_dataset(features)
    sampler = get_sampler('dev_cross', args.local_rank, data)
    test_dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    return [train_dataloader, dev_dataloader, test_dataloader]


def load_all_train_data_loaders(args, logger, processor, tokenizer, label_list):
    is_tok_level_task = True if args.data_type == 'absa' else False
    dataloader_list = []
    domains_list = args.src_domains.split(',')
    trg_domains_list = args.trg_domains.split(',')
    for domain in domains_list:
        examples = get_examples_from_raw_data(args.root_data_dir, [domain], trg_domains_list, 'train', processor)
        features = convert_examples_to_features(examples, label_list, [domain], args.max_seq_length, tokenizer,
                                     logger, is_tok_level_task)
        print_data_params(logger, 'train', len(examples), args.train_batch_size)
        data = create_tensor_dataset(features)
        sampler = get_sampler('train', args.local_rank, data)
        cur_dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        dataloader_list.append(cur_dataloader)
    return dataloader_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_domains", type=str, required=True, help="Source names separated with comma - NO SPACES.")
    parser.add_argument("--trg_domains", type=str, required=True, help="Target names separated with comma - NO SPACES.")
    parser.add_argument("--data_type", type=str, required=True, help="'blitzer' or 'rumour'.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save experiment results.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help="BERT model to use.")
    parser.add_argument("--model_name", type=str, default='BertDomainCls', help="[BertDomainCls, BertDomainAttn]")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Max seq-len after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_conditional_input", action='store_true', help="Use a domain conditional input.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of learning rate warmup.")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--load_finetuned_model", action='store_true', help="load fine-tuned model.")
    parser.add_argument("--fintuned_model_path", type=str, help="Path to a fine-tuned model.")
    parser.add_argument("--save_best_weights", action='store_true', help="save model weight best performing epoch.")
    parser.add_argument("--save_on_epoch_end", action='store_true', help="Save weights each time an epoch ends.")
    parser.add_argument('--save_according_to', type=str, default='acc', help="save according to (dev) acc/f1 or loss")
    args = parser.parse_args()

    # Remember to call 'init_all_seeds' after 'Task' initialization.
    # task = Task.init(project_name="MultiDomain-T5", task_name='/'.join(args.output_dir.split('/')[1:-1]))
    # task_logger = task.get_logger()
    task_logger = None

    device, n_gpu = run_initial_setups(args, logging, logger)
    init_all_seeds(args, n_gpu)
    check_for_arg_failures(args)
    (args.root_data_dir, args.metric_to_use, processor_class) = DATA_SPECIFICATIONS[args.data_type]
    domain_label_dict = get_domain_labels_dict(args)
    processor = processor_class(logger, domain_label_dict)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    model, tokenizer = load_model_and_tokenizer(args, num_labels, device, n_gpu)
    [_, eval_dataloader, test_dataloader] = load_all_data_loaders(args, logger, processor, tokenizer, label_list)
    train_dataloaders = load_all_train_data_loaders(args, logger, processor, tokenizer, label_list)

    if args.do_train:
        # Prepare optimizer
        dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
        num_train_optimization_steps = calc_num_train_optimization_steps(args, processor)
        num_batches_in_epoch = calc_num_batches_in_epoch(train_dataloaders)
        print(f"--- Number of batches in Epoch = {num_batches_in_epoch}")
        try:
            param_optimizer = list(model.module.named_parameters())
        except:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

        saved_results = \
            {
                'best_dev_acc': 0.0,
                'best_dev_loss': 100000.0,
                'prev_best_dev': 0.0,
                'best_dev': 0.0,
                'best_test': 0.0,
            }

        # main training loop
        # loss_fct = CrossEntropyLoss(ignore_index=-100)
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            penalty_multiplier = (epoch+1) ** 1.6
            for batch_num in trange(num_batches_in_epoch, desc="Batch"):
                optimizer.zero_grad()
                error = 0
                penalty = 0
                for train_dataloader in train_dataloaders:
                    input_ids, input_mask, segment_ids, label_ids, domain_label_ids = next(iter(train_dataloader))
                    input_ids, input_mask, segment_ids, label_ids, domain_label_ids = input_ids.to(
                        device), input_mask.to(device), segment_ids.to(device), label_ids.to(
                        device), domain_label_ids.to(device)
                    logits = model(input_ids, input_mask)
                    if args.data_type == 'absa':
                        label_ids[label_ids < 0] = int(num_labels)
                        one_hot = torch.nn.functional.one_hot(label_ids, num_classes=int(num_labels + 1))
                        one_hot = one_hot[:, :, :int(num_labels)]
                    else:
                        one_hot = torch.nn.functional.one_hot(label_ids, num_classes=int(num_labels))
                    loss_erm = F.binary_cross_entropy_with_logits(logits * dummy_w, one_hot.float(), reduction='none')
                    penalty += compute_irm_penalty(loss_erm, dummy_w)
                    error += loss_erm.mean()
                (error + penalty_multiplier * penalty).backward()
                optimizer.step()
            tr_acc, tr_loss = 0, 0
            eval_results = evaluate(args, eval_dataloader, model, device, num_labels)
            test_results = evaluate(args, test_dataloader, model, device, num_labels)
            results = {"acc": tr_acc, "loss": tr_loss, "dev_acc": eval_results[0], "dev_loss": eval_results[1],
                       "dev_cross_acc": test_results[0], "dev_cross_loss": test_results[1],
                       "test_results": test_results}

            # print and save results
            saved_results = print_and_save_results(args, epoch, logger, task_logger, results, model, saved_results)

    elif args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_results = evaluate(args, eval_dataloader, model, device, num_labels, len(domain_label_dict))
        test_results = evaluate(args, test_dataloader, model, device, num_labels, len(domain_label_dict))

        # print results
        logger.info('Dev Accuracy: {}, Test Acc'.format(eval_results[0], test_results[0]))
        logger.info('Dev Loss: {}, Test Loss'.format(eval_results[1], test_results[1]))

    else:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


if __name__ == "__main__":
    main()
