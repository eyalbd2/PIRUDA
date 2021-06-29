
import sys
import torch

from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def get_examples_from_raw_data(root_data_dir, src_domains_list, trg_domains_list, mode, processor):
    examples = []
    domains_list = trg_domains_list if mode == "dev_cross" else src_domains_list
    for domain in domains_list:
        data_dir = 'data/{}/{}'.format(root_data_dir, domain)
        if mode == "train":
            dom_examples = processor.get_train_examples(data_dir)
        elif mode == "dev":
            dom_examples = processor.get_dev_examples(data_dir)
        elif mode == "dev_cross":
            dom_examples = processor.get_test_examples(data_dir,)
        examples = examples + dom_examples

    return examples


def print_data_params(logger, mode, n_examples, batch_size):
    logger.info("***** {}-set *****".format(mode))
    logger.info("  Num examples = %d", n_examples)
    logger.info("  Batch size = %d", batch_size)
    return


def create_tensor_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_domain_label_ids = torch.tensor([f.domain_label_id for f in features], dtype=torch.long)
    if hasattr(features[0], 'output_ids'):
        all_output_ids = torch.tensor([f.output_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_domain_label_ids,
                             all_output_ids)
    else:
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_domain_label_ids)

    return data


def get_sampler(mode, local_rank, data):
    if mode == "train":
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    else:  # mode == "dev" or mode == "dev_cross"
        # Run prediction for full data
        sampler = SequentialSampler(data)
    return sampler










