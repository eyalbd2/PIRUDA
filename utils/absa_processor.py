
import csv
import pickle
import os
import random
import torch


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None, domain_label=-1):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            domain_label: (Optional) int. The domain from which the example came from.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        self.domain_label = domain_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, domain_label_id, output_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_ids
        self.domain_label_id = domain_label_id
        if output_ids is not None:
            self.output_ids = output_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class AbsaProcessor(DataProcessor):
    def __init__(self, logger, domain_label_dict):
        super().__init__()
        self.logger = logger
        self.domain_dict = domain_label_dict

    def get_train_examples(self, data_dir):
        """See base class."""
        domain_label = self.domain_dict[data_dir.split('/')[-1]]
        train_path = os.path.join(data_dir, "train")
        self.logger.info("LOOKING AT {}".format(train_path))
        with open(train_path, 'rb') as f:
            (train, labels) = pickle.load(f)
        return self._create_examples(train, labels, 'train', domain_label)

    def get_dev_examples(self, data_dir):
        """See base class."""
        domain_label = self.domain_dict[data_dir.split('/')[-1]]
        dev_path = os.path.join(data_dir, "dev")
        self.logger.info("LOOKING AT {}".format(dev_path))
        with open(dev_path, 'rb') as f:
            (dev, labels) = pickle.load(f)
        return self._create_examples(dev, labels, 'dev', domain_label)

    def get_test_examples(self, data_dir):
        """See base class."""
        domain_label = -100  # ignore_index
        test_path = os.path.join(data_dir, "test")
        self.logger.info("LOOKING AT {}".format(test_path))
        with open(test_path, 'rb') as f:
            (test, labels) = pickle.load(f)
        return self._create_examples(test, labels, 'dev_cross', domain_label)

    def get_labels(self):
        """See base class."""
        return ["O", "B-AS", "I-AS"]

    def _create_examples(self, x, labels, set_type, domain_label):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x, labels)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = data_point[0]
            text_b = None
            labels = data_point[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels, domain_label=domain_label))
        return examples


def print_data_sample(logger, example, input_ids, input_mask, segment_ids, label_id, domain_label_id, input_text=None):
    logger.info("*** Example ***")
    logger.info("guid: %s" % example.guid)
    logger.info("tokens: %s" % example.text_a)
    if input_text is not None:
        logger.info("input text: %s" % input_text)
    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    logger.info("label: %s (id = %d)" % (example.label, label_id))
    logger.info("domain label: id = %d" % domain_label_id)
    return


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


def zero_pad(input_ids, input_mask, segment_ids, max_seq_length):
    # Zero-pad up to the sequence length.
    for i in range(len(input_ids)):
        padding = [0] * (max_seq_length - len(input_ids[i]))
        input_ids[i] += padding
        input_mask[i] += padding
        segment_ids[i] += padding
    return input_ids, input_mask, segment_ids


def sanity_checks(input_ids, input_mask, segment_ids, max_seq_length):
    for i in range(len(input_ids)):
        assert len(input_ids[i]) == max_seq_length
        assert len(input_mask[i]) == max_seq_length
        assert len(segment_ids[i]) == max_seq_length


def token_classification_batch_encode_plus(text_a, labels, label_map, tokenizer, max_seq_length,
                                           pad_token_label_id=-100):
    tokens, label_ids = [], []
    tokens += ["[CLS]"]
    label_ids += [pad_token_label_id]

    for word, label in zip(text_a, labels):
        label_id = label_map[label]
        word_tokens = tokenizer.tokenize(word)
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            cur_pad = [pad_token_label_id] * (len(word_tokens) - 1)
            label_ids.extend([label_id] + cur_pad)

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[: (max_seq_length - 1)]
        label_ids = label_ids[: (max_seq_length - 1)]

    tokens += ["[SEP]"]
    label_ids += [pad_token_label_id]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    # Padding on the right
    input_ids += [0] * padding_length
    input_mask += [0] * padding_length
    label_ids += [pad_token_label_id] * padding_length

    segment_ids = [0] * max_seq_length

    return input_ids, input_mask, segment_ids, label_ids











