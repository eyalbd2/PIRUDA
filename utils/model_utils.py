
import os
import sys
import torch

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameters(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n + " - " + str(p.numel()))
    return


def save_model_weights(args, epoch, model, is_best_epoch, saved_results):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    if args.save_on_epoch_end:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '.Epoch_{}'.format(epoch + 1))
        torch.save(model_to_save.state_dict(), output_model_file)
    if args.save_best_weights and is_best_epoch:
        print("Saving model, {} improved from {} to {}".format(args.save_according_to, saved_results['prev_best_dev'],
                                                               saved_results['best_dev']))
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    return




