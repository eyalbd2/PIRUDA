
import os
import sys
import random
import numpy as np
import torch

from pytorch_pretrained_bert.optimization import BertAdam
from utils.model_utils import save_model_weights


def run_initial_setups(args, logging, logger):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))
    return device, n_gpu


def get_domain_labels_dict(args):
    all_src_domains = args.src_domains.split(',')
    domain_dict = {dom: i for i, dom in enumerate(all_src_domains)}
    print("------ domain_dict = ", domain_dict)
    return domain_dict


def print_epoch_results(args, logger, task_logger, epoch, result, saved_results):
    logger.info("***** Evaluation results *****")
    logger.info('Epoch {}'.format(epoch + 1))
    logger.info("Train:   {} = {}, Loss = {}".format(args.metric_to_use, result['acc'], result['loss']))
    logger.info("Valid:   {} = {}, Loss = {}".format(args.metric_to_use, result['dev_acc'], result['dev_loss']))
    logger.info("Test:    {} = {}, Loss = {}".format(args.metric_to_use, result['dev_cross_acc'],
                                                     result['dev_cross_loss']))
    logger.info("Best results: in domain - {}, Cross Domain - {}".format(saved_results['best_dev'],
                                                                         saved_results['best_test']))
    if task_logger is not None:
        task_logger.report_scalar(title='{}'.format(args.metric_to_use), series='Train', value=result['acc'],
                                  iteration=epoch)
        task_logger.report_scalar(title='{}'.format(args.metric_to_use), series='Dev', value=result['dev_acc'],
                                  iteration=epoch)
        task_logger.report_scalar(title='{}'.format(args.metric_to_use), series='Test', value=result['dev_cross_acc'],
                                  iteration=epoch)
        task_logger.report_scalar(title='Loss'.format(args.metric_to_use), series='Train', value=result['loss'],
                                  iteration=epoch)
        task_logger.report_scalar(title='Loss'.format(args.metric_to_use), series='Dev', value=result['dev_loss'],
                                  iteration=epoch)
        task_logger.report_scalar(title='Loss'.format(args.metric_to_use), series='Test',
                                  value=result['dev_cross_loss'],
                                  iteration=epoch)
    return


def save_logits_and_labels(args, results):
    if len(results["test_results"]) > 2:
        save_dict = {"logits": results["test_results"][2], "gold_labels": results["test_results"][3]}
        torch.save(save_dict, os.path.join(args.output_dir, "logits_and_labels-best_model"))


def save_generated_preds(args, results):
    if len(results["test_results"]) > 4 and results["test_results"][4] is not None:
        accum_pred_ids = results["test_results"][4]
        torch.save(accum_pred_ids, os.path.join(args.output_dir, "generated_ids"))


def update_saved_results(args, result, saved_results):
    # Init a flag with false
    is_best_epoch = False
    if args.save_according_to == 'acc' and result["dev_acc"] > saved_results['best_dev_acc']:
        is_best_epoch = True
        saved_results['prev_best_dev'] = saved_results['best_dev_acc']
        saved_results['best_dev_acc'] = result["dev_acc"]
        saved_results['best_dev'] = result["dev_acc"]
        saved_results['best_test'] = result["dev_cross_acc"]
        save_logits_and_labels(args, result)
        save_generated_preds(args, result)

    elif args.save_according_to == 'loss' and result["dev_loss"] < saved_results['best_dev_loss']:
        is_best_epoch = True
        saved_results['prev_best_dev'] = saved_results['best_dev_loss']
        saved_results['best_dev_loss'] = result["dev_loss"]
        saved_results['best_dev'] = result["dev_loss"]
        saved_results['best_test'] = result["dev_cross_acc"]
        save_logits_and_labels(args, result)
        save_generated_preds(args, result)

    return is_best_epoch, saved_results


def write_best_results(args, saved_results):
    final_output_eval_file = os.path.join(args.output_dir, "final_eval_results.txt")
    with open(final_output_eval_file, "w") as writer:
        writer.write("Results:\n")
        writer.write("  %s = %s\n" % ('in', str(saved_results['best_dev'])))
        writer.write("  %s = %s\n" % ('cross', str(saved_results['best_test'])))
    return


def print_and_save_results(args, epoch, logger, task_logger, result, model, saved_results):
    is_best_epoch, saved_results = update_saved_results(args, result, saved_results)
    save_model_weights(args, epoch, model, is_best_epoch, saved_results)
    print_epoch_results(args, logger, task_logger, epoch, result, saved_results)
    write_best_results(args, saved_results)
    # if is_best_epoch:
    #     pred_path = os.path.join(args.output_dir, "model_preds")
    #     torch.save(np.argmax(result["test_results"][2].detach().cpu().numpy(), axis=1), pred_path)
    return saved_results


def init_all_seeds(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    return


def check_for_arg_failures(args):
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if os.path.exists(args.output_dir):
        print(("Output directory ({}) already exists and is not empty.".format(args.output_dir)))
    else:
        os.makedirs(args.output_dir)
    return


def calc_num_train_optimization_steps(args, processor):
    domains_list = args.src_domains.split(',')
    train_examples = []
    for domain in domains_list:
        data_dir = 'data/{}/{}'.format(args.root_data_dir, domain)
        dom_examples = processor.get_train_examples(data_dir)
        train_examples = train_examples + dom_examples

    num_train_optimization_steps = int((len(train_examples) / args.train_batch_size) + 0.5) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    return num_train_optimization_steps


def prepare_optimizer(args, model, num_train_optimization_steps):
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
    return optimizer


def update_global_epoch_params(tr_loss, loss_item, nb_tr_steps):
    tr_loss += loss_item
    nb_tr_steps += 1
    return tr_loss, nb_tr_steps

