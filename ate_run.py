from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from ate_modeling import BertForSequenceLabeling
from optimization import BertAdam
from tokenization import BertTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ate_features import ATEProcessor, convert_examples_to_features, get_labels
from utils import get_logger, get_aspect_chunks

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max(0, (1.0 - x) / (1.0 - warmup))

def parse_input_parameter():
    global logger
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--task_name", default="ate", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--data_name", default="", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--valid_file", default=None, type=str, required=False)
    parser.add_argument("--test_file", default=None, type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \nthan this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n0 (default value): dynamic loss scaling.\nPositive power of 2: static loss scaling value.\n")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal CoQA evaluation.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.");
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument("--use_ghl", action='store_true', help="Whether use weighted cross entropy to decoder.")
    parser.add_argument("--use_vat", action='store_true', help="Whether use vat to encoder.")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  {}: {}".format(key, args.__dict__[key]))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    task_config = {
        "use_ghl": args.use_ghl,
        "use_vat": args.use_vat,
    }

    return args, task_config

def init_device(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    return device, n_gpu

def init_model(args, num_labels, task_config, device, n_gpu):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        if "model_state_dict" in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else \
        os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                                    num_labels=num_labels, task_config=task_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model

def prep_optimizer(args, model, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False, max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    return optimizer

def dataloader_train(args, tokenizer, file_path):
    dataset = ATEProcessor(file_path=file_path, set_type="train")
    logger.info("Loaded train file: {}".format(file_path))
    labels = get_labels(dataset.label_list)

    features = convert_examples_to_features(dataset.examples, labels,
                                                  args.max_seq_length, tokenizer,
                                                  verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_thread_reader)

    return dataloader, train_data, labels

def dataloader_val(args, tokenizer, file_path, labels, set_type="val"):

    dataset = ATEProcessor(file_path=file_path, set_type=set_type)
    logger.info("Loaded eval file: {}".format(file_path))

    eval_features = convert_examples_to_features(dataset.examples, labels,
                                            args.max_seq_length, tokenizer,
                                            verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader, eval_data

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer, optimizer, global_step, num_train_optimization_steps):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = 100
    start_time = time.time()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    weight_gradient = None  # Init in model: [bin_num]
    weight_gradient_labels = None  # Init in model:
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss, acc_sum, weight_gradient, weight_gradient_labels = model(input_ids, segment_ids, input_mask, label_ids, weight_gradient, weight_gradient_labels)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += float(loss.item())
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_step == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.num_train_epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss.item()), (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    tr_loss = tr_loss / len(train_dataloader)
    return tr_loss, global_step

def cal_f1(y_true, y_pred):
    correct_pred, total_ground, total_pred = 0., 0., 0.
    for ground_seq, pred_seq in zip(y_true, y_pred):
        lab_chunks = get_aspect_chunks(ground_seq, default="O")
        lab_pred_chunks = get_aspect_chunks(pred_seq, default="O")
        lab_chunks = set(lab_chunks)
        lab_pred_chunks = set(lab_pred_chunks)

        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_ground += len(lab_chunks)

    p = correct_pred / total_pred if total_pred > 0 else 0.
    r = correct_pred / total_ground if total_ground > 0 else 0.
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.
    return p,r,f1

def eval_epoch(model, eval_dataloader, label_list, device):
    model.eval()

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list)}
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()

        label_ids = label_ids.to('cpu').numpy()
        for i, lab_ids in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, l in enumerate(lab_ids):
                if l != -1:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

            y_true.append(temp_1)
            y_pred.append(temp_2)

    p, r, f1 = cal_f1(y_true, y_pred)
    logger.info("p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(p, r, f1))

    return p, r, f1

def predict(epoch, args, test_dataloader, model, label_list, tokenizer, device):
    model.eval()

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list)}
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Test", ncols=100, ascii=True):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()

        label_ids = label_ids.to('cpu').numpy()
        for i, lab_ids in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, l in enumerate(lab_ids):
                if l != -1:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])

            y_true.append(temp_1)
            y_pred.append(temp_2)

    p, r, f1 = cal_f1(y_true, y_pred)
    logger.info("p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(p, r, f1))

def save_model(epoch, args, model):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, num_labels, task_config, device):
    model_file = os.path.join(
        args.output_dir,
        "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        logger.info("Model loaded from %s", model_file)
        model = BertForSequenceLabeling.from_pretrained(args.bert_model,
                                                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                        state_dict=model_state_dict, num_labels=num_labels, task_config=task_config)
        model.to(device)
    else:
        model = None
    return model

DATALOADER_DICT = {}
DATALOADER_DICT["ate"] = {"train":dataloader_train, "eval":dataloader_val}

DATASET_DICT={}
DATASET_DICT["lap"] = {"train_file":"laptops_2014_train.txt", "valid_file":"laptops_2014_trial.txt", "test_file":"laptops_2014_test.gold.txt"}
DATASET_DICT["res"] = {"train_file":"restaurants_union_train.txt", "valid_file":"restaurants_union_trial.txt", "test_file":"restaurants_union_test.gold.txt"}
for i in ["2014", "2015", "2016"]:
    DATASET_DICT["res{}".format(i)] = {"train_file": "restaurants_{}_train.txt".format(i), "valid_file": "restaurants_{}_trial.txt".format(i), "test_file": "restaurants_{}_test.gold.txt".format(i)}
for i in range(10):
    DATASET_DICT["twt{}".format(i+1)] = {"train_file":"twitter_{}_train.txt".format(i+1), "valid_file":"twitter_{}_test.gold.txt".format(i+1), "test_file":"twitter_{}_test.gold.txt".format(i+1)}

def main():
    global logger
    args, task_config = parse_input_parameter()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device, n_gpu = init_device(args)

    data_name = args.data_name.lower()
    if data_name in DATASET_DICT:
        args.train_file = DATASET_DICT[data_name]["train_file"]
        args.valid_file = DATASET_DICT[data_name]["valid_file"]
        args.test_file = DATASET_DICT[data_name]["test_file"]
    else:
        assert args.train_file is not None
        assert args.valid_file is not None
        assert args.test_file is not None

    task_name = args.task_name.lower()
    if task_name not in DATALOADER_DICT:
        raise ValueError("Task not found: %s" % (task_name))

    if n_gpu > 1 and (args.use_ghl):
        logger.warning("Multi-GPU make the results not reproduce.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Generate label list from training dataset
    file_path = os.path.join(args.data_dir, args.train_file)
    train_dataloader, train_examples, label_list = DATALOADER_DICT[task_name]["train"](args, tokenizer, file_path)
    logging.info("Labels are = %s:", "["+", ".join(label_list)+"]")
    num_labels = len(label_list)

    model = init_model(args, num_labels, task_config, device, n_gpu)

    # Generate test dataset
    file_path = os.path.join(args.data_dir, args.test_file)
    test_dataloader, test_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                        labels=label_list, set_type="test")
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        num_train_optimization_steps = (int(len(
            train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        optimizer = prep_optimizer(args, model, num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        file_path = os.path.join(args.data_dir, args.valid_file)
        eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                            labels=label_list, set_type="val")
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        global_step = 0
        for epoch in range(args.num_train_epochs):
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer,
                                               optimizer, global_step, num_train_optimization_steps)
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.num_train_epochs, tr_loss)
            save_model(epoch, args, model)
            eval_epoch(model, eval_dataloader, label_list, device)

        logger.info("***Results on test***")
        eval_epoch(model, test_dataloader, label_list, device)
    elif args.do_eval:
        eval_epoch(model, test_dataloader, label_list, device)
    else:
        if args.init_model:
            eval_epoch(model, test_dataloader, label_list, device)
        else:
            for epoch in range(args.num_train_epochs):
                # Load a trained model that you have fine-tuned
                model = load_model(epoch, args, num_labels, task_config, device)
                if not model:
                    break
                eval_epoch(model, test_dataloader, label_list, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard break~")