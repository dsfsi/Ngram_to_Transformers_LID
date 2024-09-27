import argparse
import glob
import logging
import os
import random
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm, trange
# from util_textclass import get_labels, read_instances_from_file, convert_instances_to_features_and_labels
import sklearn.metrics
import argparse
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


# In[ ]:


class Instance:

    def __init__(self, text, label):
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

def read_instances_from_file(data_dir, mode, delimiter="\t"):
    file_path = os.path.join(data_dir, "{}.csv".format(mode))
    instances = []

    df = pd.read_csv(file_path, sep=',')
    N = df.shape[0]

    for i in range(N):
        instances.append(Instance(df['sentence'].iloc[i], df['label'].iloc[i]))
    '''
    with open(file_path, "r", encoding='utf-8') as input_file:
        line_data = input_file.read()

    line_data = line_data.splitlines()
    for l, line in enumerate(line_data):
        if l==0:
            continue
        else:
            text_vals = line.strip().split(delimiter)
            text, label = ' '.join(text_vals[:-1]), text_vals[-1]
            instances.append(Instance(text, label))
    '''

    return instances

def convert_instances_to_features_and_labels(instances, tokenizer, labels, max_seq_length):
    label_map = {label: i for i, label in enumerate(labels)}

    features = []
    for instance_idx, instance in enumerate(instances):
        tokenization_result = tokenizer.encode_plus(text=instance.text,
                                                    max_length=max_seq_length, pad_to_max_length=True,
                                                     truncation=True)
        token_ids = tokenization_result["input_ids"]
        try:
            token_type_ids = tokenization_result["token_type_ids"]
        except:
            token_type_ids = None
        attention_masks = tokenization_result["attention_mask"]

        if instance.label not in label_map:
            continue
        label = label_map[instance.label]

        if "num_truncated_tokens" in tokenization_result:
            logger.info(f"Removed {tokenization_result['num_truncated_tokens']} tokens from {instance.text} as they "
                         f"were longer than max_seq_length {max_seq_length}.")

        if instance_idx < 3:
            logger.info("Tokenization example")
            logger.info(f"  text: {instance.text}")
            logger.info(f"  tokens (by input): {tokenizer.tokenize(instance.text)}")
            logger.info(f"  token_ids: {tokenization_result['input_ids']}")
            #logger.info(f"  token_type_ids: {tokenization_result['token_type_ids']}")
            logger.info(f"  attention mask: {tokenization_result['attention_mask']}")

        features.append(
            InputFeatures(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, label=label)
        )

    return features

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        return  ['nso', 'tsn', 'sot','ven', 'af', 'tso', 'ssw', 'zul', 'xho', 'nbl', 'eng']

# CSV_Train_Test_Split_NCHLT_plus_Vuk
class Sort_CSV_into2:
      def __init__(self,):
             pass
      
      def clean_text(self, text):
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters and numbers (retain only alphabets and spaces)
            # text = re.sub(r'[^A-Za-z\s]', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Convert text to lowercase
            text = text.lower()
            
            return text
      def read_and_init(self, train_data_root_path):
            #Train, Dev, Test
            # Step 4: Split the data
            # Check if the current item is a file
            data_tr = []
            data_d = []
            data_te = []
            for item in os.listdir(train_data_root_path):
                    # Construct the full path to the current item
                    item_path = os.path.join(train_data_root_path , item)
                    # Check if the current item is a directory
                    if os.path.isdir(item_path):
                        print("-------------------------------------------------------------")
                        print("Collecting data For language: ", item_path.split("/")[-1])

                        # Loop through the contents of the subfolder
                        for sub_item in os.listdir(item_path):
                            # Construct the full path to the sub-item
                            sub_item_path = os.path.join(item_path, sub_item)

                            # Check if the sub-item is a file or directory
                            if os.path.isdir(sub_item_path):
                                language = item_path.split("/")[-1]
                                files_manager = {}
                                for file_name in os.listdir(sub_item_path):
                                        file_path = os.path.join(sub_item_path, file_name)
                                        #print(file_path)
                                        if 'train' in file_name:
                                                files_manager['train_file'] = file_path
                                                # data_tr.append(pd.read_csv(files_manager['train_file']))
                                                tr_list = list(pd.read_csv(files_manager['train_file'])['sentence'])
                                                new_tr_list = []
                                                for sent in tr_list:
                                                            clean_sent = self.clean_text(sent)
                                                            split_s = clean_sent.split()
                                                            if len(split_s) >=3 and len(split_s) <= 50:
                                                                        new_tr_list.append(clean_sent)
                                                data = {'sentence': new_tr_list, 'label': [item_path.split("/")[-1] for i in range(len(new_tr_list))]}
                                                data_tr.append(pd.DataFrame(data)) 
                                        elif 'dev' in file_name:
                                                files_manager['dev_file'] = file_path
                                                # data_d.append(pd.read_csv(files_manager['dev_file']))
                                                d_list = list(pd.read_csv(files_manager['dev_file'])['sentence'])
                                                new_d_list = []
                                                for sent in d_list:
                                                            clean_sent = self.clean_text(sent)
                                                            split_s = clean_sent.split()
                                                            if len(split_s) >=3 and len(split_s) <= 50:
                                                                        new_d_list.append(clean_sent)
                                                data = {'sentence': new_d_list, 'label': [item_path.split("/")[-1] for i in range(len(new_d_list))]}
                                                data_d.append(pd.DataFrame(data)) 
                                        elif 'test' in file_name:
                                                files_manager['test_file'] = file_path  
                                                # data_te.append(pd.read_csv(files_manager['test_file']))
                                                t_list = list(pd.read_csv(files_manager['test_file'])['sentence'])
                                                new_t_list = []
                                                for sent in t_list:
                                                            clean_sent = self.clean_text(sent)
                                                            split_s = clean_sent.split()
                                                            if len(split_s) >=3 and len(split_s) <= 50:
                                                                        new_t_list.append(clean_sent)
                                                data = {'sentence': new_t_list, 'label': [item_path.split("/")[-1] for i in range(len(new_t_list))]}
                                                data_te.append(pd.DataFrame(data)) 
                                        else:
                                                print("This file is unknow") 
                            else:
                                # its a file
                                files_manager = {}
                                if 'train' in sub_item_path:
                                        files_manager['train_file'] =sub_item_path
                                        # data_tr.append(pd.read_csv(files_manager['train_file']))
                                        tr_list = list(pd.read_csv(files_manager['train_file'])['sentence'])
                                        new_tr_list = []
                                        for sent in tr_list:
                                                    clean_sent = self.clean_text(sent)
                                                    split_s = clean_sent.split()
                                                    if len(split_s) >=3 and len(split_s) <= 50:
                                                                new_tr_list.append(clean_sent)
                                        data = {'sentence': new_tr_list, 'label': [item_path.split("/")[-1] for i in range(len(new_tr_list))]}
                                        data_tr.append(pd.DataFrame(data)) 
                                elif 'dev' in sub_item_path:
                                        files_manager['dev_file'] = sub_item_path
                                        # data_d.append(pd.read_csv(files_manager['dev_file']))
                                        d_list = list(pd.read_csv(files_manager['dev_file'])['sentence'])
                                        new_d_list = []
                                        for sent in d_list:
                                                    clean_sent = self.clean_text(sent)
                                                    split_s = clean_sent.split()
                                                    if len(split_s) >=3 and len(split_s) <= 50:
                                                                new_d_list.append(clean_sent)
                                        data = {'sentence': new_d_list, 'label': [item_path.split("/")[-1] for i in range(len(new_d_list))]}
                                        data_d.append(pd.DataFrame(data)) 
                                elif 'test' in sub_item_path:
                                            files_manager['test_file'] = sub_item_path  
                                            data_te.append(pd.read_csv(files_manager['test_file']))
                                            t_list = list(pd.read_csv(files_manager['test_file'])['sentence'])
                                            new_t_list = []
                                            for sent in t_list:
                                                        clean_sent = self.clean_text(sent)
                                                        split_s = clean_sent.split()
                                                        if len(split_s) >=3 and len(split_s) <= 50:
                                                                    new_t_list.append(clean_sent)
                                            data = {'sentence': new_t_list, 'label': [item_path.split("/")[-1] for i in range(len(new_t_list))]}
                                            data_te.append(pd.DataFrame(data)) 
                                else:
                                    print("This file is unknow")
                                        
                                        
                                        
            self.df_train = pd.concat(data_tr, ignore_index = True, sort=False)
            self.df_dev = pd.concat(data_d, ignore_index = True, sort=False)
            self.df_test = pd.concat(data_te, ignore_index = True, sort=False)   

            self.df_train = self.df_train.sample(frac = 1)
            self.df_dev = self.df_dev.sample(frac = 1)
            self.df_test = self.df_test.sample(frac = 1)
            self.copy_df_test =  self.df_test.copy()

      def save_dataframes(self,lang):
                           #Save prepared data as csv file
                            self.df_train.to_csv(lang + 'train.csv', index=False)
                            self.df_dev.to_csv(  lang + 'dev.csv', index=False)
                            self.df_test.to_csv( lang + 'test.csv', index=False)

    


# In[ ]:


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# In[ ]:


def train(args, train_dataset, dev_dataset, labels, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    '''
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer__.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler__.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    '''

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    '''
    if os.path.exists(args.output_dir): #model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    '''
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    eval_fones = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type not in ["distilbert", "xlmroberta"]:
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # EVALUATE + EARLY STOPPING
        if global_step > 500 and args.n_gpu == 1: # and global_step % args.save_steps == 0:
            eval_results, _ = evaluate(args, model, tokenizer, labels, "dev", display_res=True)
            f1_step = round(eval_results["acc"], 5)
            eval_fones.append(f1_step)
            print("eval result: ", global_step, f1_step)


            max_f1 = max(eval_fones)

            if f1_step == max_f1:
                output_dir = args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                #torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                #logger.info("Saving optimizer and scheduler states to %s", output_dir)


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break



    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, mode, prefix="", display_res=False):
    # load_and_cache_examples(args, tokenizer, labels, mode='train')
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type not in ["distilbert", "xlmroberta"]:
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    eval_report = sklearn.metrics.classification_report(out_label_ids, preds,
                                                        labels=range(len(labels)),
                                                        target_names=labels,
                                                        output_dict=True
                                                        )

    results = {
        "loss": eval_loss,
        "precision": eval_report["weighted avg"]["precision"],
        "recall": eval_report["weighted avg"]["recall"],
        "f1": eval_report["weighted avg"]["f1-score"],
        "acc": sklearn.metrics.accuracy_score(out_label_ids, preds),
    }

    if not display_res:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, preds

def inference_eval( headlines, text,text_class, args, model, tokenizer, labels, mode, prefix="", display_res=False):
        # load_and_cache_examples(args, tokenizer, labels, mode='train')
        eval_dataset   =  load_inference_examples(args, headlines, text, text_class,  tokenizer, labels, mode=mode)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
        # multi-gpu eval
        #if args.n_gpu > 1:
        #    model = torch.nn.DataParallel(model)
    
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # print("  Num examples = %d", len(eval_dataset))
        # print("  Batch size = %d", args.eval_batch_size)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
    
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
    
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        eval_report = sklearn.metrics.classification_report(out_label_ids, preds,
                                                            labels=range(len(labels)),
                                                            target_names=labels,
                                                            output_dict=True
                                                            )
    
        results = {
            "loss": eval_loss,
            "precision": eval_report["weighted avg"]["precision"],
            "recall": eval_report["weighted avg"]["recall"],
            "f1": eval_report["weighted avg"]["f1-score"],
            "acc": sklearn.metrics.accuracy_score(out_label_ids, preds),
        }
    
        if not display_res:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
    
        return results, preds

def load_and_cache_examples(args, tokenizer, labels, mode='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        instances = read_instances_from_file(args.data_dir, mode)
        features = convert_instances_to_features_and_labels(instances, tokenizer, labels, args.max_seq_length)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if args.model_type != 'xlmroberta':
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def load_inference_examples(args, headlines, text,text_class,  tokenizer, labels, mode='test'):
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format('infer',
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length)
            ),
        )        
        
        instances = []
        # instances.append(Instance(df['headline'].iloc[i] +' '+df['text'].iloc[i], df['category'].iloc[i]))
        instances.append(Instance(headline +' '+ text, text_class))
        # covert to features
        features = convert_instances_to_features_and_labels(instances, tokenizer, labels, args.max_seq_length)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            
        if args.local_rank == 0 and not evaluate:
                torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        if args.model_type != 'xlmroberta':
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        else:
            all_token_type_ids = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset
            
    


# In[ ]:


def get_args():
    """
    Get training arguments
    """
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        #help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=False,
        help="The input model directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_finetune", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    return parser.parse_known_args()


# In[ ]:


# data_path = "./LID_DATA_VuK_NCHLT"
# create_data = Sort_CSV_into2()
# create_data.read_and_init('./CSV_Train_Test_Split_NCHLT_plus_Vuk/')
# create_data.save_dataframes('./LID_DATA_VuK_NCHLT/')


# In[ ]:


# data_path = data_path
# args, _ = get_args()
# args.data_dir = data_path  # to-change: supply data directory
# args.output_dir = "../../../../../ext_data/neo/CTEXT_LID/afroxlmrbase_lid/" # to-change: supply output directory
# args.model_type = "xlmroberta" #"bert"
# args.model_name_or_path = "Davlan/afro-xlmr-base" #"bert-base-multilingual-cased"
# args.max_seq_length = 200
# args.output_result = "test_result"
# args.output_prediction_file = "test_prediction"
# args.num_train_epochs = 2
# args.per_gpu_train_batch_size = 2
# args.save_steps = 10000
# args.seed = 1
# args.do_train = True
# args.do_eval = True
# args.do_predict = True


# In[ ]:


# def main(args):
#     if (
#         os.path.exists(args.output_dir)
#         and os.listdir(args.output_dir)
#         and args.do_train
#         and not args.overwrite_output_dir
#     ):
#         raise ValueError(
#             "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
#                 args.output_dir
#             )
#         )

#     # Setup distant debugging if needed
#     if args.server_ip and args.server_port:
#         # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
#         import ptvsd

#         print("Waiting for debugger attach")
#         ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
#         ptvsd.wait_for_attach()

#     # Setup CUDA, GPU & distributed training
#     if args.local_rank == -1 or args.no_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#         args.n_gpu = torch.cuda.device_count()
#     else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend="nccl")
#         args.n_gpu = 1
#     args.device = device

#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
#     )
#     logger.warning(
#         "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,",
#         args.local_rank,
#         device,
#         args.n_gpu,
#         bool(args.local_rank != -1),
#     )

#     # Set seed
#     set_seed(args)

#     labels = get_labels(args.labels)
#     num_labels = len(labels)


#     # Load pretrained model and tokenizer
#     if args.local_rank not in [-1, 0]:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

#     args.model_type = args.model_type.lower()
#     config_class, model_class, tokenizer_class = AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
#     config = config_class.from_pretrained(
#         args.config_name if args.config_name else args.model_name_or_path,
#         num_labels=num_labels,
#         id2label={str(i): label for i, label in enumerate(labels)},
#         label2id={label: i for i, label in enumerate(labels)},
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
#     tokenizer = tokenizer_class.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         do_lower_case=args.do_lower_case,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
#     model = model_class.from_pretrained(
#         args.model_name_or_path,
#         from_tf=bool(".ckpt" in args.model_name_or_path),
#         config=config,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )

#     if args.local_rank == 0:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

#     model.to(args.device)

#     logger.info("Training/evaluation parameters %s", args)

#     # Training
#     if args.do_train:
#         train_dataset = load_and_cache_examples(args, tokenizer, labels, mode="train")
#         dev_dataset = load_and_cache_examples(args, tokenizer, labels, mode="dev")
#         global_step, tr_loss = train(args, train_dataset, dev_dataset, labels, model, tokenizer)
#         logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


#     # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
#     if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
#         # Create output directory if needed
#         output_dir = args.output_dir
#         if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
#             os.makedirs(output_dir)

#         logger.info("Saving model checkpoint to %s", output_dir)
#         # Save a trained model, configuration and tokenizer using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         model_to_save = (
#             model.module if hasattr(model, "module") else model
#         )  # Take care of distributed/parallel training
#         model_to_save.save_pretrained(output_dir)
#         tokenizer.save_pretrained(output_dir)

#         # Good practice: save your training arguments together with the trained model
#         torch.save(args, os.path.join(output_dir, "training_args.bin"))

#         # Load a trained model and vocabulary that you have fine-tuned
#         #model = model_class.from_pretrained(args.output_dir)
#         #tokenizer = tokenizer_class.from_pretrained(args.output_dir)
#         #model.to(args.device)

#     # Evaluation
#     results = {}
#     if args.do_eval and args.local_rank in [-1, 0]:
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         checkpoints = [args.output_dir]
#         if args.eval_all_checkpoints:
#             checkpoints = list(
#                 os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
#             )
#             logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
#         logger.info("Evaluate the following checkpoints: %s", checkpoints)
#         for checkpoint in checkpoints:
#             global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
#             prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

#             model = model_class.from_pretrained(checkpoint)
#             model.to(args.device)
#             result, _ = evaluate(args, model, tokenizer, labels, mode='dev', prefix=prefix)
#             result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
#             results.update(result)


#     if args.do_predict and args.local_rank in [-1, 0]:
#         tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#         model = model_class.from_pretrained(args.output_dir)
#         model.to(args.device)
#         result, predictions = evaluate(args, model, tokenizer, labels, mode='test')
#         predictions = list(predictions)
#         id2label = {str(i): label for i, label in enumerate(labels)}

#         # Save results
#         output_test_results_file = os.path.join(args.output_dir, args.output_result+".txt")
#         with open(output_test_results_file, "w") as writer:
#             for key in sorted(result.keys()):
#                 writer.write("{} = {}\n".format(key, str(result[key])))

#         output_test_predictions_file = os.path.join(args.output_dir, args.output_prediction_file+".txt")
#         with open(output_test_predictions_file, "w", encoding='utf-8') as writer:
#             df = pd.read_csv(os.path.join(args.data_dir, "test.tsv"), sep='\t')
#             N = df.shape[0]

#             texts = list(df['headline'].values)
#             for i in range(N):
#                 output_line = texts[i] + "\t" + id2label[str(predictions[i])] + "\n"
#                 writer.write(output_line)
#             '''
#             with open(os.path.join(args.data_dir, "test.tsv"), "r", encoding='utf-8') as f:
#                 line_data = f.read()
#             line_data =  line_data.splitlines()
#             for l, line in enumerate(line_data):
#                 if l == 0:
#                     continue
#                 else:
#                     text_vals = line.strip().split("\t")
#                     if len(text_vals) < 2: text_vals += [7]
#                     text, label = text_vals
#                     output_line = text + "\t" + id2label[str(predictions[l-1])] + "\n"
#                     writer.write(output_line)
#             '''

#     return results

# if __name__ == "__main__":
#       main(args)


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        #help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=False,
        help="The input model directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_finetune", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--lang", default="nso", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


    data_path = "../LID_DATA_VuK_NCHLT"
    create_data = Sort_CSV_into2()
    create_data.read_and_init('../Csv_Train_test_split_NC/')
    create_data.save_dataframes('../LID_DATA_VuK_NCHLT/')
    
    args = parser.parse_args()
    args.data_dir = data_path  # to-change: supply data directory
    args.output_dir = args.output_dir # to-change: supply output directory
    args.model_type = args.model_type #"bert"
    args.model_name_or_path = args.model_name_or_path #"bert-base-multilingual-cased"
    args.max_seq_length = 200
    args.output_result = "test_result_" + str(args.seed)
    args.output_prediction_file = "test_prediction_" + str(args.seed)
    args.num_train_epochs = 10
    args.per_gpu_train_batch_size = 16
    args.save_steps = 10000
    args.do_train = True
    args.do_eval = True
    args.do_predict = True
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, mode="train")
        dev_dataset = load_and_cache_examples(args, tokenizer, labels, mode="dev")
        global_step, tr_loss = train(args, train_dataset, dev_dataset, labels, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        output_dir = args.output_dir
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        #model = model_class.from_pretrained(args.output_dir)
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        #model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, mode='dev', prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)


    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, mode='test')
        predictions = list(predictions)
        id2label = {str(i): label for i, label in enumerate(labels)}

        # Save results
        output_test_results_file = os.path.join(args.output_dir, args.output_result+".txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        output_test_predictions_file = os.path.join(args.output_dir, args.output_prediction_file+".txt")
        with open(output_test_predictions_file, "w", encoding='utf-8') as writer:
            df = pd.read_csv(os.path.join(data_path, "test.csv"), sep=',')
            N = df.shape[0]

            texts = list(df['sentence'].values)
            true_labels  = list(df['label'].values)
            for i in range(N):
                writer.write(f'The sentence: \t {texts[i]} \n')
                writer.write(f'Model prediction : \t {id2label[str(predictions[i])]}, True label: \t {true_labels[i]} \n')    
                writer.write(f'----------------------------------------------------------------------------------------------------- \n') 
            '''
            with open(os.path.join(args.data_dir, "test.tsv"), "r", encoding='utf-8') as f:
                line_data = f.read()
            line_data =  line_data.splitlines()
            for l, line in enumerate(line_data):
                if l == 0:
                    continue
                else:
                    text_vals = line.strip().split("\t")
                    if len(text_vals) < 2: text_vals += [7]
                    text, label = text_vals
                    output_line = text + "\t" + id2label[str(predictions[l-1])] + "\n"
                    writer.write(output_line)
            '''

    return results

if __name__ == "__main__":
      main()


# In[ ]:


# # Text to test goes here
# text_class = 'uncategorized'
# headline = ""
# text = ""
# print(text)


# In[ ]:


# # Setup CUDA, GPU & distributed training
# if args.local_rank == -1 or args.no_cuda:
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     args.n_gpu = torch.cuda.device_count()
# else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#     torch.cuda.set_device(args.local_rank)
#     device = torch.device("cuda", args.local_rank)
#     torch.distributed.init_process_group(backend="nccl")
#     args.n_gpu = 1
# args.device = device
# args.per_gpu_eval_batch_size  = 1
# args.per_gpu_train_batch_size = 1
# args.eval_batch_size = 1
# args.model_type = args.model_type.lower()
# labels  =  ['sports', 'health', 'technology', 'business', 'politics', 'entertainment', 'religion', 'uncategorized']
# id2label = {str(i): label for i, label in enumerate(labels)}
# config_class, model_class, tokenizer_class = AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
# tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
# model = model_class.from_pretrained(args.output_dir)
# model.to(args.device)
# result, predictions = inference_eval(headline, text, text_class, args, model, tokenizer, labels, mode='test')
# predictions = list(predictions)

# # Save results
# # output_test_results_file = os.path.join(args.output_dir, args.output_result+".txt")
# # with open(output_test_results_file, "w") as writer:
# #     for key in sorted(result.keys()):
# #         writer.write("{} = {}\n".format(key, str(result[key])))
# output_line = headline + text + "\t" + id2label[str(predictions[0])] + "\n"
# print(output_line)

