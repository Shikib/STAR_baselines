import argparse
import json
import numpy as np
import os
import random
import torch

from typing import Any, Dict, Tuple
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW
from tokenizers import BertWordPieceTokenizer
from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel
from sklearn.metrics import f1_score

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--schema_path", type=str)
    parser.add_argument("--token_vocab_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--task", type=str, choices=["action", "generation"])
    parser.add_argument("--use_schema", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--device", default=0, type=int, help="GPU device #")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def evaluate(model,
             eval_dataloader,
             schema_dataloader,
             tokenizer,
             task,
             device=0,
             args=None): 
    # Get schema pooled outputs
    if args.use_schema:
        with torch.no_grad():    
            sc_batch = next(iter(schema_dataloader))
            if torch.cuda.is_available():
                for key, val in sc_batch.items():
                    sc_batch[key] = sc_batch[key].to(device)

            sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"])[1]
            sc_action_label = sc_batch["action"]

    model.eval()
    pred = []
    true = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # Move to GPU
            if torch.cuda.is_available():
                for key, val in batch.items():
                    batch[key] = batch[key].to(device)

            if task == "action":
                if args.use_schema:
                    action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                                     attention_mask=batch["attention_mask"],
                                                     token_type_ids=batch["token_type_ids"],
                                                     sc_pooled_output=sc_pooled_output,
                                                     sc_action_label=sc_action_label)
                else:
                    action_logits, _ = model(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"],
                                             token_type_ids=batch["token_type_ids"])

                # Argmax to get predictions
                action_preds = torch.argmax(action_logits, dim=1).cpu().tolist()

                pred += action_preds
                true += batch["action"].cpu().tolist()

    # Perform evaluation
    if task == "action":
        acc = sum(p == t for p,t in zip(pred, true))/len(pred)
        f1 = f1_score(true, pred, average='weighted')
        print(acc, f1)
        return f1

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    #special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(labels == 0, dtype=torch.bool), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.token_to_id("[MASK]")

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random].cuda()

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def train(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.num_epochs == 0:
        # This means we're evaluating. Don't create the directory.
        pass
    else:
        raise Exception("Directory {} already exists".format(args.output_dir))

    # Dump arguments to the checkpoint directory, to ensure reproducability.
    if args.num_epochs > 0:
        json.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.json'), "w+"))
        torch.save(args, os.path.join(args.output_dir, "run_args"))

    # Configure tokenizer
    token_vocab_name = os.path.basename(args.token_vocab_path).replace(".txt", "")
    tokenizer = BertWordPieceTokenizer(args.token_vocab_path,
                                       lowercase=True)
    tokenizer.enable_padding(length=args.max_seq_length)

    # Data readers
    if args.task == "action":
        dataset_initializer = NextActionDataset
    else:
        raise ValueError("Not a valid task type: {}".format(args.task))

    dataset = dataset_initializer(args.data_path,
                                  tokenizer,
                                  args.max_seq_length,
                                  token_vocab_name)

    train_dataset = filter_dataset(dataset,
                                   data_type="happy",
                                   percentage=0.8,
                                   train=True)

    test_dataset = filter_dataset(dataset,
                                  data_type="happy",
                                  percentage=0.2,
                                  train=False)

    # Load the schema for the next action prediction
    schema = NextActionSchema(args.schema_path,
                              tokenizer,
                              30,
                              dataset.action_label_to_id,
                              token_vocab_name)

    # Data loaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=0)

    schema_train_dataloader = DataLoader(dataset=schema,
                                         batch_size=args.train_batch_size,
                                         pin_memory=True,
                                         shuffle=True)

    schema_test_dataloader = DataLoader(dataset=schema,
                                        batch_size=len(schema),
                                        pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 pin_memory=True)

    # Load model
    if args.task == "action":
        if args.use_schema:
            model = SchemaActionBertModel(args.model_name_or_path,
                                          dropout=args.dropout,
                                          num_action_labels=len(train_dataset.action_label_to_id))
        else:
            model = ActionBertModel(args.model_name_or_path,
                                    dropout=args.dropout,
                                    num_action_labels=len(train_dataset.action_label_to_id))
    else:
        raise ValueError("Cannot instantiate model for task: {}".format(args.task))

    if torch.cuda.is_available():
        model.to(args.device)

    # Train
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    best_score = -1
    for epoch in trange(args.num_epochs, desc="Epoch"):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_dataloader): 
            num_batches += 1

            # Transfer to gpu
            if torch.cuda.is_available():
                for key, val in batch.items():
                    batch[key] = batch[key].to(args.device)

            # Train model
            if args.task == "action":
                if args.use_schema:
                    # Get schema batch and move to GPU
                    sc_batch = next(iter(schema_train_dataloader))
                    if torch.cuda.is_available():
                        for key, val in sc_batch.items():
                            sc_batch[key] = sc_batch[key].to(args.device)

                    _, loss = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"],
                                    action_label=batch["action"],
                                    sc_input_ids=sc_batch["input_ids"],
                                    sc_attention_mask=sc_batch["attention_mask"],
                                    sc_token_type_ids=sc_batch["token_type_ids"],
                                    sc_action_label=sc_batch["action"])
                else:
                    _, loss = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"],
                                    action_label=batch["action"])

                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

                loss.backward()
                epoch_loss += loss.item()

            if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()

        print("Epoch loss: {}".format(epoch_loss / num_batches))

    # Evaluate on test set
    print("Loading up best model for test evaluation...")
    score = evaluate(model, test_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print("Best result for {}: Score: {}".format(args.task, score))
    return score

if __name__ == "__main__":
    args = read_args()
    args.seed = 42
    print(args)
    score = train(args)
    print(score)
