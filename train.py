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
from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel
from tokenizers import BertWordPieceTokenizer
from data_readers import filter_dataset, GPTDataset, NextActionDataset, NextActionSchema, ResponseGenerationDataset
from models import ActionBertModel, SchemaActionBertModel
from sklearn.metrics import f1_score

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--schema_path", type=str)
    parser.add_argument("--token_vocab_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--action_output_dir", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--task", type=str, choices=["action", "generation"])
    parser.add_argument("--use_schema", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--schema_max_seq_length", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--device", default=0, type=int, help="GPU device #")
    parser.add_argument("--max_grad_norm", default=-1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def get_topk_actions(model,
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
                    if type(sc_batch[key]) is list:
                        continue

                    sc_batch[key] = sc_batch[key].to(device)

            sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"])[1]
            sc_action_label = sc_batch["action"]
            sc_tasks = sc_batch["task"]

    model.eval()
    pred = []
    histories = []
    responses = []
    for batch in tqdm(eval_dataloader, desc="Generating Actions"):
        with torch.no_grad():
            # Move to GPU
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(device)

            if args.use_schema:
                action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                                 attention_mask=batch["attention_mask"],
                                                 token_type_ids=batch["token_type_ids"],
                                                 tasks=batch["tasks"],
                                                 sc_pooled_output=sc_pooled_output,
                                                 sc_tasks=sc_tasks,
                                                 sc_action_label=sc_action_label)
            else:
                action_logits, _ = model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"],
                                         token_type_ids=batch["token_type_ids"])

            # Argmax to get predictions
            action_preds = torch.topk(action_logits, k=3, dim=1)[1].tolist()

            pred += action_preds
            histories += batch["history"]
            responses += batch["response"]

    # Return actions
    return pred, histories, responses

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
                    if type(sc_batch[key]) is list:
                        continue

                    sc_batch[key] = sc_batch[key].to(device)

            sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"])[1]
            sc_action_label = sc_batch["action"]
            sc_tasks = sc_batch["task"]

    model.eval()
    pred = []
    true = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            # Move to GPU
            if torch.cuda.is_available():
                for key, val in batch.items():
                    if type(batch[key]) is list:
                        continue

                    batch[key] = batch[key].to(device)

            if task == "action":
                if args.use_schema:
                    action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                                     attention_mask=batch["attention_mask"],
                                                     token_type_ids=batch["token_type_ids"],
                                                     tasks=batch["tasks"],
                                                     sc_pooled_output=sc_pooled_output,
                                                     sc_tasks=sc_tasks,
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

def train(args, exp_setting=None):
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

    sc_tokenizer = BertWordPieceTokenizer(args.token_vocab_path,
                                       lowercase=True)
    sc_tokenizer.enable_padding(length=args.schema_max_seq_length)

    # Data readers
    if args.task == "action":
        dataset_initializer = NextActionDataset
    elif args.task == "generation":

        dataset_initializer = ResponseGenerationDataset
    else:
        raise ValueError("Not a valid task type: {}".format(args.task))

    dataset = dataset_initializer(args.data_path,
                                  tokenizer,
                                  args.max_seq_length,
                                  token_vocab_name)

    # Get the action to id mapping
    if args.task == "generation":
        action_dataset = NextActionDataset(args.data_path,
                                           tokenizer,
                                           args.max_seq_length,
                                           token_vocab_name)
        action_label_to_id = action_dataset.action_label_to_id
        actions = sorted(action_label_to_id, key=action_label_to_id.get)

    if exp_setting is not None:
        if "domain" in exp_setting:
            data_type = exp_setting.get("data_type")
            train_dataset = filter_dataset(dataset,
                                           data_type=data_type,
                                           percentage=1.0,
                                           domain=exp_setting.get("domain"),
                                           exclude=True,
                                           train=True)

            test_dataset = filter_dataset(dataset,
                                          data_type=data_type,
                                          percentage=1.0,
                                          domain=exp_setting.get("domain"),
                                          exclude=False,
                                          train=False)
        elif "task" in exp_setting:
            data_type = exp_setting.get("data_type")
            train_dataset = filter_dataset(dataset,
                                           data_type=data_type,
                                           percentage=1.0,
                                           task=exp_setting.get("task"),
                                           exclude=True,
                                           train=True)

            test_dataset = filter_dataset(dataset,
                                          data_type=data_type,
                                          percentage=1.0,
                                          task=exp_setting.get("task"),
                                          exclude=False,
                                          train=False)
        else:
            data_type = exp_setting.get("data_type")
            train_dataset = filter_dataset(dataset,
                                           data_type=data_type,
                                           percentage=0.8,
                                           train=True)

            test_dataset = filter_dataset(dataset,
                                          data_type=data_type,
                                          percentage=0.2,
                                          train=False)


    # Load the schema for the next action prediction
    schema = NextActionSchema(args.schema_path,
                              sc_tokenizer,
                              args.schema_max_seq_length,
                              dataset.action_label_to_id if args.task == "action" else action_label_to_id,
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
                                        pin_memory=True,
                                        shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.train_batch_size,
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

        if torch.cuda.is_available():
            model.to(args.device)
    elif args.task == "generation":
        if args.use_schema:
            action_model = SchemaActionBertModel(args.model_name_or_path,
                                                 dropout=args.dropout,
                                                 num_action_labels=len(action_label_to_id))
        else:
            action_model = ActionBertModel(args.model_name_or_path,
                                           dropout=args.dropout,
                                           num_action_labels=len(action_label_to_id))

        gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

        if torch.cuda.is_available():
            action_model.to(args.device)
            gpt_model.to(args.device)
    else:
        raise ValueError("Cannot instantiate model for task: {}".format(args.task))


    if args.task == "generation":
        # Load action model
        action_model.load_state_dict(torch.load(os.path.join(args.action_output_dir, "model.pt")))

        # Generate actions for both the train and test set
        train_preds, train_histories, train_responses = get_topk_actions(action_model, train_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)
        test_preds, test_histories, test_responses = get_topk_actions(action_model, test_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)

        # Convert preds to responses. If the schema does not contain the action - reply mapping, use the name of the action instead 
        # Eventually this should be manually fixed in the schema.
        actions = sorted(action_label_to_id, key=action_label_to_id.get)
        train_preds = [
            "[PRED] " + " [PRED] ".join([schema.action_to_response.get(p, actions[p]) for p in pred])
            for pred in train_preds
        ]
        test_preds = [
            "[PRED] " + " [PRED] ".join([schema.action_to_response.get(p, actions[p]) for p in pred])
            for pred in test_preds
        ]


        train_dataset = GPTDataset(tokenizer=gpt_tokenizer, 
                                   histories=train_histories,
                                   responses=train_responses,
                                   pred_responses=train_preds)

        test_dataset = GPTDataset(tokenizer=gpt_tokenizer, 
                                  histories=test_histories,
                                  responses=test_responses,
                                  pred_responses=test_preds)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=4,
                                      shuffle=True)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     pin_memory=True)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                outputs = gpt_model(batch.to(args.device), labels=batch.to(args.device))
        
                loss, logits = outputs[:2]                        
                loss.backward()

                epoch_loss += loss.item()
                optimizer.step()
                model.zero_grad()

            print("Epoch loss: {}".format(epoch_loss / num_batches))

    elif args.task == "action":
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
                        if type(batch[key]) is list:
                            continue

                        batch[key] = batch[key].to(args.device)

                # Train model
                if args.use_schema:
                    # Get schema batch and move to GPU
                    sc_batch = next(iter(schema_test_dataloader))

                    # Filter out only the relevant actions
                    relevant_inds = []
                    batch_actions = set(batch['action'].tolist())
                    batch_tasks = set(batch['tasks'][0])
                    for i,action in enumerate(sc_batch['action'].tolist()):
                        if action in batch_actions:
                            relevant_inds.append(i)
                            batch_actions.remove(action)

                    # Add random inds until we hit desired batch size
                    while len(relevant_inds) < 64:
                        ind = random.randint(0, len(sc_batch['task'])-1)
                        if ind in relevant_inds: 
                            continue

                        relevant_inds.append(ind)

                    # Filter out sc batch to only relevant inds
                    sc_batch = {
                        k: [v[i] for i in relevant_inds] if type(v) is list else v[relevant_inds]
                        for k,v in sc_batch.items()
                    }

                    if torch.cuda.is_available():
                        for key, val in sc_batch.items():
                            if type(sc_batch[key]) is list:
                                continue

                            sc_batch[key] = sc_batch[key].to(args.device)

                    _, loss = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"],
                                    tasks=batch["tasks"],
                                    action_label=batch["action"],
                                    sc_input_ids=sc_batch["input_ids"],
                                    sc_attention_mask=sc_batch["attention_mask"],
                                    sc_token_type_ids=sc_batch["token_type_ids"],
                                    sc_tasks=sc_batch["task"],
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

    if args.num_epochs == 0:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    else:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    score = evaluate(model, test_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)
    print("Best result for {}: Score: {}".format(args.task, score))
    return score

if __name__ == "__main__":
    args = read_args()
    print(args)

    domains = ['ride', 'trip', 'plane', 'spaceship', 'meeting', 'weather', 'party', 'doctor', 'trivia', 'apartment', 'restaurant', 'hotel', 'bank']
    tasks = ['hotel_service_request', 'bank_balance', 'weather', 'bank_fraud_report', 'party_rsvp', 'apartment_search', 'trivia', 'ride_book', 'apartment_schedule', 'hotel_book', 'ride_status', 'restaurant_search', 'doctor_schedule', 'doctor_followup', 'restaurant_book', 'plane_search', 'meeting_schedule', 'spaceship_life_support', 'party_plan', 'plane_book', 'spaceship_access_codes', 'hotel_search', 'trip_directions']

    #scores = []
    #for domain in domains:
    #    print("DOMAIN", domain)
    #    exp_setting = {"domain": domain, "data_type": "unhappy"}

    #    args.output_dir = args.output_dir + domain + "/"

    #    scores.append(train(args, exp_setting))
    #    print(scores)
    #    print(np.mean(scores))

    exp_setting = {"data_type": "happy"}
    score = train(args, exp_setting)
    print(score)


    #scores = []
    #for task in tasks:
    #    print("TASK", task)
    #    exp_setting = {"task": task, "data_type": "happy"}

    #    args.output_dir = args.output_dir + task + "/"

    #    scores.append(train(args, exp_setting))
    #    print(scores)
    #    print(np.mean(scores))
    #print(np.mean(scores))
