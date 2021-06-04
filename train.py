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
from data_readers import filter_dataset, get_entities, GPTDataset, NextActionDataset, NextActionSchema, ResponseGenerationDataset
from models import ActionBertModel, SchemaActionBertModel
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu
from sklearn.preprocessing import MultiLabelBinarizer

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

def id_em(pred, true):
    common = ['hello, how can i help?', 'thank you and goodbye.', 'is there anything else that i can do for you?']
    arr = [p==t for p,t in zip(pred,true) if t.lower() not in common]
    exact_match = sum(arr)/len(arr)
    return exact_match

entities = get_entities()
mlb = MultiLabelBinarizer()
mlb.fit([[e] for e in entities])
def entity_f1(pred, true):
    def _get_ents(arr):
        ents = []
        for utt in arr:
            ents.append([ent for ent in entities if ent in utt])
        return ents

    pred_ents = mlb.transform(_get_ents(pred))
    true_ents = mlb.transform(_get_ents(true))
    return f1_score(pred_ents, true_ents, average='weighted')

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

            sc_all_output, sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"])
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
                                                 sc_all_output=sc_all_output,
                                                 sc_tasks=sc_tasks,
                                                 sc_action_label=sc_action_label,
                                                 sc_full_graph=schema_dataloader.dataset.full_graph)
                # Argmax to get predictions
                action_preds = torch.topk(action_logits, k=1, dim=1)[1].tolist()
            else:
                action_preds = [0] * len(batch["history"])


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
    if args.use_schema and task == "action":
        with torch.no_grad():    
            sc_batch = next(iter(schema_dataloader))
            if torch.cuda.is_available():
                for key, val in sc_batch.items():
                    if type(sc_batch[key]) is list:
                        continue

                    sc_batch[key] = sc_batch[key].to(device)

            sc_all_output, sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"])
            sc_action_label = sc_batch["action"]
            sc_tasks = sc_batch["task"]

    pred = []
    true = []
    sentence = []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            if task == "action":
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
                                                     sc_all_output=sc_all_output,
                                                     sc_pooled_output=sc_pooled_output,
                                                     sc_tasks=sc_tasks,
                                                     sc_action_label=sc_action_label,
                                                     sc_full_graph=schema_dataloader.dataset.full_graph)
                else:
                    action_logits, _ = model(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"],
                                             token_type_ids=batch["token_type_ids"])

                # Argmax to get predictions
                action_preds = torch.argmax(action_logits, dim=1).cpu().tolist()

                pred += action_preds
                true += batch["action"].cpu().tolist()
                sentence += [tokenizer.decode(e.tolist(), skip_special_tokens=False).replace(" [PAD]", "") for e in batch["input_ids"]]
            elif task == "generation":
                batch = batch.to(device)

                # Find start index
                st_ind = 2257
                start_ind = batch[0].tolist().index(2257)+3
                input_batch = batch[:,:start_ind]

                generated = model.generate(input_batch, max_length=start_ind + 50)

                def _get_outputs(ids):
                    words = tokenizer.decode(ids[0].tolist()).split()
                    start_ind = words.index("[START]") if "[START]" in words else 0
                    end_ind = words.index("[END]") if "[END]" in words else -1
                    output = " ".join(words[start_ind+1:end_ind])
                    return output

                generated_output = _get_outputs(generated)
                true_output = _get_outputs(batch)

                pred.append(generated_output)
                true.append(true_output)

    # Perform evaluation
    if task == "action":
        acc = sum(p == t for p,t in zip(pred, true))/len(pred)
        f1 = f1_score(true, pred, average='weighted')
        print(acc, f1)

        id_map = eval_dataloader.dataset.action_label_to_id
        label_map = sorted(id_map, key=id_map.get)
        print("INCORRECT ==========================================================")
        for i in range(len(true)):
            if pred[i] != true[i]:
                print(sentence[i] + "\n", label_map[true[i]], label_map[pred[i]])
        print("CORRECT ==========================================================")
        for i in range(len(true)):
            if pred[i] == true[i]:
                print(sentence[i] + "\n", label_map[true[i]], label_map[pred[i]])


        return f1, acc
    elif task == "generation":
        #exact_match = sum([p==t for p,t in zip(pred,true)])/len(true)
        bleu = corpus_bleu([[t.split()] for t in true], [p.split() for p in pred], weights=(0,0,0,1.0))
        id_exact_match = id_em(pred, true)
        ef1 = entity_f1(pred,true)

        print(bleu, id_exact_match, ef1)
        return bleu, id_exact_match, ef1

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
        #args.num_epochs = 0
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
    #dataset.examples = [e for e in dataset if 'weather' in e['tasks']]

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
    train_dataset.examples = sorted(train_dataset.examples, key=lambda i:i['tasks'])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=False,
                                  num_workers=0)

    schema_train_dataloader = DataLoader(dataset=schema,
                                         batch_size=min(len(schema), args.train_batch_size),
                                         pin_memory=True,
                                         shuffle=True)

    schema_test_dataloader = DataLoader(dataset=schema,
                                        batch_size=len(schema),
                                        pin_memory=True,
                                        shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.train_batch_size,
                                 pin_memory=True)

    print("Training set size", len(train_dataloader.dataset))
    print("Testing set size", len(test_dataloader.dataset))
    print("Experimental Setting", exp_setting)

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
        gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
        gpt_model.resize_token_embeddings(len(gpt_tokenizer))
        if torch.cuda.is_available():
            action_model.to(args.device)
            gpt_model.to(args.device)
    else:
        raise ValueError("Cannot instantiate model for task: {}".format(args.task))


    if args.task == "generation":
        if args.use_schema:
            # Load action model
            action_model.load_state_dict(torch.load(os.path.join(args.action_output_dir, "model.pt")))

            # Generate actions for both the train and test set
            train_preds, train_histories, train_responses = get_topk_actions(action_model, train_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)
            test_preds, test_histories, test_responses = get_topk_actions(action_model, test_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)

            # Convert preds to responses. If the schema does not contain the action - reply mapping, use the name of the action instead 
            # Eventually this should be manually fixed in the schema.
            actions = sorted(action_label_to_id, key=action_label_to_id.get)
            #train_preds = ["[PRED] " + schema.action_to_response[action_label_to_id[e['action']]] for e in train_dataloader.dataset ]
            train_preds = [action_label_to_id[e['action']] for e in train_dataloader.dataset]
            ignore_inds = set([i for i,e in enumerate(train_preds) if e not in schema.action_to_response])

            train_preds = [
                "[PRED] " + schema.action_to_response[p] for i,p in enumerate(train_preds) if i not in ignore_inds]
            

            train_histories = [e for i,e in enumerate(train_histories) if i not in ignore_inds]
            train_responses = [e for i,e in enumerate(train_responses) if i not in ignore_inds]

            #train_preds = [
            #    "[PRED] " + " [PRED] ".join([schema.action_to_response.get(p, actions[p]) for p in pred])
            #    for pred in train_preds
            #]
            test_preds = [
                "[PRED] " + " [PRED] ".join([schema.action_to_response.get(p, actions[p]) for p in pred])
                for pred in test_preds
            ]
        else:
            # Doesn't actually run action prediction
            _, train_histories, train_responses = get_topk_actions(action_model, train_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)
            _, test_histories, test_responses = get_topk_actions(action_model, test_dataloader, schema_test_dataloader, tokenizer, task=args.task, args=args)

            train_preds = [""]*len(train_histories)
            test_preds = [""]*len(test_histories)


        train_dataset = GPTDataset(tokenizer=gpt_tokenizer, 
                                   histories=train_histories,
                                   responses=train_responses,
                                   pred_responses=train_preds)

        test_dataset = GPTDataset(tokenizer=gpt_tokenizer, 
                                  histories=test_histories,
                                  responses=test_responses,
                                  pred_responses=test_preds)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True)

        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     pin_memory=True)

        optimizer = AdamW(gpt_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        model = gpt_model
        tokenizer = gpt_tokenizer
        for epoch in trange(args.num_epochs, desc="Epoch"):
            gpt_model.train()
            epoch_loss = 0
            num_batches = 0
            for batch in tqdm(train_dataloader):
                outputs = gpt_model(batch.to(args.device), labels=batch.to(args.device))
        
                loss, logits = outputs[:2]                        

                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

                loss.backward()

                if args.grad_accum <= 1 or num_batches % args.grad_accum == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    gpt_model.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            print("Epoch loss: {}".format(epoch_loss / num_batches))

    elif args.task == "action":
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        best_score = -1

        model.tokenizer = tokenizer
        for epoch in trange(args.num_epochs, desc="Epoch"):
            model.train()
            epoch_loss = 0
            num_batches = 0

            model.zero_grad()
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
                    batch_action_tasks = [(batch['action'][i].item(), batch['tasks'][i]) for i in range(len(batch['action']))]
                    for i,action in enumerate(sc_batch['action'].tolist()):
                        if (action, sc_batch['task'][i]) in batch_action_tasks:
                            relevant_inds.append(i)
                            #batch_actions.remove(action)

                    # Add random inds until we hit desired batch size
                    #while len(relevant_inds) < min(len(schema), 64):
                    #    ind = random.randint(0, len(sc_batch['task'])-1)
                    #    if ind in relevant_inds: 
                    #        continue

                    #    relevant_inds.append(ind)

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
                                    sc_action_label=sc_batch["action"],
                                    sc_full_graph=schema.full_graph)
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
    tasks = ['hotel_service_request', 'bank_balance', 'weather', 'bank_fraud_report', 'party_rsvp', 'apartment_search', 'trivia', 'ride_book', 'apartment_schedule', 'hotel_book', 'ride_status', 'restaurant_search', 'doctor_schedule', 'doctor_followup', 'restaurant_book', 'plane_search', 'meeting_schedule',  'party_plan', 'plane_book', 'spaceship_access_codes', 'hotel_search', 'trip_directions']

    # TODO: removed spaceship lifesupport

    #scores = []
    #for domain in domains:
    #    print("DOMAIN", domain)
    #    exp_setting = {"domain": domain, "data_type": "happy"}

    #    args.action_output_dir = args.action_output_dir + domain + "/"
    #    args.output_dir = args.output_dir + domain + "/"

    #    scores.append(train(args, exp_setting))
    #    print(scores)
    #    print(np.mean(scores))

    #exp_setting = {"data_type": "happy"}
    #score = train(args, exp_setting)
    #print(score)
    #import pdb; pdb.set_trace()


    scores = []
    #old_scores = [(0.1251822880329993, 0.008695652173913044, 0.4422287223225904), (0.2030153366259423, 0.2334293948126801, 0.5154030787129109), (0.09997489644784736, 0.03586321934945788, 0.524756762789609), (0.13554532880138379, 0.055315471045808126, 0.5218483431511356), (0.12952421111813903, 0.043560606060606064, 0.5096936399549333), (0.06351615001083892, 0.025806451612903226, 0.2910344199776204), (0.013483511127957976, 0.0057684384013185, 0.32428577205905135), (0.2229493128323843, 0.03289473684210526, 0.5159379065636661)]
    #old_scores = [(0.13810549037760272, 0.002898550724637681, 0.5338795693511099), (0.2403602589361103, 0.23631123919308358, 0.483329522331328), (0.09254531529231554, 0.041701417848206836, 0.33598698870880594), (0.1554872000833363, 0.14001728608470182, 0.5034600823266697), (0.12799401197604787, 0.026515151515151516, 0.4804521855742575), (0.018745357409085013, 0.0, 0.2661509871806041), (0.005401486362824454, 0.0, 0.3822273025173828), (0.14336143657089925, 0.027046783625730993, 0.537998560369733)]


    #old_scores = [0.370145609907578, 0.6936094742462428, 0.5733870590228247, 0.45211109471009503, 0.37460121412747927, 0.38071057973117794, 0.3757033547898641, 0.4347140329420736, 0.560474140681303]
    #old_scores = [0.4181504089603669, 0.6665471093947086, 0.628755986954502, 0.3765951990641796, 0.42640688701075397, 0.3308334938135245]
    #old_scores = [0.3055877654220461, 0.52494652771303, 0.5458386347906655, 0.43991664511573825, 0.24908908018826842, 0.25510913052932116, 0.15745862252990223, 0.06364949767977886, 0.3060432849050964, 0.5046084486467972, 0.48505465068648285, 0.44144129710330693, 0.39130821002152216]

    #old_scores = [0.37390275848779864, 0.5919105079512341, 0.5552268398963037, 0.4326607236595678, 0.2826485909464619, 0.30522023576371404, 0.11256835957894906, 0.24244531003723505, 0.40038272491291355, 0.5382107311640719, 0.44551830900005107, 0.4227847203142083, 0.46961304936006326, 0.7538503411190403, 0.45100884710050376, 0.33314844218625644, 0.5343383933970168, 0.5408618827576072, 0.6387976672887278, 0.5575414172950797]

    #old_scores = [0.4627303097385849, 0.6657940755842202, 0.45476291060702423, 0.48067097820820065, 0.4356467356080915, 0.4544275471482722, 0.46795481359900093, 0.5930224296927255, 0.6201425111328526, 0.5541689558045297, 0.536076750710897, 0.25711207213338994, 0.6209493174497825, 0.7144009623569185, 0.5585668878580609, 0.49326759419712946]
    #old_scores = [0.5053615185098657, 0.6747939031969111, 0.5906308038966452, 0.6054052219734716, 0.49193551441139766]
    #old_scores = [0.13091714163142734, 0.05936899081061183, 0.23842181459317163, 0.1632424215180666, 0.2505202563791628, 0.33738299841839703, 0.18259675430705485, 0.021114744240237536, 0.8193427790094152, 0.2522276037804978, 0.3190781612117253, 0.2851415637548379, 0.2672976920651196]
    old_scores = []


    #old_scores = [(0.47027603689589903, 0.48284313725490197), (0.438210718298963, 0.4860759493670886), (0.627604402070792, 0.6735751295336787), (0.5985664296132878, 0.5810055865921788), (0.31238035757550753, 0.32870771899392887), (0.42882991026213535, 0.5365079365079365), (0.4719889081464009, 0.5240474703310432), (0.5105024721363549, 0.5308219178082192), (0.24318212769144956, 0.31970260223048325), (0.5974995098743199, 0.6113821138211382), (0.5539429856503028, 0.6178861788617886), (0.4276582765053761, 0.4119318181818182), (0.6339501413534022, 0.7003610108303249), (0.7073728649261628, 0.7276995305164319), (0.5907700122606176, 0.5786163522012578), (0.5319049080103434, 0.5545977011494253)]

    
    orig_dir = args.action_output_dir
    orig_output_dir = args.output_dir
    for i,task in enumerate(domains):
        print("TASK", task)
        exp_setting = {"domain": task, "data_type": "happy"}

        args.action_output_dir = orig_dir + task + "/"
        args.output_dir = orig_output_dir + task + "/"

        if i < len(old_scores):
            scores.append(old_scores[i])
        else:
            scores.append(train(args, exp_setting))
        print(scores)

        if args.task == "generation":
            print(np.mean([e[0] for e in scores]))
            print(np.mean([e[1] for e in scores]))
            print(np.mean([e[2] for e in scores]))
        else:
            print("f1", np.mean([e[0] for e in scores]))
            print("acc", np.mean([e[1] for e in scores]))

    #print(np.mean(scores))
