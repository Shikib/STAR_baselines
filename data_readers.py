import copy
import json
import numpy as np
import os
import pickle
import torch

from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm

def get_entities(path="STAR/dialogues/"):
    # Load conversations
    convs = [json.load(open(path+fn)) for fn in os.listdir(path)]
    entities = []
    for conv in convs:
        for event in conv['Events']:
            if 'PrimaryItem' not in event:
                continue

            for entity in event['PrimaryItem'].values():
                if type(entity) is list:
                    entities += entity
                else:
                    entities.append(entity)

    entities = list(set([str(e) for e in entities]))
    return entities

def filter_dataset(dataset,
                   data_type="happy", # happy, unhappy, multitask
                   domain=None,
                   task=None,
                   exclude=False,
                   percentage=1.0,
                   train=True):
    """
    Split the dataset according to the criteria

    - data_type:
        - happy: Only the happy dialogs
        - unhappy: Only the happy + unhappy dialogs (no multitask)
        - multitask: All the dialogs

    - domain:
        - Requirements:
            - task should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - task:
        - Requirements:
            - domain should be None
            - data_type should be happy/unhappy
        - If exclude = True
            - Exclude dialog of this domain
        - If exclude = False
            - Include ONLY dialogs of this domain

    - percentage:
        - Take only a certain percentage of the available data (after filters)
        - If train = True
            - Take the first [percentage]% of the data
        - If train = False:
            - Take the last [percentage]% of the data
    """
    examples = dataset.examples

    # Filter based on happy/unhappy/multitask
    if data_type == "happy":
        examples = [ex for ex in examples if ex.get("happy")]
    elif data_type == "unhappy":
        examples = [ex for ex in examples if not ex.get("multitask")]

    # Filter based on domain
    if domain is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert task is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["domains"][0] != domain]
        else:
            examples = [ex for ex in examples if ex["domains"][0] == domain]

    # Filter based on task
    if task is not None:
        assert data_type in ["happy", "unhappy"], "For zero-shot experiments, can only use single-task data"
        assert domain is None, "Can filter by domain OR task, not both"

        if exclude:
            examples = [ex for ex in examples if ex["tasks"][0] != task]
        else:
            examples = [ex for ex in examples if ex["tasks"][0] == task]

    # Split based on percentage
    all_dialog_ids = sorted(list(set([ex['dialog_id'] for ex in examples])))
    if train:
        selected_ids = all_dialog_ids[:int(len(all_dialog_ids)*percentage)]
    else:
        selected_ids = all_dialog_ids[-int(len(all_dialog_ids)*percentage):]

    selected_ids = set(selected_ids)
    examples = [ex for ex in examples if ex['dialog_id'] in  selected_ids]

    # Filter out only the relevant keys for each example (so that DataLoader doesn't complain)
    keys = ["input_ids", "attention_mask", "token_type_ids", "action", "tasks", "history", "response"]
    examples = [{k:v for k,v in ex.items() if k in keys} for ex in examples]
    for ex in examples:
        ex["tasks"] = ex["tasks"][0]

    # Return new dataset
    new_dataset = copy.deepcopy(dataset)
    new_dataset.examples = examples
    return new_dataset

class NextActionSchema(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 action_label_to_id,
                 vocab_file_name):
        # Check if cached pickle file exists
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        cached_path = os.path.join(data_dirname, "schema_action_cached")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.examples, self.action_to_response, self.full_graph = pickle.load(f)

            return None

        # Read all of the JSON files in the data directory
        tasks = [
            json.load(open(data_path + fn + "/" + fn + ".json")) for fn in os.listdir(data_path)
        ]

        self.action_to_response = {}

        # Iterate over the schemas and get (1) the prior states and (2) the 
        # next actions.
        node_to_utt = {}
        self.examples = []
        for task in tqdm(tasks):
            # Get the graph
            graph = task['graph']

            # For every edge in the graph, get examples of transfer to each action
            for prev,action in graph.items():
                if False and prev in task['r_graph']:
                    sys_utt = '[wizard] ' + task['replies'][task['r_graph'][prev]] + ' [SEP] '
                    utterance = '[user] ' + sys_utt + task['replies'][prev] + ' [SEP]'
                else:
                    utterance = '[user] ' + task['replies'][prev] + ' [SEP]'

                node_to_utt[prev] = utterance 

                # For next action prediction, we can normalize the diff query types
                #if action in ['query_check', 'query_book']:
                #    action = 'query'

                if action not in action_label_to_id:
                    continue

                action_label = action_label_to_id[action]
                self.action_to_response[action_label] = task['replies'][action] 
                encoded = tokenizer.encode(utterance)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "action": action_label,
                    "task": task['task'],
                })

        self.full_graph = {}
        #for task in tqdm(tasks):
        #    graph = task['graph']
        #    r_graph = task['r_graph']

        #    # all of the nodes that are possible at a given turn
        #    nodes = list(graph.keys())

        #    # modify the reverse graph to map from sys turn to next sys turn
        #    f_graph = defaultdict(list)
        #    for u_t, s_t in r_graph.items():
        #        f_graph[s_t].append(graph[u_t])

        #    # bfs to get the full graph
        #    task_graph = defaultdict(list)
        #    for node in nodes:
        #        active = [graph[node]]
        #        for i in range(len(nodes)+1):
        #            if len(active) == 0:
        #                break

        #            task_graph[node_to_utt[node]].append([action_label_to_id[action] for action in active if action in action_label_to_id])

        #            new_active = []
        #            for active_node in active:
        #                if active_node in f_graph:
        #                    new_active += f_graph[active_node]

        #            active = list(set(new_active))

        #    self.full_graph[task['task']] = task_graph

        # Write to cache
        with open(cached_path, "wb+") as f:
            pickle.dump([self.examples, self.action_to_response, self.full_graph], f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class NextActionDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 vocab_file_name):
        # Check if cached pickle file exists
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        cached_path = os.path.join(data_dirname, "action_cached")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.action_label_to_id, self.examples = pickle.load(f)

            return None

        # Read all of the JSON files in the data directory
        conversations = [
            json.load(open(data_path + fn)) for fn in os.listdir(data_path)
        ]

        # Iterate over the conversations and get (1) the dialogs and (2) the 
        # actions for all wizard turns.
        self.examples = []
        self.action_label_to_id = {}
        for conv in tqdm(conversations):
            # History (so far) for this dialog
            history = ""
            for i,utt in enumerate(conv['Events']):
                # NOTE: Ground truth action labels only exist when wizard picks suggestion. 
                # We skip all custom utterances for action prediction.
                if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion']:
                    # Tokenize history
                    processed_history = ' '.join(history.strip().split()[:-1])
                    encoded_history  = tokenizer.encode(processed_history)

                    # Convert action label to id
                    query_label = 'query'
                    if 'ActionLabel' not in utt:
                        query_check = 'Check' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        query_book = 'Book' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        # In case of a bug, if both book and check are on - we treat it as a check.
                        if query_check:
                            query_label = 'query_check' 
                        elif query_book:
                            query_label = 'query_book' 

                    action_label = utt['ActionLabel'] if 'ActionLabel' in utt else query_label
                    if action_label not in self.action_label_to_id:
                        self.action_label_to_id[action_label] = len(self.action_label_to_id)
                    action_label_id = self.action_label_to_id[action_label]

                    # Include metadata 
                    domains = conv['Scenario']['Domains']
                    tasks = [e['Task'] for e in conv['Scenario']['WizardCapabilities']]
                    happy = conv['Scenario']['Happy']
                    multitask = conv['Scenario']['MultiTask']

                    # Add to data
                    self.examples.append({
                        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
                        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
                        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
                        "action": action_label_id,
                        "dialog_id": conv['DialogueID'],
                        "domains": domains,
                        "tasks": tasks,
                        "happy": happy,
                        "multitask": multitask,
                        "orig_history": processed_history,
                        "orig_action": action_label,
                    })

                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # If it's a knowledge base item, format it as a string
                    if utt['Action'] == 'return_item':
                        utt_text = "[RESULT] "
                        if 'Item' not in utt:
                            utt_text += "NO RESULT"
                        else:
                            for key,val in utt['Item'].items():
                                utt_text += "{} = {} ; ".format(key, val)

                    if utt_text != "":
                        history += "[{}] {} [SEP] ".format(utt['Agent'], utt_text.strip())

        # Write to cache
        with open(cached_path, "wb+") as f:
            pickle.dump([self.action_label_to_id, self.examples], f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class ResponseGenerationDataset(Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_length,
                 vocab_file_name):
        # Check if cached pickle file exists
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        cached_path = os.path.join(data_dirname, "response_cached")
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

            return None

        # Read all of the JSON files in the data directory
        conversations = [
            json.load(open(data_path + fn)) for fn in os.listdir(data_path)
        ]

        # Iterate over the conversations and get (1) the dialogs and (2) the 
        # actions for all wizard turns.
        self.examples = []
        for conv in tqdm(conversations):
            # History (so far) for this dialog
            history = ""
            for i,utt in enumerate(conv['Events']):
                # NOTE: Ground truth action labels only exist when wizard picks suggestion. 
                # We skip all custom utterances for action prediction.
                print("SKIPPING ALL UTTERS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion', 'utter']:
                if utt['Agent'] == 'Wizard' and utt['Action'] in ['query', 'pick_suggestion']:
                    # Current utterance
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # Tokenize history
                    processed_history = history.strip()
                    encoded_history = tokenizer.encode(processed_history)

                    # Include metadata 
                    domains = conv['Scenario']['Domains']
                    tasks = [e['Task'] for e in conv['Scenario']['WizardCapabilities']]
                    happy = conv['Scenario']['Happy']
                    multitask = conv['Scenario']['MultiTask']

                    # Convert action label to id
                    query_label = 'query'
                    if 'ActionLabel' not in utt:
                        query_check = 'Check' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        query_book = 'Book' in [e['RequestType'][1:-1] for e in utt['Constraints'] if 'RequestType' in e]
                        # In case of a bug, if both book and check are on - we treat it as a check.
                        if query_check:
                            query_label = 'query_check' 
                        elif query_book:
                            query_label = 'query_book' 

                    action_label = utt['ActionLabel'] if 'ActionLabel' in utt else query_label

                    # Add to data
                    self.examples.append({
                        "history": processed_history,
                        "response": utt_text,
                        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
                        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
                        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
                        "dialog_id": conv['DialogueID'],
                        "domains": domains,
                        "tasks": tasks,
                        "happy": happy,
                        "multitask": multitask,
                        "action": action_label,
                    })

                # Process and concatenate to history
                if utt['Agent'] in ['User', 'Wizard', 'KnowledgeBase']:
                    utt_text = ""

                    # If there's text, just use it directly
                    if utt['Action'] in ['pick_suggestion', 'utter']:
                        utt_text = utt['Text']

                    # If it's a knowledge base query, format it as a string
                    if utt['Action'] == 'query':
                        utt_text = "[QUERY] "
                        for constraint in utt['Constraints']:
                            key = list(constraint.keys())[0]
                            val = constraint[key]

                            utt_text += "{} = {} ; ".format(key, val)

                    # If it's a knowledge base item, format it as a string
                    if utt['Action'] == 'return_item':
                        utt_text = "[RESULT] "
                        if 'Item' not in utt:
                            utt_text += "NO RESULT"
                        else:
                            for key,val in utt['Item'].items():
                                utt_text += "{} = {} ; ".format(key, val)

                    if utt_text != "":
                        history += "[{}] {} [SEP] ".format(utt['Agent'], utt_text.strip())

        # Write to cache
        with open(cached_path, "wb+") as f:
            pickle.dump(self.examples, f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class GPTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        histories,
	responses,
	pred_responses,
        block_size=512,
    ):
        self.examples = []

        for history, response, pred_response in tqdm(zip(histories, responses, pred_responses)):
            # Consider only portions of each input (for a total of 512)
            example = " ".join(history.split()[-200:]) + " " + pred_response + " [START] " + response + " [END]"
            example = " ".join(example.split())
            tokenized = tokenizer(example, padding="max_length", max_length=400)
            self.examples.append(tokenized['input_ids'][-400:])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

