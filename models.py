import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any

class ActionBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path,
                 dropout,
                 num_action_labels):
        super(ActionBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_action_labels = num_action_labels
        self.action_classifier = nn.Linear(self.bert_model.config.hidden_size, num_action_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                action_label=None):
        pooled_output = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[1]
        action_logits = self.action_classifier(self.dropout(pooled_output))

        # Compute losses if labels provided
        if action_label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(action_logits.view(-1, self.num_action_labels), action_label.type(torch.long))
        else:
            loss = torch.tensor(0)

        return action_logits, loss

class SchemaActionBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path,
                 dropout,
                 num_action_labels):
        super(SchemaActionBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_action_labels = num_action_labels
        self.action_classifier = nn.Linear(self.bert_model.config.hidden_size, num_action_labels)
        self.p_schema = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                tasks,
                action_label,
                sc_input_ids,
                sc_attention_mask,
                sc_token_type_ids,
                sc_tasks,
                sc_action_label,
                sc_full_graph):
        all_output, pooled_output = self.bert_model(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        action_logits = self.action_classifier(self.dropout(pooled_output))

        sc_all_output, sc_pooled_output = self.bert_model(input_ids=sc_input_ids,
                                                          attention_mask=sc_attention_mask,
                                                          token_type_ids=sc_token_type_ids)



        #dists = pooled_output.mm(sc_pooled_output.t())
        #probs = F.softmax(dists, dim=-1)

        # Zero out any attention across different tasks
        #for i in range(dists.size(0)):
        #    for j in range(dists.size(1)):
        #        if tasks[i] != sc_tasks[j]:
        #            dists[i,j] = -1e10

        all_output_flat = all_output.view(-1, all_output.size(-1))
        i_probs = F.softmax(all_output_flat.mm(sc_all_output.view(-1, 768).t()), dim=-1).view(all_output_flat.size(0), -1, sc_input_ids.size(-1)).sum(dim=-1)

        probs = i_probs.view(input_ids.size(0), -1, i_probs.size(-1)).mean(dim=1)

        action_probs = torch.zeros(probs.size(0), self.num_action_labels).cuda().scatter_add(-1, sc_action_label.unsqueeze(0).repeat(probs.size(0), 1), probs)

        sc_prob = F.sigmoid(self.p_schema(pooled_output))

        action_lps = torch.log(action_probs+1e-10)
        #action_lps = torch.log(action_probs*sc_prob + (1 - sc_prob)*F.softmax(action_logits, dim=-1))

        # Compute losses if labels provided
        if action_label is not None:
            loss_fct = NLLLoss()
            loss = loss_fct(action_lps.view(-1, self.num_action_labels), action_label.type(torch.long))
        else:
            loss = torch.tensor(0)

        return action_lps, loss

    def predict(self,
                input_ids,
                attention_mask,
                token_type_ids,
                tasks,
                sc_all_output,
                sc_pooled_output,
                sc_tasks,
                sc_action_label,
                sc_full_graph):

        #action_lps = self.forward(input_ids, attention_mask, token_type_ids, tasks, None, None, None, None, sc_tasks, sc_action_label, sc_full_graph)[0]

        all_output, pooled_output = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        action_logits = self.action_classifier(self.dropout(pooled_output))
        #dists = pooled_output.mm(sc_pooled_output.t())
        #
        #probs = F.softmax(dists, dim=-1)

        all_output_flat = all_output.view(-1, all_output.size(-1))
        i_probs = F.softmax(all_output_flat.mm(sc_all_output.view(-1, 768).t()), dim=-1).view(all_output_flat.size(0), -1, sc_all_output.size(-2)).sum(dim=-1)

        probs = i_probs.view(input_ids.size(0), -1, i_probs.size(-1)).mean(dim=1)


        # Zero out any attention across different tasks
        for i in range(probs.size(0)):
            for j in range(probs.size(1)):
                if tasks[i] != sc_tasks[j]:
                    probs[i,j] = 0


        action_probs = torch.zeros(probs.size(0), self.num_action_labels).cuda().scatter_add(-1, sc_action_label.unsqueeze(0).repeat(probs.size(0), 1), probs)

        sc_prob = F.sigmoid(self.p_schema(pooled_output))

        #action_lps = torch.log(action_probs)#*sc_prob + (1 - sc_prob)*F.softmax(action_logits, dim=-1))
        #action_lps = torch.log(action_probs*sc_prob + (1 - sc_prob)*F.softmax(action_logits, dim=-1))
        action_lps = torch.log(action_probs*sc_prob)

        return action_lps, 0

#        # Full graph probabilities [SLOW]
#        user_turns = []
#        user_batch_inds = []
#        user_fwd_lens = []
#        user_turn_weights = []
#        for i in range(input_ids.size(0)):
#            all_utts = all_output[i][input_ids[i] == 102]
#
#            weights = torch.softmax(torch.arange(int((len(all_utts)+1)/2)).float(), dim=-1).cuda()
#
#            for j in range(int((len(all_utts)+1)/2)):
#                ind = -(j*2 + 1)
#
#                user_turns.append(all_utts[ind])
#                user_batch_inds.append(i)
#                user_fwd_lens.append(j)
#                #user_turn_weights.append(weights[j])
#                user_turn_weights.append(1.0 if j == 0 else 0)
#
#        # Iterate by task [usually will only be 1, but need to handle edge cases]
#        action_probs = torch.zeros(input_ids.size(0), self.num_action_labels).cuda()
#        for task in set(tasks):
#            # All schema bert outputs
#            schema_utts_inp = []
#            schema_utts_mask = []
#            schema_utts_typ = []
#            schema_fwd_actions = []
#            for schema_utt,act_list in sc_full_graph[task].items():
#                encoded = self.tokenizer.encode(schema_utt)
#
#                schema_utts_inp.append(torch.tensor(encoded.ids)[-50:])
#                schema_utts_mask.append(torch.tensor(encoded.attention_mask)[-50:])
#                schema_utts_typ.append(torch.tensor(encoded.type_ids)[-50:])
#
#                schema_fwd_actions.append(act_list)
#
#            schema_input_ids = torch.stack(schema_utts_inp).cuda()
#            schema_mask = torch.stack(schema_utts_mask).cuda()
#            schema_token_type_ids = torch.stack(schema_utts_typ).cuda()
#            sc_pooled_output = self.bert_model(input_ids=schema_input_ids,
#                                               attention_mask=schema_mask,
#                                               token_type_ids=schema_token_type_ids)[1]
#
#            # Get probabilities
#            user_turns_stack = torch.stack(user_turns)
#            turn_dists = user_turns_stack.mm(sc_pooled_output.t())
#
#            for i in range(len(user_turns)):
#                batch_ind = user_batch_inds[i]
#                fwd_len = user_fwd_lens[i]
#                turn_weight = user_turn_weights[i]
#
#                if tasks[batch_ind] != task:
#                    continue
#
#                # iterate over the schema and get all fwd lens
#                valid_dists = []
#                fwd_actions = []
#                for j in range(len(schema_fwd_actions)):
#                    if fwd_len < len(schema_fwd_actions[j]) and len(schema_fwd_actions[j][fwd_len]) > 0:
#                        valid_dists.append(turn_dists[i,j])
#                        fwd_actions.append(schema_fwd_actions[j][fwd_len])
#
#                turn_probs = torch.softmax(torch.stack(valid_dists), dim=-1)
#
#                for j in range(len(turn_probs)):
#                    for act in fwd_actions[j]:
#                        action_probs[batch_ind][act] += turn_weight * turn_probs[j] / len(fwd_actions[j])
