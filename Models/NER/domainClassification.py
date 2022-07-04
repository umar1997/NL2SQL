import os
import sys

import json
import numpy as np 
import pandas as pd

import time
from tqdm import tqdm, trange

dir_path = os.path.dirname(os.path.realpath('./../'))
sys.path.append(dir_path)

import torch
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score

from dataPreparation import Data_Preprocessing
from dataProcessing import Data_Processing
from addedLayers import CustomModel
from evaluationTools import Eval


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



class Training:
    def __init__(self, checkpoint_tokenizer, checkpoint_model, HYPER_PARAMETERS, logger_progress, logger_results):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

        self.checkpoint_tokenizer = checkpoint_tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_tokenizer, do_lower_case = False)

        self.model = None
        self.checkpoint_model = checkpoint_model

        self.logger_progress = logger_progress
        self.logger_results = logger_results
        self.HYPER_PARAMETERS = HYPER_PARAMETERS

        # MAKE LABELS GENERAL AND ADDED LAYERS
        # -----------------------------------
        # self.general_labels = True
        self.general_labels = self.HYPER_PARAMETERS["GENERAL_LABELS"]
        self.added_layers = self.HYPER_PARAMETERS["ADDED_LAYERS"]
        # self.added_layers = False


        dataPreprocessing = Data_Preprocessing()
        tokens, tags_names, tag_idx, tag_values = dataPreprocessing.main()
        if self.general_labels == True:
            self.tokens, self.tags_names, self.tag_idx, self.tag_values = self.makeGeneralLabels(tokens, tags_names, tag_idx, tag_values)
        else:
            self.tokens, self.tags_names, self.tag_idx, self.tag_values = tokens, tags_names, tag_idx, tag_values
            # Tokens and Tags are list of lists
            # tag_idx is dictionary of encoding and tag value
            # tag_value is list of all tags (domains) 
        self.logger_progress.critical('Data Preprocessing Completed')

        dataProc = Data_Processing(self.tokens, self.tags_names, self.tokenizer, self.tag_idx, self.HYPER_PARAMETERS)
        input_ids, tags, attention_masks = dataProc.getProcessedData()
        self.input_ids, self.tags, self.attention_masks = input_ids, tags, attention_masks
        # print('Inputs: {}'.format(input_ids[0]))
        # print('Tags: {}'.format(tags[0]))
        # print('Attention Mask: {}'.format(attention_masks[0]))
        # print('Lengths Matching: {}, {}, {}'.format(len(input_ids[0]), len(tags[0]), len(attention_masks[0])))
        self.logger_progress.critical('Data Processing Completed')

        self.logger_progress.critical('Training Initialized!')

    def makeGeneralLabels(self, tokens, tags_names, tag_idx, tag_values):
        for i, sent in enumerate(tags_names):
            for j, tn in enumerate(sent):
                if tn.startswith('B-') or tn.startswith('I-'):
                    tags_names[i][j] = tn[2:]
        for j, t in enumerate(tag_values):
            if t.startswith('B-') or t.startswith('I-'):
                tag_values[j] = t[2:]
        tag_values = list(set(tag_values))
        tag_idx = {t: i for i, t in enumerate(tag_values)}
        # print(tag_idx)
        
        return tokens, tags_names, tag_idx, tag_values


    def train_test_split(self,):
        # Get Train -  Val/Test Split for Inputs and Tags 
        tr_input, val_test_input, tr_tag, val_test_tag = train_test_split(self.input_ids,self.tags,random_state=self.HYPER_PARAMETERS['RANDOM_SEED'],test_size=self.HYPER_PARAMETERS['TEST_SPLIT']) 
        tr_masks, val_test_masks, _, _ = train_test_split(self.attention_masks, self.input_ids, random_state=self.HYPER_PARAMETERS['RANDOM_SEED'], test_size=self.HYPER_PARAMETERS['TEST_SPLIT'])

        # Get Val - Test Split for Inputs and Tags
        val_input, test_input, val_tag, test_tag = train_test_split(val_test_input,val_test_tag,random_state=self.HYPER_PARAMETERS['RANDOM_SEED'],test_size=0.5) 
        val_masks, test_masks, _, _ = train_test_split(val_test_masks, val_test_input, random_state=self.HYPER_PARAMETERS['RANDOM_SEED'], test_size=0.5)

        test_dict = {
            "Random Seed" : self.HYPER_PARAMETERS['RANDOM_SEED'],
            "Train Size" : (1- self.HYPER_PARAMETERS['TEST_SPLIT']),
            "Val & Test Size" : self.HYPER_PARAMETERS['TEST_SPLIT']/2,
            "Train Input Length" : len(tr_input),
            "Train Tag Length" : len(tr_tag),
            "Train Mask Length" : len(tr_masks),
            "Validation Input Length" : len(val_input),
            "Validation Tag Length" : len(val_tag),
            "Validation Mask Length" : len(val_masks),
            "Test Input Length" : len(test_input),
            "Test Tag Length" : len(test_tag),
            "Test Mask Length" : len(test_masks),
            "Test Input" : test_input,
            "Test Tag" : test_tag,
            "Test Mask" : test_masks
        }

        # Dumping JSON
        json_dump= json.dumps(test_dict, cls=NumpyEncoder)
        with open("test_split.json", "w") as f:
            json.dump(json_dump, f)

        # How to read the json is in Extra/Train_Val_Test_Splits

        return tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks

    def convert_to_tensors(self, tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks):

        tr_input = torch.tensor(tr_input)
        val_input = torch.tensor(val_input)

        tr_tag = torch.tensor(tr_tag)
        val_tag = torch.tensor(val_tag)

        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)

        # print('Input Train Size: {}, {}, {}:'.format(len(tr_masks),len(tr_input), len(tr_tag)))
        # print('Input Val Size: {}, {}, {}:'.format(len(val_masks),len(val_input), len(val_tag)))

        return tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks
    
    def data_loader(self, tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks):

        train_data = TensorDataset(tr_input, tr_masks, tr_tag)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.HYPER_PARAMETERS['BATCH_SIZE'])

        valid_data = TensorDataset(val_input, val_masks, val_tag)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.HYPER_PARAMETERS['BATCH_SIZE'])

        return train_dataloader, valid_dataloader

    def model_init(self,):

        if self.added_layers: # Added Layers is True
            model = AutoModelForTokenClassification.from_pretrained(
            self.checkpoint_model,
            num_labels=len(self.tag_idx),
            output_attentions = False,
            output_hidden_states = False)
        else: # Added Layers is False
            model = CustomModel(checkpoint=self.checkpoint_model, num_labels=len(self.tag_idx))

        model.cuda()
        self.model = model

        # print(self.model)

    def optimizer_and_lr_scheduler(self, train_dataloader):

        if self.HYPER_PARAMETERS['OPTIMIZER'] == 'AdamW':
            FULL_FINETUNING = True
            if FULL_FINETUNING: # Fine Tuning
                param_optimizer = list(self.model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                optimizer_grouped_parameters = [
                    # Setting Weight Decay Rate 0.01 if it isnt bias, gamma and beta
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                    'weight_decay_rate': 0.01},
                    # If it is set to 0.0
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                    'weight_decay_rate': 0.0}
                ]
            else: # Non Fine Tuning
                param_optimizer = list(self.model.classifier.named_parameters())
                optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

            # Optimizer
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr= self.HYPER_PARAMETERS['LEARNING_RATE'],
                eps= self.HYPER_PARAMETERS['EPSILON'],
                # weight_decay=0.01 Doing stuff above for weight decay
            )

            # Create the learning rate scheduler.
            scheduler = False

        elif self.HYPER_PARAMETERS['OPTIMIZER'] == 'SGD':
            
            # Optimizer
            optimizer = SGD(
                self.model.parameters(),
                lr=self.HYPER_PARAMETERS['LEARNING_RATE'], # 0.1 usually
                momentum=0.9, # 0.9 usually
                dampening=0,
                weight_decay=0,
                nesterov=False
            )

            # Total number of training steps is number of batches * number of epochs.
            total_steps = len(train_dataloader) * self.HYPER_PARAMETERS['EPOCHS']

            # Create the learning rate scheduler.
            if self.HYPER_PARAMETERS['LR_SCHEDULER'] == 'LinearWarmup':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )
            
            elif self.HYPER_PARAMETERS['LR_SCHEDULER'] == 'LRonPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.1, 
                    patience=10, # Number of epochs with no improvement after which learning rate will be reduced
                    threshold=0.0001, 
                    threshold_mode='rel', 
                    cooldown=0, 
                    min_lr=0, 
                    eps=1e-08, 
                    verbose=False)



        return optimizer, scheduler


    def training_and_validation(self, train_dataloader, valid_dataloader, optimizer, scheduler):

        loss_values, validation_loss_values = [], []
        E = 1
        for _ in trange(self.HYPER_PARAMETERS['EPOCHS'], desc= "Epoch \n"):
            print('\n')
            print('     Epoch #{}'.format(E))
            self.logger_results.info('Epoch #{}'.format(E))
        
            start = time.time()

            self.model.train()
            total_loss=0 # Reset at each Epoch
            
            ###################### TRAINING
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch # Mantained the order for both train_data/val_data
                
                self.model.zero_grad() # Clearing previous gradients for each epoch
                
                outputs = self.model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels) # Forward pass
                
                loss = outputs[0]
                loss.backward() # Getting the loss and performing backward pass
                
                total_loss += loss.item() # Tracking loss
                
                # Preventing exploding grads
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.HYPER_PARAMETERS['MAX_GRAD_NORM'])
                
                optimizer.step() # Updates parameters
                if scheduler:
                    scheduler.step() # Update learning_rate

            avg_train_loss = total_loss/len(train_dataloader) 
            print('     Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))
            self.logger_results.info('Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))

            loss_values.append(avg_train_loss) # Storing loss values to plot learning curve
            ###################### VALIDATION
            self.model.eval()
            
            eval_loss = 0
            predictions, true_labels = [], []
            
            for batch in valid_dataloader:
                batch = tuple(t.to(self.device)for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad(): # No backprop
                    outputs = self.model(b_input_ids, token_type_ids =None,
                                attention_mask=b_input_mask, labels=b_labels)
                    
                logits = outputs[1].detach().cpu().numpy() # Getting Probabilities for Prediction Classes
                label_ids = b_labels.to('cpu').numpy() # Golden Labels
                
                loss = outputs[0]
                eval_loss += loss.item()

                predictions.extend([list(p) for p in np.argmax(logits, axis=2)]) # Taking Max among Prediction Classes
                true_labels.extend(label_ids)

            avg_eval_loss = eval_loss / len(valid_dataloader)
            print('     Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))
            self.logger_results.info('Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))

            validation_loss_values.append(avg_eval_loss)
            
            pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels)
                        for p_i, l_i, in zip(p,l)if self.tag_values[l_i] !='PAD']
            
            valid_tags = [self.tag_values[l_i]for l in true_labels
                        for l_i in l if self.tag_values[l_i] !='PAD']
            
            print('     Validation Accuracy: {}%'.format(accuracy_score(pred_tags,valid_tags)*100))
            print('     Validation F-1 Score:{}'.format(f1_score([pred_tags], [valid_tags])))

            self.logger_results.info('Validation Accuracy: {}%  |  Validation F-1 Score:{}'.format(accuracy_score(pred_tags,valid_tags)*100, f1_score([pred_tags], [valid_tags])))

            stop = time.time()
            print('     Epoch #{} Duration:{}'.format(E, stop-start))
            self.logger_results.info('Duration: {}\n'.format(stop-start))
            E+=1
            print('-'*20)
            time.sleep(3)


        CR, labels, acc = Eval(predictions, true_labels, self.tag_values)
        self.logger_results.info('Classification Report:')
        self.logger_results.info('\n{}'.format(CR))
        self.logger_results.info('Labels: {}'.format(labels))
        self.logger_results.info('Accuracies: {}'.format(acc))

    def run(self,):

        tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks = self.train_test_split()
        tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks = self.convert_to_tensors(tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks)
        train_dataloader, valid_dataloader = self.data_loader(tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks)
        self.model_init()
        self.logger_progress.critical('Model Initialized!')
        optimizer, scheduler = self.optimizer_and_lr_scheduler(train_dataloader)
        self.logger_progress.critical('Starting Training. . .\n')
        self.training_and_validation(train_dataloader, valid_dataloader, optimizer, scheduler)
        self.logger_progress.critical('Training Completed!')


