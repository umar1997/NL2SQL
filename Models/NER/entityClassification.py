##################################### Import Libraries
import numpy as np 
import pandas as pd

import time
from tqdm import tqdm, trange

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score

from dataPreparation import Data_Preprocessing
from dataProcessing import Data_Processing

import argparse
from packaging import version


pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)



class Training:
    def __init__(self, HYPER_PARAMETERS):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case = False)

        self.model = None

        self.HYPER_PARAMETERS = HYPER_PARAMETERS

        dataPreprocessing = Data_Preprocessing()
        tokens, tags, tag_idx, tag_values = dataPreprocessing.main()
        print('     Data Preprocessing Successfully Completed')

        self.tokens, self.tags, self.tag_idx, self.tag_values = tokens, tags, tag_idx, tag_values

        dataProc = Data_Processing(tokens, tags, self.tokenizer, tag_idx, self.HYPER_PARAMETERS)
        input_ids, tags, attention_masks = dataProc.getProcessedData()
        # print('Inputs: {}'.format(input_ids[0]))
        # print('Tags: {}'.format(tags[0]))
        # print('Attention Mask: {}'.format(attention_masks[0]))
        # print('Lengths Matching: {}, {}, {}'.format(len(input_ids[0]), len(tags[0]), len(attention_masks[0])))
        print('     Data Processing Successfully Completed')

        self.input_ids, self.tags, self.attention_masks = input_ids, tags, attention_masks
        print('     Class Initialized!')

    def train_test_split(self,):
        # Get Train Test Split for Inputs and Tags
        tr_input, val_input, tr_tag, val_tag = train_test_split(self.input_ids,self.tags,random_state=45,test_size=.15) 
        # Get Split for NER
        tr_masks, val_masks, _, _ = train_test_split(self.attention_masks, self.input_ids, random_state=45, test_size=.15)

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
        # Getting BERT's pretrained Token Classification model
        model = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        num_labels=len(self.tag_idx),
        output_attentions = False,
        output_hidden_states = False)

        model.cuda()
        self.model = model

        # print(self.model)

    def optimizer_and_lr_scheduler(self, train_dataloader):
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
            eps= self.HYPER_PARAMETERS['EPSILON']
        )
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * self.HYPER_PARAMETERS['EPOCHS']

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        return optimizer, scheduler


    def training_and_validation(self, train_dataloader, valid_dataloader, optimizer, scheduler):

        loss_values, validation_loss_values = [], []
        E = 1
        for _ in trange(self.HYPER_PARAMETERS['EPOCHS'], desc= "Epoch \n"):
            print('\n')
            print('     Epoch #{}'.format(E))
        
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
                scheduler.step() # Update learning_rate
                
            avg_train_loss = total_loss/len(train_dataloader) 
            print('     Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))
            
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

            validation_loss_values.append(avg_eval_loss)
            
            pred_tags = [self.tag_values[p_i] for p, l in zip(predictions, true_labels)
                        for p_i, l_i, in zip(p,l)if self.tag_values[l_i] !='PAD']
            
            valid_tags = [self.tag_values[l_i]for l in true_labels
                        for l_i in l if self.tag_values[l_i] !='PAD']
            
            print('     Validation Accuracy: {}%'.format(accuracy_score(pred_tags,valid_tags)*100))
            print('     Validation F-1 Score:{}'.format(f1_score([pred_tags], [valid_tags])))

            stop = time.time()
            print('     Epoch #{} Duration:{}'.format(E, stop-start))
            E+=1
            print('-'*20)
            time.sleep(3)

    def run(self,):

        tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks = self.train_test_split()
        tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks = self.convert_to_tensors(tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks)
        train_dataloader, valid_dataloader = self.data_loader(tr_input, val_input, tr_tag, val_tag, tr_masks, val_masks)
        self.model_init()
        print('     Model Initialized!')
        optimizer, scheduler = self.optimizer_and_lr_scheduler(train_dataloader)
        print('     Starting Training. . .')
        self.training_and_validation(train_dataloader, valid_dataloader, optimizer, scheduler)

    

import warnings
if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    # parser = argparse.ArgumentParser()

    # # add arguments
    # parser.add_argument("--model_type", default='bert', type=str, required=True,
    #                     help="valid values: bert, _, _")
    # parser.add_argument("--data_dir", type=str, required=True,
    #                     help="The input data directory.")
    # parser.add_argument("--seed", default=3, type=int,
    #                     help='random seed')
    # parser.add_argument("--max_seq_length", default=80, type=int,
    #                     help="maximum number of tokens allowed in each sentence")
    # parser.add_argument("--batch_size", default=16, type=int,
    #                     help="The batch size for training and evaluation.")
    # parser.add_argument("--learning_rate", default=3e-5, type=float,
    #                     help="The initial learning rate for optimizer.")
    # parser.add_argument("--num_epochs", default=3, type=int,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")

    # global_args = parser.parse_args()


    HYPER_PARAMETERS = {
        "MAX_LEN" : 80, # Max Length of the sentence
        "BATCH_SIZE" : 16,
        "EPOCHS" : 3,
        "MAX_GRAD_NORM" : 1.0,
        "LEARNING_RATE" : 3e-5,
        "EPSILON" : 1e-8

        # "MAX_LEN" : global_args.max_seq_length, 
        # "BATCH_SIZE" : global_args.batch_size,
        # "EPOCHS" : global_args.num_epochs,
        # "MAX_GRAD_NORM" : global_args.max_grad_norm,
        # "LEARNING_RATE" : global_args.learning_rate,
        # "EPSILON" : global_args.adam_epsilon
        # "TEST_SPLIT": 0.15,
        # "RANDOM_SEED": 42
    }


    print('Entity Classification Training')
    print('------------------------------')
    train = Training(HYPER_PARAMETERS)
    train.run()
    print('------------------------------')


script = """
python entityClassification.py \
    --model_type bert \
    --data_dir ./../../Data/Chia_w_scope_data.csv \
    --max_seq_length 80 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 3 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0

"""
