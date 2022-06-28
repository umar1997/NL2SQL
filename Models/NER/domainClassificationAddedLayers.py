import os
import sys

import numpy as np 
import pandas as pd

import time
from tqdm import tqdm, trange

import argparse
from packaging import version

dir_path = os.path.dirname(os.path.realpath('./../'))
sys.path.append(dir_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, accuracy_score

from dataPreparation import Data_Preprocessing
from dataProcessing import Data_Processing
from log import get_logger


pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)


# https://jovian.ai/rajbsangani/emotion-tuned-sarcasm
class CustomModel(nn.Module):
    def __init__(self,checkpoint,num_labels): 
        super(CustomModel,self).__init__() 
        self.num_labels = num_labels 
        input_dim = 768
        self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        # self.dropout = nn.Dropout(0.1) 
        # self.classifier = nn.Linear(768,num_labels)
        self.fcx = nn.Linear(input_dim, int(input_dim*0.7))
        self.fcy = nn.Linear(int(input_dim*0.7), int(input_dim*0.3))
        self.fcz = nn.Linear(int(input_dim*0.3), self.num_labels)


    def forward(self, input_ids=None, attention_mask=None,labels=None):

        # https://discuss.pytorch.org/t/how-to-confirm-parameters-of-frozen-part-of-network-are-not-being-updated/142482
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # https://huggingface.co/docs/transformers/main_classes/output

        # print(outputs) gives "class transformers.modeling_outputs.BaseModelOutputWithPooling"
        # ( last_hidden_state, pooler_output, hidden_states, attentions)

        # print(outputs[0].shape, end='-----------------------------------')          # torch.Size([16, 80, 768])
        # print(outputs[1].shape, end='-----------------------------------')          # torch.Size([16, 768])
        # print(outputs[2][0].shape, end='-----------------------------------')       # torch.Size([16, 80, 768])
        # print(outputs[2][1].shape, end='-----------------------------------')       # torch.Size([16, 80, 768])
        # print(outputs[3][0].shape, end='-----------------------------------')       # torch.Size([16, 12, 80, 80])
        # print(outputs[3][0].shape, end='-----------------------------------')       # torch.Size([16, 12, 80, 80])

        # sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state
        # # print(sequence_output.shape) # torch.Size([16, 80, 768])
        # # print(sequence_output[:,:,:].view(-1,768).shape) # torch.Size([1280, 768])

        # logits = self.classifier(sequence_output[:,:,:].view(-1,768)) # calculate losses
        # # Logits Shape = torch.Size([1280, 24])
        # # Labels Shape = torch.Size([16, 80])
        # # print(labels.view(-1).shape) # Labels Shape = torch.Size([1280])

        sequence_output = outputs[0]
        x = sequence_output[:,:,:].view(-1,768)
        x = self.fcx(x)
        x = F.relu(x)
        x = self.fcy(x)
        x = F.relu(x)
        logits = self.fcz(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


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

        general_labels = True

        dataPreprocessing = Data_Preprocessing()
        tokens, tags_names, tag_idx, tag_values = dataPreprocessing.main()
        if general_labels == True:
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
        
        return tokens, tags_names, tag_idx, tag_values

    def train_test_split(self,):
        # Get Train Test Split for Inputs and Tags 
        tr_input, val_input, tr_tag, val_tag = train_test_split(self.input_ids,self.tags,random_state=self.HYPER_PARAMETERS['RANDOM_SEED'],test_size=self.HYPER_PARAMETERS['TEST_SPLIT']) 
        # Get Split for NER
        tr_masks, val_masks, _, _ = train_test_split(self.attention_masks, self.input_ids, random_state=self.HYPER_PARAMETERS['RANDOM_SEED'], test_size=self.HYPER_PARAMETERS['TEST_SPLIT'])

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

#         model = AutoModelForTokenClassification.from_pretrained(
#         self.checkpoint_model,
#         num_labels=len(self.tag_idx),
#         output_attentions = False,
#         output_hidden_states = False)
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
                params,
                lr=self.HYPER_PARAMETERS['LEARNING_RATE'], # 0.1 usually
                momentum=0, # 0.9 usually
                dampening=0,
                weight_decay=0,
                nesterov=False,
                maximize=False
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
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels) # Forward pass

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
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    
                logits = outputs[1].detach().cpu().numpy() # Getting Probabilities for Prediction Classes
                label_ids = b_labels.to('cpu').numpy() # Golden Labels
                
                loss = outputs[0]
                eval_loss += loss.item()

                
                predictions.extend([p for p in np.argmax(logits, axis=1)]) # Taking Max among Prediction Classes
                true_labels.extend(list(label_ids.reshape(-1)))
            

            avg_eval_loss = eval_loss / len(valid_dataloader)
            print('     Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))
            self.logger_results.info('Average Val Loss For Epoch {}: {}'.format(E, avg_eval_loss))

            validation_loss_values.append(avg_eval_loss)
            
            pred_tags = [self.tag_values[p] for p, l in zip(predictions, true_labels) if self.tag_values[l] !='PAD']
            
            valid_tags = [self.tag_values[l]for l in true_labels if self.tag_values[l] !='PAD']

            
            print('     Validation Accuracy: {}%'.format(accuracy_score(pred_tags,valid_tags)*100))
            print('     Validation F-1 Score:{}'.format(f1_score([pred_tags], [valid_tags])))

            self.logger_results.info('Validation Accuracy: {}%  |  Validation F-1 Score:{}'.format(accuracy_score(pred_tags,valid_tags)*100, f1_score([pred_tags], [valid_tags])))

            stop = time.time()
            print('     Epoch #{} Duration:{}'.format(E, stop-start))
            self.logger_results.info('Duration: {}\n'.format(stop-start))
            E+=1
            print('-'*20)
            time.sleep(3)

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


import warnings
if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_type", default='bert-base-cased', type=str, required=True,
                        help="valid values: bert-base-cased, dmis-lab/biobert-v1.1, _")
    parser.add_argument("--tokenizer_type", default='bert-base-cased', type=str, required=True,
                        help="valid values: bert-base-cased, _, _")
    parser.add_argument("--data_dir", type=str,
                        help="The input data directory.")
    parser.add_argument("--seed", default=3, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=80, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The batch size for training and evaluation.")
    parser.add_argument("--val_split", default=0.15, type=float,
                        help="Train test split for validation split.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default='AdamW', type=str,
                        help="valid values: AdamW, SGD")
    parser.add_argument("--scheduler", default='None', type=str,
                        help="valid values: LinearWarmup, LRonPlateau")
    parser.add_argument("--log_folder", default='./Log_Files/', type=str,
                        help="Name of log folder.")
    parser.add_argument("--log_file", default='sample.log', type=str,
                        help="Name of log file.")

    global_args = parser.parse_args()


    HYPER_PARAMETERS = {

        "MAX_LEN" : global_args.max_seq_length, 
        "BATCH_SIZE" : global_args.batch_size,
        "EPOCHS" : global_args.num_epochs,
        "MAX_GRAD_NORM" : global_args.max_grad_norm,
        "LEARNING_RATE" : global_args.learning_rate,
        "EPSILON" : global_args.adam_epsilon,
        "TEST_SPLIT": global_args.val_split,
        "RANDOM_SEED": global_args.seed,
        "OPTIMIZER": global_args.optimizer,
        "LR_SCHEDULER": global_args.scheduler,
    }

    seed = global_args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    file_name = global_args.log_folder + global_args.log_file
    logger_meta = get_logger(name='META', file_name=file_name, type='meta')
    logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
    logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')

    logger_meta.warning("TOKENIZER_TYPE: {}".format(global_args.tokenizer_type))
    logger_meta.warning("MODEL_TYPE: {}".format(global_args.model_type))
    for i, (k, v) in enumerate(HYPER_PARAMETERS.items()):
        if i == (len(HYPER_PARAMETERS) - 1):
            logger_meta.warning("{}: {}\n".format(k, v))
        else:
            logger_meta.warning("{}: {}".format(k, v))

    print('Entity Classification Training')
    print('------------------------------')
    train = Training(global_args.tokenizer_type, global_args.model_type, HYPER_PARAMETERS, logger_progress, logger_results)
    train.run()


script = """
python domainClassificationAddedLayers.py \
    --model_type dmis-lab/biobert-v1.1 \
    --tokenizer_type dmis-lab/biobert-v1.1 \
    --data_dir ./../../Data/Chia_w_scope_data.csv \
    --max_seq_length 80 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 10 \
    --val_split 0.15 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler LinearWarmup \
    --log_folder ./Log_Files/ \
    --log_file added_notfreeze_general_biobert.log
"""

####################################### Different Models
# dmis-lab/biobert-large-cased-v1.1

# dmis-lab/biobert-v1.1
# bert-base-cased
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# fidukm34/biobert_v1.1_pubmed-finetuned-ner-finetuned-ner
# sciarrilli/biobert-base-cased-v1.2-finetuned-ner (Token Classifcation)
# emilyalsentzer/Bio_ClinicalBERT

# bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16
# algoprog/mimics-tagging-roberta-base
# monologg/biobert_v1.1_pubmed