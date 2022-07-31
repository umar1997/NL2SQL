import torch

import time
from tqdm import tqdm, trange

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import get_linear_schedule_with_warmup

from get_dataset import CustomDataset
from T5_Model import T5_FineTuner


class Training:

    def __init__(self, HYPER_PARAMETERS, logger_progress, logger_results):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

        self.logger_progress = logger_progress
        self.logger_results = logger_results
        self.HYPER_PARAMETERS = HYPER_PARAMETERS

        T5_Model = T5_FineTuner(self.HYPER_PARAMETERS)
        self.model = T5_Model.model
        self.model.to(self.device)

        self.tokenizer = T5_Model.tokenizer

    def get_number_of_parameters(self,):

        pytorch_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(pytorch_trainable_params, pytorch_total_params)
    

    def data_loader(self,):

        training_dataset = CustomDataset(
            data_split='Train',
            max_input_length= self.HYPER_PARAMETERS["MAX_INPUT_LENGTH"],
            max_output_length= self.HYPER_PARAMETERS["MAX_OUTPUT_LENGTH"],
        )

        validation_dataset = CustomDataset(
            data_split='Validation',
            max_input_length= self.HYPER_PARAMETERS["MAX_INPUT_LENGTH"],
            max_output_length= self.HYPER_PARAMETERS["MAX_OUTPUT_LENGTH"],
        )

        train_params = {
            'batch_size': self.HYPER_PARAMETERS["TRAIN_BATCH_SIZE"],
            'shuffle': True,
            'num_workers': 0
        }

        val_params = {
            'batch_size': self.HYPER_PARAMETERS["EVAL_BATCH_SIZE"],
            'shuffle': False,
            'num_workers': 0
            }
        
        training_loader = DataLoader(training_dataset, **train_params)
        validation_loader = DataLoader(validation_dataset, **val_params)

        return training_loader, validation_loader

    def optimizer_and_lr_scheduler(self, training_loader):
        param_optimizer = list(self.model.named_parameters())
        # no_decay = ['bias', 'gamma', 'beta']
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # Setting Weight Decay Rate 0.01 if it isnt bias, gamma and beta
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay_rate': self.HYPER_PARAMETERS["WEIGHT_DECAY"]},
            # If it is set to 0.0
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.0}
        ]

        # Optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr= self.HYPER_PARAMETERS['LEARNING_RATE'],
            eps= self.HYPER_PARAMETERS['EPSILON'],
            # weight_decay=0.01 Doing stuff above for weight decay
        )

        # Scheeduler

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(training_loader) * self.HYPER_PARAMETERS['EPOCHS']

        if self.HYPER_PARAMETERS['LR_SCHEDULER'] == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=100, 
                eta_min=1e-5, 
                last_epoch=-1, 
                verbose=False
            )

        elif self.HYPER_PARAMETERS['LR_SCHEDULER'] == 'LinearWarmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

        return optimizer, scheduler

    def ids_to_clean_text(self, generated_ids):
        """
        Get text from the generated token ids.
        """
        def lmap(f, x):
            return list(map(f, x))
        
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def training_phase(self, train_dataloader, optimizer):
        train_loss=0

        self.model.train()
        CHECK = 1
        for step, batch in enumerate(tqdm(train_dataloader, desc ="Training DataLoader")):

            input_ids = batch['source_ids'].to(self.device, dtype = torch.long)
            attention_mask = batch['source_mask'].to(self.device, dtype = torch.long)
            decoder_attention_mask = batch['target_mask'].to(self.device, dtype = torch.long)
            y = batch['target_ids'].to(self.device, dtype = torch.long)

            decoder_input_ids = y[:, :].contiguous()

            labels = y[:, :].clone().detach()
            labels[y[:, :] == self.tokenizer.pad_token_id] = -100
            
            # print(input_ids.shape, attention_mask.shape, decoder_attention_mask.shape, y.shape, decoder_input_ids.shape)
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
            
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward() # Getting the loss and performing backward pass
            optimizer.step()

            train_loss += loss.item() # Tracking loss

            # Preventing exploding grad
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.HYPER_PARAMETERS['MAX_GRAD_NORM'])

            # if CHECK == 10:
            #     break
            # CHECK+= 1


        avg_train_loss = train_loss/len(train_dataloader)
        return avg_train_loss, optimizer

    def validation_phase(self, valid_dataloader):
        eval_loss = 0

        CHECK = 1

        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(valid_dataloader, desc ="Validation DataLoader")):

                input_ids = batch['source_ids'].to(self.device, dtype = torch.long)
                attention_mask = batch['source_mask'].to(self.device, dtype = torch.long)
                decoder_attention_mask = batch['target_mask'].to(self.device, dtype = torch.long)
                y = batch['target_ids'].to(self.device, dtype = torch.long)

                decoder_input_ids = y[:, :].contiguous()

                labels = y[:, :].clone().detach()
                labels[y[:, :] == self.tokenizer.pad_token_id] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
            
                loss = outputs[0]

                eval_loss += loss.item()

                # if CHECK == 5:
                #     break
                # CHECK+= 1

        avg_val_loss = eval_loss/len(valid_dataloader)
        return avg_val_loss

    def training_and_validation(self, train_dataloader, valid_dataloader, optimizer, scheduler):
        
        loss_values, validation_loss_values = [], []
        E = 1

        # data = next(iter(train_dataloader)) 
        # print(self.tokenizer.batch_decode(data['source_ids']))

        for _ in range(self.HYPER_PARAMETERS['EPOCHS']):
            self.logger_results.info('Epoch #{}'.format(E))
            print('Epoch #{}'.format(E))

            start = time.time()

            ###################### TRAINING
            avg_train_loss, optimizer = self.training_phase(train_dataloader, optimizer)
            print('Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))
            self.logger_results.info('Average Train Loss For Epoch {}: {}'.format(E, avg_train_loss))
            loss_values.append(avg_train_loss)  # Storing loss values to plot learning curve
            
            ###################### VALIDATION
            avg_val_loss = self.validation_phase(valid_dataloader)
            print('Average Validation Loss For Epoch {}: {}'.format(E, avg_val_loss))
            self.logger_results.info('Average Val Loss For Epoch {}: {}'.format(E, avg_val_loss))
            validation_loss_values.append(avg_val_loss)  # Storing loss values to plot learning curve

            ###################### SAVE MODEL
            self.logger_results.info('Saving Model . . .')
            torch.save({
                'epoch': E,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'valid_loss': avg_val_loss,
                }, './model_checkpoints/T5_Fine_Tuned_Epoch_{}.pth'.format(E))


            ######################
            stop = time.time()
            print('Epoch #{} Duration:{}'.format(E, stop-start))
            self.logger_results.info('Duration: {}\n'.format(stop-start))
            E+=1
            time.sleep(3)

        

    def run(self,):
        self.logger_progress.critical('Model Initialized!')
        training_loader, validation_loader = self.data_loader()
        self.logger_progress.critical('Data Loaders Created!')
        optimizers, scheduler = self.optimizer_and_lr_scheduler(training_loader)
        self.logger_progress.critical('Optimizer and Scheduler Created!')
        self.logger_progress.critical('Starting Training. . .\n')
        self.training_and_validation(training_loader, validation_loader, optimizers, scheduler)
        self.logger_progress.critical('Training Completed!')
