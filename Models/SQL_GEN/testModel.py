# LINKS:
# https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
# https://github.com/amazon-research/nl2sql-omop-cdm/blob/main/src/engine/step4/model_dev/t5_evaluation.py

import torch
from torch.optim import AdamW

from T5_Model import T5_FineTuner


class Inference:

    def __init__(self, HYPER_PARAMETERS, PATH, INPUT_TEXT):

        self.HYPER_PARAMETERS = HYPER_PARAMETERS
        self.model = None
        self.tokenizer = None
        self.PATH = PATH
        self.INPUT_TEXT = INPUT_TEXT
        # PATH = './model_checkpoints/T5_Fine_Tuned_Epoch_1.pth'

    def load_model(self,):
        self.model = T5_FineTuner(self.HYPER_PARAMETERS)
        PARAMS = self.optimizer_params()
        optimizer = AdamW(**PARAMS)

        
        checkpoint = torch.load(self.PATH)

        self.model.load_state_dict(checkpoint['model_state_dict'], False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']

        self.tokenizer = self.model.tokenizer

    def optimizer_params(self,):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # Setting Weight Decay Rate 0.01 if it isnt bias, gamma and beta
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay_rate': self.HYPER_PARAMETERS["WEIGHT_DECAY"]},
            # If it is set to 0.0
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.0}
        ]

        PARAMS = {
            'params' : optimizer_grouped_parameters,
            'lr' : self.HYPER_PARAMETERS['LEARNING_RATE'],
            'eps' : self.HYPER_PARAMETERS['EPSILON'],
        }

        return PARAMS

    def generate_query(self,):

        input_text = "translate English to SQL: %s" % self.INPUT_TEXT

        features = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length= self.HYPER_PARAMETERS['MAX_INPUT_LENGTH'],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            )      

        output = self.model.model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=self.HYPER_PARAMETERS['MAX_OUTPUT_LENGTH'],
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )

        output = self.tokenizer.decode(output[0])
        print(output)

        # generic sql post-processing
        # output = re.sub(PAD_P, "", output)
        # output = output.replace("[", "<").replace("]", ">").strip()

        return output

    def run(self,):
        self.load_model()
        self.generate_query()




# https://huggingface.co/transformers/v2.9.1/main_classes/model.html#transformers.PreTrainedModel.generate
# generated_ids = self.model.generate(
#     input_ids,
#     attention_mask=attention_mask
# )

# print(self.tokenizer.decode(generated_ids[0]))
# print('############################')
# preds = self.ids_to_clean_text(generated_ids)
# target = self.ids_to_clean_text(decoder_input_ids)

# print(preds)
# print('---------')
# print(target)

# if CHECK == 3:
#     break
# CHECK+= 1


            # print('input_ids',self.tokenizer.decode(input_ids[0]))
            # print('decoder_input_ids', self.tokenizer.decode(decoder_input_ids[0]))
            # print('y', self.tokenizer.decode(y[0]))
            # print('y', y[0])
            # print('lmlabels', lm_labels[0])