# LINKS:
# https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
# https://github.com/amazon-research/nl2sql-omop-cdm/blob/main/src/engine/step4/model_dev/t5_evaluation.py


import re
import torch
import numpy as np

from transformers import AutoTokenizer

from Models.NER.Ner_Model import NER_FineTuner
from Models.SQL_GEN.T5_Model import T5_FineTuner



################################################################## HELPER FUNCTIONS AND VARIABLES

tag_values = ['B-Person', 'O', 'B-Value', 'I-Value', 'B-Condition', 'B-Qualifier', 'B-Procedure', 'I-Procedure', 'B-Measurement', 'I-Measurement', 'B-Temporal', 'I-Condition', 'I-Qualifier', 'I-Temporal', 'B-Observation', 'I-Observation', 'B-Drug', 'B-Device', 'I-Device', 'I-Drug', 'B-Negation', 'I-Negation', 'I-Person', 'PAD']

def cleanInput(txt):
    txt = txt.replace('/',' ')
    txt = txt.replace('-',' ')
    txt = txt.replace(';',' ')
    txt = re.sub(r' {2,}', ' ',txt)
    txt = txt.replace('.\n','')
    txt = txt.replace('\n', '')
    txt = txt.strip()
    return txt

def removePunctuation(word):
    word = re.sub(r'^(\.|,|\(|\))', '', word)
    word = re.sub(r'(\.|,|\(|\))$', '', word)
    return word

################################################################## NAMED ENTITY RECOGNITION

def NER(INPUT):
    # INPUT = 'Count of patients with paracetamol and brufen'
    
    txt = cleanInput(INPUT) 
    tokens = [removePunctuation(w) for w in txt.split()]
    input_ = ' '.join(tokens)

    ner_model = NER_FineTuner(
        checkpoint_tokenizer = 'dmis-lab/biobert-v1.1',
        checkpoint_model = 'dmis-lab/biobert-v1.1',
        num_tags = 24
    )

    ner_model.load_state_dict(torch.load('./Models/Model_Files/ner_model.pt'))
    tokenizer = AutoTokenizer.from_pretrained("./Models/Model_Files/tokenizer/")

    ner_model.to(device)
    ner_model.eval()

    tokenized_sentence = tokenizer.encode(input_) # Encode sentence with BERT tokenizer
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    with torch.no_grad(): # Forward Pass without Backprop
        output = ner_model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(),axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

    new_tokens, new_labels = [], []

    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith('##'):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    assert len(new_tokens) == len(new_labels)

    # print(new_tokens)
    # print(new_labels)
    return new_tokens, new_labels

##################################################################  PREPROCESSING

def PREPROCESS(new_tokens, new_labels, INPUT):

    # new_tokens = ['[CLS]', 'Count', 'of', 'patients', 'with', 'paracetamol', 'paracetamol2', 'and', 'brufen', 'brufen2', 'brufen3',  'and', 'severe', 'headache','[SEP]']
    # new_labels= ['O', 'O', 'O', 'O', 'O', 'B-Drug', 'I-Drug', 'O', 'B-Drug', 'I-Drug', 'I-Drug', 'O', 'B-Condition', 'I-Condition', 'O']
    # INPUT = ' '.join(new_tokens)

    new_tokens = new_tokens[1:-1]
    new_labels = new_labels[1:-1]

    assert len(new_tokens) == len(new_labels)

    LABELS, TOKENS = [], []
    for token, labels in zip(new_tokens, new_labels):
            if labels.startswith('I-'):
                TOKENS[-1] = TOKENS[-1] + ' ' + token
                LABELS[-1] = LABELS[-1]
            else:
                TOKENS.append(token)
                LABELS.append(labels)
    assert len(TOKENS) == len(LABELS)

    entity2count = {}
    arg2token = {}
    arg_value = '<ARG-#><*>'
    for i, label in enumerate(LABELS):
        if label.startswith('B-'):
            try:
                entity2count[label[2:]] += 1
            except:
                entity2count[label[2:]] = 1
            
            countOfEntity = entity2count[label[2:]] - 1
            arg = arg_value.replace('#',str(label[2:]).upper()).replace('*',str(countOfEntity))
            arg2token[TOKENS[i]] = arg


    for k, v in arg2token.items():
        INPUT = INPUT.replace(k, v)

    # print(og_tokens)
    return INPUT

################################################################## SQL GENERATION

def GENERATE_SQL(INPUT, ORIGINAL):    
    # INPUT = 'Count of patients with <ARG-DRUG><0> and <ARG-DRUG><1>'
    
    HYPER_PARAMS = {
        "MODEL_NAME" : 'mrm8488/t5-base-finetuned-wikiSQL',
        "TOKENIZER_NAME" : 'mrm8488/t5-base-finetuned-wikiSQL',
        "FREEZE_EMBEDDINGS" : True,
        "FREEZE_ENCODER" : True
    }
    
    sql_model = T5_FineTuner(
        HYPER_PARAMS
    )

    sql_model.load_state_dict(torch.load('./Models/Model_Files/sql_gen_model_.pt'))
    t5_tokenizer = AutoTokenizer.from_pretrained("./Models/Model_Files/T5_tokenizer/")

    input_text = "translate English to SQL: %s" % INPUT

    # print(sql_model)
    # print(type(sql_model))
    # print(t5_tokenizer)

    # # We use Encode Plus instead of Batch Encode Plus for a single sequence
    # # https://datascience.stackexchange.com/questions/103103/what-is-the-difference-between-batch-encode-plus-and-encode-plus
    # # Difference between Encode and Encode Plus
    # # https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging

    features = t5_tokenizer.encode_plus(input_text, return_tensors="pt")
    # print(features)

    output = sql_model.model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=512,
            num_beams=2,
            early_stopping= True,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )
    
    output = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    OUTPUT= output.replace("<pad>", "").replace("</s>", "").replace("[", "<").replace("]", ">").strip()
    # print(output)

    return OUTPUT

################################################################## PIPELINE START

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ORIGINAL = ''
    INPUT = ORIGINAL

    tokens_, labels_ = NER(INPUT)
    INPUT = PREPROCESS(tokens_, labels_, INPUT)
    OUTPUT = GENERATE_SQL(INPUT, ORIGINAL)

