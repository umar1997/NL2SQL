import re
import json
import torch
import numpy as np

from transformers import AutoTokenizer
from Models.NER.Ner_Model import NER_FineTuner

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


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT = 'patients with paracetamol and brufen'
    # <ARG-DRUG><0> and <ARG-DRUG><1>



################################################################## NAMED ENTITY RECOGNITION
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
    
    for token, label in zip(new_tokens, new_labels): # Showing labels against the words
        print('{}\t{}'.format(label,token))

################################################################## SQL GENERATION PREPROCESSING
