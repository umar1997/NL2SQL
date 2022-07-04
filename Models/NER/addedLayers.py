import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput


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