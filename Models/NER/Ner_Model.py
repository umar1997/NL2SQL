import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForTokenClassification

class NER_FineTuner(nn.Module):

    def __init__(self, checkpoint_tokenizer, checkpoint_model, num_tags):
    
        super(NER_FineTuner, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = False)
        self.model = AutoModelForTokenClassification.from_pretrained(
            checkpoint_model,
            num_labels= num_tags,
            output_attentions = False,
            output_hidden_states = False
        )

    def forward(self, 
        input_ids, 
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):

        return self.model(
            input_ids, 
            token_type_ids=None,
            attention_mask=attention_mask, 
            labels=labels
        )