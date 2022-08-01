# import pytorch_lightning as pl
import torch.nn as nn


from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

class T5_FineTuner(nn.Module):

    def __init__(self, HYPER_PARAMS):

        super(T5_FineTuner, self).__init__()

        self.HYPER_PARAMS = HYPER_PARAMS

        self.model = T5ForConditionalGeneration.from_pretrained(self.HYPER_PARAMS["MODEL_NAME"])
        self.tokenizer = T5Tokenizer.from_pretrained(self.HYPER_PARAMS["TOKENIZER_NAME"])

        if self.HYPER_PARAMS["FREEZE_EMBEDDINGS"]:
            self.freeze_embeddings()
        if self.HYPER_PARAMS["FREEZE_ENCODER"]:
            self.freeze_params(self.model.get_encoder())


        new_special_tokens = self.added_tokens()
        additional_special_tokens = (
            self.tokenizer.additional_special_tokens + new_special_tokens
        )
        # Adding tokens here
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )
        # Check if new tokens have been added successfully (If num_added_toks = 0 (bool) that means no extra token has been added)
        num_added_toks = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": new_special_tokens}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))


    def freeze_embeddings(self,):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)


    def freeze_params(self, model):
        """
        Freezes model paramters.
        """
        for params in model.parameters():
            params.requires_grad = False

    
    def added_tokens(self,):
        """Tokens to be added to the pretrained tokenizer/vocab."""
        added_tokens = [
            "[ARG-DRUG]",
            "[ARG-CONDITION]",
            "[ARG-GENDER]",
            "[ARG-RACE]",
            "[ARG-ETHNICITY]",
            "[ARG-STATE]",
            "[ARG-AGE]",
            "[ARG-TIMEDAYS]",
            "[ARG-TIMEYEARS]",
            "[GENDER-TEMPLATE]",
            "[RACE-TEMPLATE]",
            "[ETHNICITY-TEMPLATE]",
            "[STATEID-TEMPLATE]",
            "[CONDITION-TEMPLATE]",
            "[DRUG-TEMPLATE]",
            "[ARG-CONDITION]",
            "[STATENAME-TEMPLATE]",
            "[ARG-DRUG]",
            "[ARG-DAYS]",
            "DATEDIFF",
            "DISTINCT",
            "GREATEST",
            "[SCHEMA]",
            "SELECT",
            "GROUP",
            "LEAST",
            "UNION",
            "COUNT",
            "WHERE",
            "JOIN",
            "FROM",
            "AND",
            "AS",
            "OR",
            "BY",
            "ON",
        ] + [f"[{i}]" for i in range(10)]

        return added_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_input_ids = decoder_input_ids,
            labels=labels,
        )


