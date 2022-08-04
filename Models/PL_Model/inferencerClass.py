import re
import argparse

from T5PL_Model import T5FineTuner
import torch


PAD_P = re.compile("<pad> |</s>")


class Inferencer(object):
    def __init__(self, path):

        model_path = (path)
        checkpoint = torch.load(model_path)
        args = argparse.Namespace(**checkpoint["hyper_parameters"])
        self.model = T5FineTuner(args)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.tokenizer = self.model.tokenizer

    def __call__(self, input_text):
        """Maps a general NLQ (with placeholders) to a general SQL query (with placeholders)

        Args:
            input_text (str): General Natural Language question text.

        Returns:
            str: Generic SQL Query.
        """
        input_text = "translate English to SQL: %s" % input_text

        features = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.model.hparams.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        output = self.model.model.generate(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            max_length=self.model.hparams.max_output_length,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )

        output = self.tokenizer.decode(output[0])

        # generic sql post-processing
        output = re.sub(PAD_P, "", output)
        output = output.replace("[", "<").replace("]", ">").strip()

        return output
