import os
import sys
path = os.path.realpath('./Models/PL_Model/')
dir_path = os.path.dirname(path)
# print(path)
# print(dir_path)
# /home/umar.salman/G42/NLP2SQL/Models/PL_Model
# /home/umar.salman/G42/NLP2SQL/Models
sys.path.append(dir_path)

import re
import argparse
from PL_Model.T5PL_Model import T5FineTuner
import torch


PAD_P = re.compile("<pad> |</s>")


class Inferencer(object):
    def __init__(self,):

        model_path = ('./Models/Model_Files/sql_gen_model_checkpoint.ckpt')
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
