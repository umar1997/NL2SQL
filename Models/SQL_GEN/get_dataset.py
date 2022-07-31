import pandas as pd

from transformers import T5Tokenizer

from nlp import load_dataset
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom Dataset Object for DataLoader

    data_split values: Train, Validation, Test

    """

    def __init__(self, data_split, max_input_length, max_output_length):

        # data_split values: Train, Validation, Test that are recieved
        self.paths = Paths = {
            'Data': './../../Data/',
            'Train': './../../Data/PreparedText2SQL/train.csv',
            'Validation': './../../Data/PreparedText2SQL/validation.csv',
            'Test': './../../Data/PreparedText2SQL/test.csv',
        }

        # https://huggingface.co/docs/datasets/v0.3.0/loading_datasets.html#from-local-files
        # data_split must be in lower case when using load_dataset: train, validation, test
        self.data_split = data_split.lower()
        self.dataset = load_dataset("csv", data_files={data_split.lower(): self.paths[data_split]})
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_samples = len(pd.read_csv(self.paths[data_split]))

        # if self.num_samples:
        #     self.dataset[data_split] = self.dataset[data_split].select(
        #         list(range(0, self.num_samples))
        #     )

        # SentencePiece provides Python wrapper that supports both SentencePiece training and segmentation. 
        # You can install Python binary package of SentencePiece with.
        # T5Tokenizer requires the SentencePiece library but it was not found in your environment
        # pip install sentencepiece
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=self.max_input_length)
        # model_max_length: The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is loaded with from_pretrained()

    def __len__(self,):
        """
        Gets the number of samples in the dataset.
        """
        return self.dataset[self.data_split].shape[0]

    def convert_to_features(self, example_batch):
        """
        Encodes/tokenizes a single example (source and targets).
        Args:
            example_batch(pd.Series): A single example of the dataset[at least two columns: Folded_Question(source) & Query_Generated(target)]
        Returns:
            Tuple of source and targets (Tokenized/encoded versions).
        """

        input_ = "translate English to SQL: " + example_batch["Folded_Question"]
        input_ = input_.replace("<", "[").replace(">", "]")
        target_ = example_batch["Query_Generated"]
        target_ = target_.replace("<", "[").replace(">", "]")

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets

    def __getitem__(self, index):
        
        """
        This __getitem__ is called by the DataLoader which acts as the Dataset argument

        Gets a single tokenized example from the dataset.
        Args:
            index: Example index in the dataset.
        Returns:
            Dictionary of source token ids, source masks, target ids, and target masks.
        """

        source, targets = self.convert_to_features(self.dataset[self.data_split][index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

def get_dataset_object(data_split, max_input_length, max_output_length):
    """
    data_split values: Train, Validation, Test
    """
    return CustomDataset(
        data_split=data_split,
        max_input_length= max_input_length,
        max_output_length= max_output_length,
    )


if __name__ == '__main__':

    dataset_obj = CustomDataset('Train', 256, 512)
    data_split, max_input_length, max_output_length = 'Test', 256, 512
    x = get_dataset_object(data_split, max_input_length, max_output_length)
    print(x)
