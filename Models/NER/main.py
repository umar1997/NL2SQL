import warnings

import argparse
from packaging import version

import torch
import numpy as np
import transformers

from domainClassification import Training
from log import get_logger


pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    parser = argparse.ArgumentParser()

    # ADD ARGUEMENTS
    parser.add_argument("--model_type", default='bert-base-cased', type=str, required=True,
                        help="valid values: bert-base-cased, dmis-lab/biobert-v1.1, _")
    parser.add_argument("--tokenizer_type", default='bert-base-cased', type=str, required=True,
                        help="valid values: bert-base-cased, _, _")
    parser.add_argument("--data_dir", type=str,
                        help="The input data directory.")
    parser.add_argument("--seed", default=3, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=80, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The batch size for training and evaluation.")
    parser.add_argument("--val_split", default=0.30, type=float,
                        help="Train test split for validation split.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default='AdamW', type=str,
                        help="valid values: AdamW, SGD")
    parser.add_argument("--scheduler", default='None', type=str,
                        help="valid values: Linear Warmup, LRonPlateau")
    parser.add_argument("--log_folder", default='./Log_Files/', type=str,
                        help="Name of log folder.")
    parser.add_argument("--log_file", default='sample.log', type=str,
                        help="Name of log file.")

    global_args = parser.parse_args()

    # MODEL HYPER PARAMETERS
    HYPER_PARAMETERS = {
        # "MAX_LEN" : 80, # Max Length of the sentence
        # "BATCH_SIZE" : 16,
        # "EPOCHS" : 3,
        # "MAX_GRAD_NORM" : 1.0,
        # "LEARNING_RATE" : 3e-5,
        # "EPSILON" : 1e-8

        "MAX_LEN" : global_args.max_seq_length, 
        "BATCH_SIZE" : global_args.batch_size,
        "EPOCHS" : global_args.num_epochs,
        "MAX_GRAD_NORM" : global_args.max_grad_norm,
        "LEARNING_RATE" : global_args.learning_rate,
        "EPSILON" : global_args.adam_epsilon,
        "TEST_SPLIT": global_args.val_split,
        "RANDOM_SEED": global_args.seed,
        "OPTIMIZER": global_args.optimizer,
        "LR_SCHEDULER": global_args.scheduler,
        "GENERAL_LABELS": False,
        "ADDED_LAYERS": False
    }

    # SEEDS
    seed = global_args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # LOG FILE SET UP
    file_name = global_args.log_folder + global_args.log_file
    logger_meta = get_logger(name='META', file_name=file_name, type='meta')
    logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
    logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')

    logger_meta.warning("TOKENIZER_TYPE: {}".format(global_args.tokenizer_type))
    logger_meta.warning("MODEL_TYPE: {}".format(global_args.model_type))
    for i, (k, v) in enumerate(HYPER_PARAMETERS.items()):
        if i == (len(HYPER_PARAMETERS) - 1):
            logger_meta.warning("{}: {}\n".format(k, v))
        else:
            logger_meta.warning("{}: {}".format(k, v))

    # RUN MODEL
    print('Entity Classification Training')
    print('------------------------------')
    train = Training(global_args.tokenizer_type, global_args.model_type, HYPER_PARAMETERS, logger_progress, logger_results)
    train.run()


#----------------------------------------------------------------------------------
script = """
python main.py \
    --model_type dmis-lab/biobert-v1.1 \
    --tokenizer_type dmis-lab/biobert-v1.1 \
    --data_dir ./../../Data/Chia_w_scope_data.csv \
    --max_seq_length 80 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --val_split 0.30 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler LinearWarmup \
    --log_folder ./Log_Files/ \
    --log_file biobert_.log
"""

####################################### Different Models
# dmis-lab/biobert-large-cased-v1.1

# dmis-lab/biobert-v1.1
# bert-base-cased
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# fidukm34/biobert_v1.1_pubmed-finetuned-ner-finetuned-ner
# sciarrilli/biobert-base-cased-v1.2-finetuned-ner (Token Classifcation)
# emilyalsentzer/Bio_ClinicalBERT

# bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16
# algoprog/mimics-tagging-roberta-base
# monologg/biobert_v1.1_pubmed