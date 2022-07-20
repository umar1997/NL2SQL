import os
import sys

dir_path = os.path.dirname(os.path.realpath('./../'))
sys.path.append(dir_path)

import argparse
from packaging import version

import random

import torch
import numpy as np
import transformers

from trainModel import Training
from log import get_logger

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda'), \
    'You are currently running on CPU rather than on CUDA.'

if __name__ == '__main__':

    # ADD ARGUEMENTS
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='mrm8488/t5-base-finetuned-wikiSQL', type=str, required=True,
                        help="valid values: mrm8488/t5-base-finetuned-wikiSQL, _")
    parser.add_argument("--tokenizer_name", default='mrm8488/t5-base-finetuned-wikiSQL', type=str, required=True,
                        help="valid values: mrm8488/t5-base-finetuned-wikiSQL, _")
    parser.add_argument("--data_dir", type=str,
                        help="The input data directory.")
    parser.add_argument("--max_input_length", default=256, type=int,
                        help="maximum number of tokens allowed in each sentence input")
    parser.add_argument("--max_output_length", default=576, type=int,
                        help="maximum number of tokens allowed in each sentence output")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--seed", default=3, type=int,
                        help='random seed')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight Decay for Adam optimizer.")
    parser.add_argument("--num_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="The batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="The batch size for evaluation.")
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
        "MODEL_NAME" : global_args.model_name,
        "TOKENIZER_NAME" : global_args.tokenizer_name, 
        "MAX_INPUT_LENGTH" : global_args.max_input_length,
        "MAX_OUTPUT_LENGTH" : global_args.max_output_length,
        "TRAIN_BATCH_SIZE" : global_args.train_batch_size,
        "EVAL_BATCH_SIZE" : global_args.eval_batch_size,
        "EPOCHS" : global_args.num_epochs,
        "LEARNING_RATE" : global_args.learning_rate,
        "EPSILON" : global_args.adam_epsilon,
        "RANDOM_SEED": global_args.seed,
        "MAX_GRAD_NORM" : global_args.max_grad_norm,
        "OPTIMIZER": global_args.optimizer,
        "LR_SCHEDULER": global_args.scheduler,
        "FREEZE_ENCODER": True,
        "FREEZE_EMBEDDINGS": True
    }

    # SEEDS
    def set_seed(seed):
        """Seed random seed if needed."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    set_seed(global_args.seed)


    # LOG FILE SET UP
    file_name = global_args.log_folder + global_args.log_file
    logger_meta = get_logger(name='META', file_name=file_name, type='meta')
    logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
    logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')

    logger_meta.warning("TOKENIZER_NAME: {}".format(global_args.tokenizer_name))
    logger_meta.warning("MODEL_NAME: {}".format(global_args.model_name))
    for i, (k, v) in enumerate(HYPER_PARAMETERS.items()):
        if i == (len(HYPER_PARAMETERS) - 1):
            logger_meta.warning("{}: {}\n".format(k, v))
        else:
            logger_meta.warning("{}: {}".format(k, v))

    # RUN MODEL

    from T5_Model import T5_FineTuner

    print('T5 Fine-Tuning SQL Generation')
    print('------------------------------')
    init = T5_FineTuner(HYPER_PARAMETERS)
    # train = Training(HYPER_PARAMETERS, logger_progress, logger_results)
    # train.run()


# ---------------------------------------------------------------------------------------------

script = """
python main.py \
    --model_name mrm8488/t5-base-finetuned-wikiSQL \
    --tokenizer_name mrm8488/t5-base-finetuned-wikiSQL \
    --data_dir ./../../Data/PreparedText2SQL \
    --max_input_length 256 \
    --max_output_length 576 \
    --learning_rate 1e-3 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --num_epochs 5 \
    --train_batch_size 128 \
    --eval_batch_size 256 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler LinearWarmup \
    --log_folder ./Log_Files/ \
    --log_file finetuned_t5.log
"""




# extra_model_params = {

#     "warmup_steps": 0,
#     "gradient_accumulation_steps": 1,
#     "n_gpu": -1,
#     "resume_from_checkpoint": True,
#     "val_check_interval": 0.8,
#     "n_train": -1,
#     "n_val": -1,
#     "n_test": -1,
#     "early_stop_callback": True,
#     "fp_16": False,
#     "opt_level": "O1",
#     "max_grad_norm": 1.0,
#     "automatic_optimization": True,
# }