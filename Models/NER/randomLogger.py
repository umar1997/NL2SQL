import os
import sys

dir_path = os.path.dirname(os.path.realpath('./../'))
sys.path.append(dir_path)

# print(dir_path)
# print(os.path.dirname('/home/umar.salman/G42/NLP2SQL'))
# print(os.path.realpath('./../'))
from log import get_logger

logger_meta = get_logger(name='META', file_name='sample.log', type='meta')
logger_progress = get_logger(name='PORGRESS', file_name='sample.log', type='progress')
logger_results = get_logger(name='RESULTS', file_name='sample.log', type='results')

logger_meta.warning('Hello Meta')
logger_progress.critical('Hello Prog')
logger_results.info('Hello Results')