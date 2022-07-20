import torch

from T5_Model import T5_FineTuner


class Training:

    def __init__(self, HYPER_PARAMETERS, logger_progress, logger_results):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device == torch.device('cuda')

        self.logger_progress = logger_progress
        self.logger_results = logger_results
        self.HYPER_PARAMETERS = HYPER_PARAMETERS

        self.model = None