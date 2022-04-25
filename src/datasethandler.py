import json

from data_utils import RDFDataSetForTableStructured
from src.model_utils import setupTokenizer


class NarrationDataSet:
    def __init__(self, modelbase, max_preamble_len=160,
                 max_len_trg=185,
                 max_rate_toks=8,
                 lower_narrations=True,
                 process_target=True,) -> None:

        # Get the tokenizer
        self.tokenizer_ = tokenizer_ = setupTokenizer(modelbase=modelbase)
        self.modelbase = modelbase
        self.max_preamble_len = max_preamble_len
        self.max_len_trg = max_len_trg
        self.max_rate_toks = max_rate_toks
        self.lower_narrations = lower_narrations

    def fit(self, trainset, testset):
        self.train_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                    trainset,
                                                    self.modelbase, max_preamble_len=self.max_preamble_len,
                                                    max_len_trg=self.max_len_trg,
                                                    max_rate_toks=self.max_rate_toks,
                                                    lower_narrations=self.lower_narrations,
                                                    process_target=self.process_target,
                                                    use_raw=False)
        self.test_dataset = RDFDataSetForTableStructured(self.tokenizer_,
                                                         testset,
                                                         self.modelbase, max_preamble_len=self.max_preamble_len,
                                                         max_len_trg=self.max_len_trg,
                                                         max_rate_toks=self.max_rate_toks,
                                                         lower_narrations=self.lower_narrations,
                                                         process_target=self.process_target,
                                                         use_raw=False)

    def transform(self, pack):
        return self.test_dataset.processTableInfo(pack)
    
    
