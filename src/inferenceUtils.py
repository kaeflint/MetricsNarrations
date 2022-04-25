import copy
import numpy as np
from pytorch_lightning import seed_everything

# class for handling the performance narration


class PerformanceNarrator:
    def __init__(self, model,
                 experiments_dataset,
                 device,
                 max_iter=5,
                 sampling=True, verbose=False) -> None:
        self.model = model.to(device)
        self.experiments_dataset = experiments_dataset
        self.vectorizer = vectorizer = lambda x: experiments_dataset.transform(
            x)
        self.max_iter = max_iter
        self.sampling = sampling
        self.verbose = verbose
        self.device = device

    def generateNarration(self, example, seed,  max_length=190,
                          length_penalty=8.6, beam_size=10, return_top_beams=1):
        seed_everything(seed)

        device = self.device

        example = copy.deepcopy(example)
        sample_too = self.sampling
        bs = beam_size

        batch = self.vectorizer(example)

        met, rate, val = batch['metrics_seq'].to(
            device), batch['rates'].to(device), batch['values'].to(device)
        #clb, di = batch['class_labels'].to(device), batch['data_info'].to(device)
        met_att = batch['metrics_attention'].to(device)
        rate_att = batch['rate_attention'].to(device)
        val_att = batch['value_attention'].to(device)

        preamble_tokens = batch['preamble_tokens'].to(device)
        preamble_attention_mask = batch['preamble_attention_mask'].to(
            device)
        #labels = batch['labels'].to(device)

        if self.model.aux_encoder is not None:
            table_rep = self.model.performAuxEncoding(
                [met, met_att], [val, val_att], [rate, rate_att])

            sample_outputs = self.mode.generator.generate(input_ids=preamble_tokens,
                                                                    attention_mask=preamble_attention_mask,
                                                                    table_inputs=table_rep,
                                                                    table_attention_mask=None,
                                                                    num_beams=bs,
                                                                    repetition_penalty=1.5,
                                                                    length_penalty=length_penalty,
                                                                    early_stopping=True,
                                                                    use_cache=True,
                                                                    max_length=max_length,
                                                                    no_repeat_ngram_size=2,
                                                                    num_return_sequences=return_top_beams,
                                                                    do_sample=sample_too
                                                                    )
        else:
                sample_outputs = self.mode.generator.generate(input_ids=preamble_tokens,
                                                                    attention_mask=preamble_attention_mask,
                                                                    num_beams=bs,
                                                                    repetition_penalty=1.5,
                                                                    length_penalty=8.6,
                                                                    early_stopping=True,
                                                                    use_cache=True,
                                                                    max_length=190,
                                                                    no_repeat_ngram_size=2,
                                                                    num_return_sequences=return_top_beams,
                                                                    do_sample=sample_too
                                                                    )
        sentence_output = [self.experiments_dataset.tokenizer.decode(s, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True) for s in sample_outputs]
        
        return sentence_output

