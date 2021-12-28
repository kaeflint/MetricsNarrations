import torch
import copy
from tqdm import tqdm
from transformers import (AdamW, BartForConditionalGeneration,
                          T5ForConditionalGeneration,T5Config,BartConfig,
                          get_linear_schedule_with_warmup)

from modeling_bart import DataNarrationBart
from modeling_t5 import DataNarration
from SelfAttentionBasedTableEncoder import (CollapsedMetricsTableEncoder,
                                            CollapsedMetricsTableEncoderBart)

device = torch.device("cuda")


class NarrationModels(object):
    def __init__(self, vocab_size, model_type, config) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.t5config = config
        self.t5config.bottom_k = 11
        self.aux_encoder = None

        if self.model_type not in ['baseline', 'base', 'eaf', 'earlyfusion', 'latefusion', 'lf', 'hybrid', 'h']:
            print('Invalid model specified')

    def buildBaseline(self,):
        print('Not implemented')

    def buildFusionModels(self,):
        print('Not implemented')

    def compile(self, lr, warmup_steps, total_steps, epsilon=1e-8):
        parameters = [p for p in self.generator.parameters()]
        if self.aux_encoder is not None:
            parameters = list(set([p for p in self.generator.parameters(
            )]+[p for p in self.aux_encoder.parameters()]))
        self.optimizer = AdamW([{'params': parameters,
                                 'lr': lr}, ],
                               lr=lr, eps=epsilon
                               )
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
        print('Compilation Completed')

    def loadModel(self, model_path):
        print(f'loading model from: {model_path}')
        state_dicts = torch.load(model_path)
        self.generator.load_state_dict(state_dicts['generator_state_dict'])
        self.generator.eval()
        if self.aux_encoder is not None:
            self.aux_encoder.load_state_dict(
                state_dicts['aux_encoder_state_dict'])
            self.aux_encoder.eval()

    def saveModel(self, model_path,):

        models = {
            'generator_state_dict': self.generator.state_dict()}

        if self.aux_encoder is not None:
            models['aux_encoder_state_dict'] = self.aux_encoder.state_dict()
        torch.save(models, model_path)
        print(f'=== Model saved @ {model_path}')

    # This method will change for modelling on a different dataset


class BartNarrationModel(NarrationModels):
    def buildBaseline(self,):

        self.generator = BartForConditionalGeneration.from_pretrained(
            self.modelbase, config=self.bartconfig)
        self.generator.resize_token_embeddings(self.vocab_size)
        self.generator.cuda()

    def buildFusionModels(self,):
        # print(self.t5config.modeltype)
        self.generator = DataNarrationBart.from_pretrained(
            self.modelbase, config=self.bartconfig)
        # Build the Entries Encoder
        self.aux_encoder = CollapsedMetricsTableEncoderBart(
            self.bartconfig, self.generator.get_encoder().embed_tokens)

        # Resize the embedding layer
        self.generator.resize_token_embeddings(self.vocab_size)
        self.generator.cuda()
        self.aux_encoder.cuda()

    def performAuxEncoding(self, metric_data, value_data, rate_data):
        table_rep = self.aux_encoder(
            metric_data, value_data, rate_data).to(device)
        return table_rep

    def __init__(self, vocab_size, model_type, modelbase='facebook/bart-base'):
        self.modelbase = modelbase
        self.bartconfig = copy.deepcopy(BartConfig.from_pretrained(
            self.modelbase,
            output_hidden_states=False))
        self.bartconfig.relative_attention_num_buckets = 32
        self.bartconfig.d_kv = 64
        self.bartconfig.num_heads = self.bartconfig.encoder_attention_heads
        self.bartconfig.dropout_rate = self.bartconfig.dropout
        self.bartconfig.layer_norm_epsilon = 1e-06
        self.bartconfig.feed_forward_proj = self.bartconfig.activation_function
        super(BartNarrationModel, self).__init__(
            vocab_size, model_type, self.bartconfig)
        if model_type in ['baseline', 'base', ]:
            self.bartconfig.modeltype = model_type
            self.buildBaseline()
        else:
            self.bartconfig.modeltype = model_type
            self.buildFusionModels()


class T5NarrationModel(NarrationModels):
    def buildBaseline(self,):

        self.generator = T5ForConditionalGeneration.from_pretrained(
            self.modelbase, config=self.t5config)
        self.generator.resize_token_embeddings(self.vocab_size)
        self.generator.cuda()

    def buildFusionModels(self,):
        # print(self.t5config.modeltype)
        self.generator = DataNarration.from_pretrained(
            self.modelbase, config=self.t5config)
        # Build the Entries Encoder
        self.aux_encoder = CollapsedMetricsTableEncoder(
            self.t5config, self.generator.encoder.embed_tokens)

        # Resize the embedding layer
        self.generator.extend_vocab(self.vocab_size)
        self.generator.cuda()
        self.aux_encoder.cuda()

    def performAuxEncoding(self, metric_data, value_data, rate_data):
        table_rep = self.aux_encoder(
            metric_data, value_data, rate_data).to(device)
        return table_rep

    def __init__(self, vocab_size, model_type, modelbase='t5-base'):
        self.modelbase = modelbase
        self.t5config = copy.deepcopy(T5Config.from_pretrained(
            self.modelbase,
            output_hidden_states=False))
        super(T5NarrationModel, self).__init__(
            vocab_size, model_type, self.t5config)
        if model_type in ['baseline', 'base', ]:
            self.t5config.modeltype = model_type
            self.buildBaseline()
        else:
            self.t5config.modeltype = model_type
            self.buildFusionModels()
