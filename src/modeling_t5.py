import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import mod
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, ModelOutput)
from transformers.models.t5.modeling_t5 import (T5Attention, T5Block,
                                                T5ForConditionalGeneration,
                                                T5LayerFF, T5LayerNorm,
                                                T5Stack)
from transformers.utils import logging

from model_utils import RelativeGlobalAttention


class GatedControl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.w1 = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w2 = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, y):
        gate = self.dropout(torch.sigmoid(self.w1(x)+self.w2(y)))
        o = gate * x + (1 - gate) * y
        return o


class RepresentationHighlight(nn.Module):
    def __init__(self, config, gate_control=False):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.FF = T5LayerFF(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gate_control = gate_control
        if gate_control:
            self.gateControl = GatedControl(self.config)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0]) if not self.gate_control else self.gateControl(
            hidden_states, self.dropout(attention_output[0]))
        #layer_output = self.FF(layer_output) + hidden_states
        outputs = (layer_output,) + attention_output[1:]

        return outputs


class EarlyFusionEncoder(T5Stack):
    # Take as input the semantic representation of the table
    # This then combined with generation prompt: eg. Question such as: A model trained to predict ...... as shown in the table:
    def __init__(self, config, embed_tokens=None, table_encoder=None):
        super(T5Stack, self).__init__(config)
        self.config = config
        print('Early Fusion')

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.table_embedding = table_encoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        self.relative_attention_for_table = RelativeGlobalAttention(d_model=config.d_model,
                                                                    num_heads=config.num_heads,
                                                                    max_len=1024, dropout=config.dropout_rate)

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self, input_ids=None,
            attention_mask=None,
            table_inputs=None,
            table_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, both_encoder_decoder=False,):

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        bs, pb_seq_len = inputs_embeds.size()[:-1]

        assert type(
            table_inputs) is not list, "Multiple table representations are not accepted at the moment"

        if len(table_inputs.size()) > 3:
            # Collapse the last but one dimention
            table_inputs = table_inputs.mean(-2)  # bs x cols x dim

        # Perform the Relative-Position Attention across the table input
        table_embedding = self.relative_attention_for_table(table_inputs)

        nb_columns = table_inputs.size(1)

        # Get the joint embeddings from the table preamble and the semantic representation of the table
        inputs_embeds = torch.cat([table_embedding, inputs_embeds], dim=1)

        # Compose the attention masks for the table and preamble
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if table_attention_mask is None:
            table_attention_mask = attention_mask.new_ones(bs, nb_columns)

        attention_mask = torch.cat(
            [table_attention_mask, attention_mask], dim=1)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (bs, pb_seq_len+nb_columns),
            inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = pb_seq_len+nb_columns
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                pb_seq_len, pb_seq_len)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :pb_seq_len, :pb_seq_len] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, pb_seq_len:, :] = 0
            # position_bias[:, :, :, pb_seq_len:] = 0
            position_bias = position_bias + extended_attention_mask

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    # head_mask=head_mask[i],
                    layer_head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                #position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class LateFusion(T5Stack):
    # Take as input the semantic representation of the table
    # This then combined with generation prompt: eg. Question such as: A model trained to predict ...... as shown in the table:
    def __init__(self, config, embed_tokens=None, table_encoder=None):
        super(T5Stack, self).__init__(config)
        self.config = config
        self.bottom_k = config.bottom_k

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.table_embedding = table_encoder
        print('Late Fusion Model FIP')

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        self.relative_attention_for_table = RelativeGlobalAttention(d_model=config.d_model,
                                                                    num_heads=config.num_heads,
                                                                    max_len=1024, dropout=config.dropout_rate)
        self.highlight_layer = RepresentationHighlight(
            self.config, gate_control=False)

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self, input_ids=None, attention_mask=None, table_inputs=None, table_attention_mask=None,

            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, both_encoder_decoder=False,):

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        bs, pb_seq_len = inputs_embeds.size()[:-1]

        assert type(
            table_inputs) is not list, "Multiple table representations are not accepted at the moment"

        if len(table_inputs.size()) > 3:
            # Collapse the last but one dimention
            table_inputs = table_inputs.mean(-2)  # bs x cols x dim

        # Perform the Relative-Position Attention across the table input
        table_embedding = self.relative_attention_for_table(table_inputs)

        nb_columns = table_inputs.size(1)

        # Get the joint embeddings from the table preamble and the semantic representation of the table
        #inputs_embeds = torch.cat([table_embedding,inputs_embeds], dim=1)

        # Compose the attention masks for the table and preamble
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if table_attention_mask is None:
            table_attention_mask = attention_mask.new_ones(bs, nb_columns)

        #attention_mask = torch.cat([table_attention_mask,attention_mask], dim=1)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (bs, pb_seq_len),
            inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = pb_seq_len  # +nb_columns
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                pb_seq_len, pb_seq_len)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :pb_seq_len, :pb_seq_len] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, pb_seq_len:, :] = 0
            # position_bias[:, :, :, pb_seq_len:] = 0
            position_bias = position_bias + extended_attention_mask
            #highlight_output = self.highlight_layer(hidden_states,table_embedding, )
            #hidden_states = highlight_output[0]

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    # head_mask=head_mask[i],
                    layer_head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                """ if i > self.bottom_k-1:
                    highlight_output = self.highlight_layer[i-self.bottom_k](
                        hidden_states, table_embedding, )
                    hidden_states = highlight_output[0]
 """
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                #position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)
        highlight_output = self.highlight_layer(
            hidden_states, table_embedding, )
        hidden_states = highlight_output[0]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class HybridFusionEncoder(T5Stack):
    # Take as input the semantic representation of the table
    # This then combined with generation prompt: eg. Question such as: A model trained to predict ...... as shown in the table:
    def __init__(self, config, embed_tokens=None, table_encoder=None):
        super(T5Stack, self).__init__(config)
        self.config = config
        self.bottom_k = config.bottom_k

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.table_embedding = table_encoder
        print('Hybrid Fusion')

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        self.relative_attention_for_table = RelativeGlobalAttention(d_model=config.d_model,
                                                                    num_heads=config.num_heads,
                                                                    max_len=1024, dropout=config.dropout_rate)
        # nn.ModuleList([RepresentationHighlight(self.config) for i in range(config.num_layers-self.bottom_k)])
        self.highlight_layer = RepresentationHighlight(
            self.config, gate_control=False)

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self, input_ids=None, attention_mask=None, table_inputs=None, table_attention_mask=None,

            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, both_encoder_decoder=False,):

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        bs, pb_seq_len = inputs_embeds.size()[:-1]

        assert type(
            table_inputs) is not list, "Multiple table representations are not accepted at the moment"

        if len(table_inputs.size()) > 3:
            # Collapse the last but one dimention
            table_inputs = table_inputs.mean(-2)  # bs x cols x dim

        # Perform the Relative-Position Attention across the table input
        table_embedding = self.relative_attention_for_table(table_inputs)

        nb_columns = table_inputs.size(1)

        # Get the joint embeddings from the table preamble and the semantic representation of the table
        inputs_embeds = torch.cat([table_embedding, inputs_embeds], dim=1)

        # Compose the attention masks for the table and preamble
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if table_attention_mask is None:
            table_attention_mask = attention_mask.new_ones(bs, nb_columns)

        attention_mask = torch.cat(
            [table_attention_mask, attention_mask], dim=1)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            (bs, pb_seq_len+nb_columns),
            inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:

            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = pb_seq_len+nb_columns
            q_len = seq_length
            k_len = seq_length

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(
                pb_seq_len, pb_seq_len)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)
            position_bias[:, :, :pb_seq_len, :pb_seq_len] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, pb_seq_len:, :] = 0
            # position_bias[:, :, :, pb_seq_len:] = 0
            position_bias = position_bias + extended_attention_mask
            #highlight_output = self.highlight_layer(hidden_states,table_embedding, )
            #hidden_states = highlight_output[0]

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    # head_mask=head_mask[i],
                    layer_head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                """ if i > self.bottom_k-1:
                    highlight_output = self.highlight_layer[i-self.bottom_k](
                        hidden_states, table_embedding, )
                    hidden_states = highlight_output[0]
 """
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                #position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)
        highlight_output = self.highlight_layer(
            hidden_states, table_embedding, )
        hidden_states = highlight_output[0]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class DataNarration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config,):
        super(T5ForConditionalGeneration, self).__init__(config)
        self.config = config
        model_type = config.modeltype
        print(str(model_type))

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        if str(model_type) in ['eaf', 'earlyfusion']:
            self.encoder = EarlyFusionEncoder(encoder_config, self.shared)
        elif str(model_type) in ['latefusion', 'lf']:
            self.encoder = LateFusion(encoder_config, self.shared)
        elif str(model_type) in ['hybrid', 'h']:
            self.encoder = HybridFusionEncoder(encoder_config, self.shared)
        else:
            print('Unknown encoder specified')

        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        self.lm_head.weight = self.shared.weight
        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        table_inputs=None,
        table_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        both_encoder_decoder=False,

        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                table_inputs=table_inputs,
                table_attention_mask=table_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(
                dtype=hidden_states.dtype, device=hidden_states.device)
        if table_attention_mask is None:
            bs, pb_seq_len = attention_mask.size()
            nb_cols = encoder_outputs[0].size(1) - pb_seq_len
            table_attention_mask = attention_mask.new_ones(bs, nb_cols)
            # both_encoder_decoder
        encoder_attention_mask = torch.cat(
            [table_attention_mask, attention_mask], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(
                    ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

        return TableNarrationLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
        )

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None,
            encoder_outputs=None,
            **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'table_attention_mask' in kwargs:
            output['table_attention_mask'] = kwargs['table_attention_mask']
        if 'table_inputs' in kwargs:
            output['table_inputs'] = kwargs['table_inputs']
        if 'both_encoder_decoder' in kwargs:
            output['both_encoder_decoder'] = kwargs['both_encoder_decoder']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("table_attention_mask", None) is not None:
            model_kwargs['table_attention_mask'] = model_kwargs['table_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        # if  model_kwargs.get("table_attention_mask", None) is not None:
        #    output['table_inputs'] = kwargs['table_inputs']
        return input_ids, model_kwargs


@dataclass
class TableNarrationLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    table_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    table_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    table_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
