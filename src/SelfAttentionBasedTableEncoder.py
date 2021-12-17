from dataclasses import dataclass
from transformers.models.t5.modeling_t5 import (
    T5Block, T5LayerNorm)
import math
import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy

import torch.nn.functional as F
from torch import Tensor, device, nn

import inspect
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.bart.modeling_bart import (BartAttention,BartEncoder,BartEncoderLayer,_expand_mask,ACT2FN)


class SelfAttentionEncoderBart(nn.Module):
    def __init__(self,config,):
        super().__init__()
        config = copy.deepcopy(config)
        self.att_block = BartEncoderLayer(config, )
        #self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        #self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self,input_embed,attention_mask=None):
        bs,seq_len,dim = input_embed.shape
        
        
        if attention_mask is not None:
            extn=_expand_mask(attention_mask,input_embed.dtype)
        else:
            attention_mask = torch.ones((bs,seq_len),device=input_embed.device,dtype=input_embed.dtype)
            extn= _expand_mask(attention_mask,input_embed.dtype)
        
        layer_outputs = self.att_block(input_embed,
                                       attention_mask=extn,
                                       layer_head_mask=None
                                       )
        
        hidden_states = layer_outputs[0]
        
        #hidden_states = self.final_layer_norm(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        
        return hidden_states
class CollapsedMetricsTableEncoderBart(nn.Module):
    """
    Takes as input the each metric token, with the value and the rating
    metric_name: [met_name,met_name_attention_mask]
                met_name (bs x nb_metrics*seq_len)
    
    """
    def __init__(self,config,embed_tokens):
        super().__init__()
        config = copy.deepcopy(config)
        self.hidden_dim = hidden_dim= config.d_model
        self.token_embedding_layer = self.embedding_layer = embed_tokens
        
        self.metric_name_sa = SelfAttentionEncoderBart(config,)
        self.metric_value_sa = SelfAttentionEncoderBart(config,)
        self.metric_rate_sa = SelfAttentionEncoderBart(config,)
        
        self.metrics_relation_module = SelfAttentionEncoderBart(config,)
        
        self.metric_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.value_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.output_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        
        
    def forward(self,metric_name,metric_value,metric_rate):
        met,mn_att = metric_name
        val,val_att = metric_value
        rate,rate_att = metric_rate
        
        bs,nb_metrics=met.shape
        
        # Collapse the nb_metrics and seqs together (i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mn = self.token_embedding_layer(met).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mr = self.token_embedding_layer(rate).view(bs,-1,self.hidden_dim) # (i.e. bs,nb_metrics * 2,self.hidden_dim)
        mv = self.token_embedding_layer(val).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        
        
        # Apply the different self attention units
        # We take the mean for each representation bs,nb_metrics ,self.hidden_dim
        mn_rep = self.metric_name_sa(mn,mn_att)
        mn_rep=mn_rep.view(bs,-1,self.hidden_dim)

        mv_rep = self.metric_value_sa(mv,val_att)
        mv_rep=mv_rep.view(bs,-1,self.hidden_dim)

        mr_rep = self.metric_rate_sa(mr,rate_att)
        mr_rep=mr_rep.view(bs,-1,self.hidden_dim)#.mean(2).unsqueeze(2).expand(-1,-1,seqs,-1)
        
        
        # Combine 
        metric_representation = torch.cat([mn_rep,mr_rep],dim=-1)
        metric_representation = self.activation_fn(self.metric_highlight_layer(metric_representation)) + mr_rep
        
        value_representation = torch.cat([mv_rep,mr_rep],dim=-1)
        value_representation = self.activation_fn(self.value_highlight_layer(value_representation)) + mr_rep
        
        value_metric = self.output_layer(torch.cat((value_representation, metric_representation),dim=-1)) + mr_rep + mv_rep + mn_rep
        output = self.dropout(self.final_layer_norm(value_metric))
        
        return output 
        #value_metric




def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
class SelfAttentionEncoder(nn.Module):
    def __init__(self,config,):
        super().__init__()
        config = copy.deepcopy(config)
        self.att_block = T5Block(config, has_relative_attention_bias=True)
        #self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        #self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self,input_embed,attention_mask=None):
        bs,seq_len,dim = input_embed.shape
        
        
        if attention_mask is not None:
            extn=get_extended_attention_mask(attention_mask,(bs, seq_len),input_embed.device)
        else:
            attention_mask = torch.ones((bs,seq_len),device=input_embed.device)
            extn= get_extended_attention_mask(attention_mask,(bs, seq_len),input_embed.device)
        
        layer_outputs = self.att_block(input_embed,
                                       attention_mask=extn,
                                       position_bias=None)
        
        hidden_states, present_key_value_state = layer_outputs[:2]
        
        #hidden_states = self.final_layer_norm(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    

class CollapsedMetricsTableEncoder(nn.Module):
    """
    Takes as input the each metric token, with the value and the rating
    metric_name: [met_name,met_name_attention_mask]
                met_name (bs x nb_metrics*seq_len)
    
    """
    def __init__(self,config,embed_tokens):
        super().__init__()
        config = copy.deepcopy(config)
        self.hidden_dim = hidden_dim= config.d_model
        self.token_embedding_layer = self.embedding_layer = embed_tokens
        
        self.metric_name_sa = SelfAttentionEncoder(config,)
        self.metric_value_sa = SelfAttentionEncoder(config,)
        self.metric_rate_sa = SelfAttentionEncoder(config,)
        
        self.metrics_relation_module = SelfAttentionEncoder(config,)
        
        self.metric_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.value_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.output_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        
    def forward(self,metric_name,metric_value,metric_rate):
        met,mn_att = metric_name
        val,val_att = metric_value
        rate,rate_att = metric_rate
        
        bs,nb_metrics=met.shape
        
        # Collapse the nb_metrics and seqs together (i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mn = self.token_embedding_layer(met).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mr = self.token_embedding_layer(rate).view(bs,-1,self.hidden_dim) # (i.e. bs,nb_metrics * 2,self.hidden_dim)
        mv = self.token_embedding_layer(val).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        
        
        # Apply the different self attention units
        # We take the mean for each representation bs,nb_metrics ,self.hidden_dim
        mn_rep = self.metric_name_sa(mn,mn_att)
        mn_rep=mn_rep.view(bs,-1,self.hidden_dim)

        mv_rep = self.metric_value_sa(mv,val_att)
        mv_rep=mv_rep.view(bs,-1,self.hidden_dim)

        mr_rep = self.metric_rate_sa(mr,rate_att)
        mr_rep=mr_rep.view(bs,-1,self.hidden_dim)#.mean(2).unsqueeze(2).expand(-1,-1,seqs,-1)
        
        
        # Combine 
        metric_representation = torch.cat([mn_rep,mr_rep],dim=-1)
        metric_representation = torch.nn.LeakyReLU()(self.metric_highlight_layer(metric_representation)) + mr_rep
        
        value_representation = torch.cat([mv_rep,mr_rep],dim=-1)
        value_representation = torch.nn.LeakyReLU()(self.value_highlight_layer(value_representation)) + mr_rep
        
        value_metric = self.output_layer(torch.cat((value_representation, metric_representation),dim=-1)) + mr_rep + mv_rep + mn_rep
        output = self.dropout(self.final_layer_norm(value_metric))
        
        return output


class CollapsedMetricsTableDecoder(torch.nn.Module):
    def __init__(self,table_encoder,input_dim,
                 nb_metrics=6,
                 max_metric_toks =8,
                 max_val_toks = 8,
                 max_rate_toks = 2,
                ):
        super(CollapsedMetricsTableDecoder, self).__init__()
        self.max_metric_toks = max_metric_toks
        self.max_val_toks = max_val_toks
        self.max_rate_toks =max_rate_toks
        self.nb_metrics=nb_metrics
        #
        nb_metrics,metric_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.metric_output= torch.nn.Linear(metric_embedding_dim,nb_metrics,bias=False,)

        nb_values,value_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.value_output= torch.nn.Linear(value_embedding_dim,nb_values,bias=False,)

        nb_rate,rate_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.rate_output= torch.nn.Linear(rate_embedding_dim,nb_rate,bias=False,)
        
        
        # Share the embeddings for the metric value and rate with the output predictors for the decoder
        self.metric_output.weight= table_encoder.embedding_layer.weight
        self.value_output.weight= table_encoder.embedding_layer.weight
        self.rate_output.weight= table_encoder.embedding_layer.weight
        

        # project the output of the table encoder into the 3 inputs to the model
        self.metric_highlight_layer = torch.nn.Linear(input_dim,metric_embedding_dim,)
        self.value_highlight_layer = torch.nn.Linear(input_dim,value_embedding_dim,)
        self.rate_highlight_layer = torch.nn.Linear(input_dim,rate_embedding_dim,)
        
    def forward(self,x,masks):
        #bs,nb_metrics,1,hidden_dim = x.shape
        #x = x.expand(-1,-1,self.max_metric_toks,-1)
        #.view(bs,-1,self.max_metric_toks,hidden_dim )
        #print(x.shape)
        met_att,val_att,rate_att = masks
        model_dim = x.shape[-1]
        bs,nb_metrics = x.shape[:2]
        metrics = self.metric_highlight_layer(x) # .unsqueeze(2).expand(-1,-1,self.max_metric_toks,-1)
        values  = self.value_highlight_layer(x)
        rates   = self.rate_highlight_layer(x)#.mean(2).unsqueeze(2).expand(-1,-1,self.max_rate_toks,-1))
        
        metrics_output =  self.metric_output(metrics* (model_dim ** -0.5))*met_att.view(bs,-1,1) 
        value_output =  self.metric_output(values* (model_dim ** -0.5))*val_att.view(bs,-1,1) 
        rates_output =  self.metric_output(rates* (model_dim ** -0.5))*rate_att.view(bs,-1,1) 
        
        return metrics_output,value_output,rates_output
    
    

    
    
class PiecewiseMetricsTableEncoder(nn.Module):
    """
    Takes as input the each metric token, with the value and the rating
    metric_name: [met_name,met_name_attention_mask]
                met_name (bs x nb_metrics x seq_len)
    
    """
    def __init__(self,config,embed_tokens):
        super().__init__()
        config = copy.deepcopy(config)
        self.hidden_dim = hidden_dim= config.d_model
        self.token_embedding_layer = self.embedding_layer = embed_tokens
        
        self.metric_name_sa = SelfAttentionEncoder(config,)
        self.metric_value_sa = SelfAttentionEncoder(config,)
        self.metric_rate_sa = SelfAttentionEncoder(config,)
        
        self.metrics_relation_module = SelfAttentionEncoder(config,)
        
        self.metric_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.value_highlight_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.output_layer = torch.nn.Linear(2*hidden_dim,hidden_dim,)
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        
    def forward(self,metric_name,metric_value,metric_rate):
        met,mn_att = metric_name
        val,val_att = metric_value
        rate,rate_att = metric_rate
        
        bs,nb_metrics,seqs=met.shape
        
        # Collapse the nb_metrics and seqs together (i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mn = self.token_embedding_layer(met).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        mr = self.token_embedding_layer(rate).view(bs,-1,self.hidden_dim) # (i.e. bs,nb_metrics * 2,self.hidden_dim)
        mv = self.token_embedding_layer(val).view(bs,-1,self.hidden_dim) #(i.e. bs,nb_metrics * seqs,self.hidden_dim)
        
        
        # Apply the different self attention units
        # We take the mean for each representation bs,nb_metrics ,self.hidden_dim
        mn_rep = self.metric_name_sa(mn,mn_att)
        mn_rep=mn_rep.view(bs,nb_metrics,-1,self.hidden_dim)

        mv_rep = self.metric_value_sa(mv,val_att)
        mv_rep=mv_rep.view(bs,nb_metrics,-1,self.hidden_dim)

        mr_rep = self.metric_rate_sa(mr,rate_att)
        mr_rep=mr_rep.view(bs,nb_metrics,-1,self.hidden_dim)#.mean(2).unsqueeze(2).expand(-1,-1,seqs,-1)
        
        
        # Combine 
        metric_representation = torch.cat([mn_rep,mr_rep],dim=-1)
        metric_representation = torch.nn.LeakyReLU()(self.metric_highlight_layer(metric_representation)) + mr_rep
        value_representation = torch.cat([mv_rep,mr_rep],dim=-1)
        value_representation = torch.nn.LeakyReLU()(self.value_highlight_layer(value_representation)) + mr_rep
        
        value_metric = self.output_layer(torch.cat((value_representation,
                                                    metric_representation),dim=-1)) + mr_rep + mv_rep + mn_rep
        output = self.dropout(self.final_layer_norm(value_metric))
        
        return output


class PiecewiseMetricsTableDecoder(torch.nn.Module):
    
    def __init__(self,table_encoder,input_dim,
                 nb_metrics=6,
                 max_metric_toks =8,
                 max_val_toks = 8,
                 max_rate_toks = 2,
                ):
        super(PiecewiseMetricsTableDecoder, self).__init__()
        self.max_metric_toks = max_metric_toks
        self.max_val_toks = max_val_toks
        self.max_rate_toks =max_rate_toks
        self.nb_metrics=nb_metrics
        #
        nb_metrics,metric_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.metric_output= torch.nn.Linear(metric_embedding_dim,nb_metrics,bias=False,)

        nb_values,value_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.value_output= torch.nn.Linear(value_embedding_dim,nb_values,bias=False,)

        nb_rate,rate_embedding_dim = table_encoder.embedding_layer.weight.shape
        self.rate_output= torch.nn.Linear(rate_embedding_dim,nb_rate,bias=False,)
        
        
        # Share the embeddings for the metric value and rate with the output predictors for the decoder
        self.metric_output.weight= table_encoder.embedding_layer.weight
        self.value_output.weight= table_encoder.embedding_layer.weight
        self.rate_output.weight= table_encoder.embedding_layer.weight
        

        # project the output of the table encoder into the 3 inputs to the model
        self.metric_highlight_layer = torch.nn.Linear(input_dim,metric_embedding_dim,)
        self.value_highlight_layer = torch.nn.Linear(input_dim,value_embedding_dim,)
        self.rate_highlight_layer = torch.nn.Linear(input_dim,rate_embedding_dim,)
        
    def forward(self,x,masks):
        #bs,nb_metrics,1,hidden_dim = x.shape
        #x = x.expand(-1,-1,self.max_metric_toks,-1)
        #.view(bs,-1,self.max_metric_toks,hidden_dim )
        met_att,val_att,rate_att = masks
        model_dim = x.shape[-1]
        bs,nb_metrics = x.shape[:2]
        metrics = self.metric_highlight_layer(x) # .unsqueeze(2).expand(-1,-1,self.max_metric_toks,-1)
        values  = self.value_highlight_layer(x)
        rates   = self.rate_highlight_layer(x)#.mean(2).unsqueeze(2).expand(-1,-1,self.max_rate_toks,-1))
        
        metrics_output =  self.metric_output(metrics* (model_dim ** -0.5))*met_att.view(bs,nb_metrics,-1,1) 
        value_output =  self.metric_output(values* (model_dim ** -0.5))*val_att.view(bs,nb_metrics,-1,1) 
        rates_output =  self.metric_output(rates* (model_dim ** -0.5))*rate_att.view(bs,nb_metrics,-1,1) 
        
        return metrics_output,value_output,rates_output