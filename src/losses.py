r"""Losses based on the divergence between probability distributions."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def _kl_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # D_KL(P || Q)
    batch, seq, vocab_size = p.shape
    unsummed_kl = F.kl_div(
        q.reshape(batch *  seq, -1).log(), p.reshape(batch *  seq, -1), reduction='none'
    )
    kl_values = unsummed_kl.sum(-1).view(batch,  seq)
    return kl_values


def _js_div_2d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_2d(p, m) + 0.5 * _kl_div_2d(q, m)


# TODO: add this to the main module


def _reduce_loss(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'none':
        return losses
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)




def js_div_loss_2d(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculate the Jensen-Shannon divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_js_div_2d(target, input), reduction)





def kl_div_loss_2d(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
    r"""Calculate the Kullback-Leibler divergence loss between heatmaps.

    Args:
        input: the input tensor with shape :math:`(B, N, H, W)`.
        target: the target tensor with shape :math:`(B, N, H, W)`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    Examples:
        >>> input = torch.full((1, 1, 2, 4), 0.125)
        >>> loss = js_div_loss_2d(input, input)
        >>> loss.item()
        0.0
    """
    return _reduce_loss(_kl_div_2d(target, input), reduction)



    # The loss functions to use
def log_softmax(x, dim: int, onnx_trace: bool = False):
    return F.log_softmax(x, dim=dim, dtype=torch.float32)

def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)
def computeLoss(net_output,target,rank_alpha,mle_only=False,ignore_index=-100,padding_idx=0):
    nsentences = target.size(0)
    target = target.view(-1)
    
    # -- mle loss
    lprobs = log_softmax(net_output,-1,True)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=ignore_index,
            reduction='none',
        )
    mle_loss = true_token_lprobs.sum()
    if mle_only:
        return mle_loss
    
    with torch.no_grad():
        
        # Make 'the triangle'.
        ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
        ctx_cands_ = (ctx_cands.tril(-1) + ignore_index)
        ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
        ctx_cands = ctx_cands.tril(-1) + ctx_cands_
        
        # Don't include the target for that timestep as a negative target.
        ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), ignore_index)
        negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
    
    # - compute loss
    one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
    
    custom_loss = -torch.log(one_minus_probs)*negative_targets
    custom_loss = custom_loss.sum()
    loss = mle_loss + rank_alpha * custom_loss
    return loss

