import numpy as np
import torch

def attention_token_weights(attentions, input_ids, tokenizer):
    """
    Produces an attention-based token-weight visualization.
    This is an evidence visualization, not proof of causality.
    """
    if not attentions:
        return []

    # attentions: tuple[layer] -> [batch, heads, seq, seq]
    layer_mats = [a[0].detach().cpu() for a in attentions]
    # Average over heads and layers.
    matrices = [m.mean(dim=0) for m in layer_mats]
    avg = torch.stack(matrices).mean(dim=0)

    # CLS/self-to-token style summary: first position attending to tokens.
    weights = avg[0].numpy()
    weights = np.maximum(weights, 0)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    result = []
    for token, weight in zip(tokens, weights):
        if token in tokenizer.all_special_tokens:
            continue
        result.append({
            "token": token,
            "weight": float(weight)
        })

    result.sort(key=lambda x: x["weight"], reverse=True)
    return result[:25]
