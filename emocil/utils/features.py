"""Frozen-encoder features: mean-pooled hidden states of BERT and Qwen (moved
here from _pilot/rp3_frozen.py on 12 Nov 2026, unchanged)."""

import numpy as np
import torch


@torch.no_grad()
def bert_features(model_name, texts, device, state=None, bs=128):
    from transformers import BertModel, BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)
    bert = BertModel.from_pretrained(model_name)
    if state is not None:                         # EmoNet state dict: keys bert.*
        bert.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("bert.")})
    bert.to(device).eval()
    feats = {f"L{i}": [] for i in range(bert.config.num_hidden_layers + 1)}; feats["pooler"] = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=50, return_tensors="pt").to(device)
        out = bert(**enc, output_hidden_states=True)
        m = enc["attention_mask"].unsqueeze(-1).float()
        for i, h in enumerate(out.hidden_states):
            feats[f"L{i}"].append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy().astype(np.float16))
        feats["pooler"].append(out.pooler_output.float().cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in feats.items()}


@torch.no_grad()
def qwen_features(model_name, texts, device, bs=64, dtype=None):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bfloat16, not float16: Qwen's hidden states carry activations of ~7e3 that
    # overflow float16 on some texts (first run, 4 Nov 2026).
    if dtype is None:                             # RP3 default; hs/frozen.py passes float32 on GPUs without bf16
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    lm = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    lm.to(device).eval()
    n_layers = lm.config.num_hidden_layers
    layers = sorted(set(list(range(4, n_layers, 4)) + [n_layers]))
    feats = {f"L{i}": [] for i in layers}
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        out = lm(**enc, output_hidden_states=True)
        m = enc["attention_mask"].unsqueeze(-1).float()
        for i in layers:
            h = out.hidden_states[i].float()
            feats[f"L{i}"].append(((h * m).sum(1) / m.sum(1)).cpu().numpy())   # float32 storage
    return {k: np.concatenate(v) for k, v in feats.items()}
