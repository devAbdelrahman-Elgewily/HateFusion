"""Smoke test for NB04 stack: tokenizer + model + LoRA + 1 forward pass.

Verifies the most failure-prone bits before committing to a 15-minute training run:
- twitter-roberta-base-2022-154m downloads and loads
- target_modules=['query','value'] match RoBERTa attention names
- PEFT + transformers 5.x interop still works
- fp16 forward pass produces logits with the right shape
"""
import sys, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

MODEL = 'cardiffnlp/twitter-roberta-base-2022-154m'
print(f'loading tokenizer + model: {MODEL}', flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
backbone = AutoModel.from_pretrained(MODEL)
print('  hidden_size:', backbone.config.hidden_size, '| layers:', backbone.config.num_hidden_layers)

lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query', 'value'], bias='none',
)
enc = get_peft_model(backbone, lora_cfg)
total = sum(p.numel() for p in enc.parameters())
train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
print(f'  total params: {total:,} | trainable (lora only): {train:,} ({100*train/total:.3f}%)')
enc.print_trainable_parameters()

head = nn.Linear(backbone.config.hidden_size, 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc.to(device); head.to(device)

texts = [
    'Why is #aussietv so white? #MKR',
    '@someone you are an idiot',
    'just had pancakes for breakfast',
]
batch = tok(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)

with torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else torch.no_grad():
    out = enc(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    cls = out.last_hidden_state[:, 0]
    logits = head(cls)
print('  forward OK, logits shape:', tuple(logits.shape), 'dtype:', logits.dtype)
print('  logits sample:', logits[0].detach().float().cpu().tolist())
print('SMOKE_OK', flush=True)
