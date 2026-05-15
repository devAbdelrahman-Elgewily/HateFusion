"""Quick smoke test for NB05: load warm-started LoRA in trainable mode, run 5 train steps + 1 val pass."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

ROOT = Path('/teamspace/studios/this_studio/HateFusion')
ADAPTER_DIR = ROOT / 'models' / 'roberta_pretrain'
LABELS_CSV = ROOT / 'data' / 'processed' / 'labels_parsed.csv'
SPLITS_DIR = ROOT / 'data' / 'MMHS150K' / 'splits'

device = torch.device('cuda')
print('1. loading tokenizer from', ADAPTER_DIR)
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
print('   vocab:', tok.vocab_size)

print('2. loading base + warm-started LoRA (is_trainable=True)')
base = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-2022-154m')
enc = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=True)
head = nn.Linear(base.config.hidden_size, 1)
class M(nn.Module):
    def __init__(self): super().__init__(); self.encoder=enc; self.head=head
    def forward(self, ids, mask):
        o = self.encoder(input_ids=ids, attention_mask=mask)
        return self.head(o.last_hidden_state[:,0])
m = M().to(device)
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
n_lora_train = sum(p.numel() for n,p in m.named_parameters() if 'lora_' in n and p.requires_grad)
print(f'   trainable total: {n_train:,} | trainable LoRA: {n_lora_train:,}')
assert n_lora_train > 0, 'LoRA frozen — is_trainable=True did not stick'

print('3. sampling a tiny train batch from labels_parsed.csv + train_ids.txt')
df = pd.read_csv(LABELS_CSV)
train_ids = set(int(l.strip()) for l in open(SPLITS_DIR/'train_ids.txt') if l.strip())
sub = df[df['tweet_id'].isin(train_ids)].head(80).reset_index(drop=True)
print('   sub:', sub.shape, 'T1 mean:', sub['T1'].mean())

class DS(Dataset):
    def __init__(self, d):
        e = tok(d['tweet_text'].astype(str).tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        self.ids = e['input_ids']; self.am = e['attention_mask']
        self.y = torch.tensor(d['T1'].to_numpy(dtype=np.float32))
    def __len__(self): return self.y.size(0)
    def __getitem__(self, i): return {'input_ids': self.ids[i], 'attention_mask': self.am[i], 'label': self.y[i]}

ds = DS(sub)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

print('4. training 5 steps with fp16 + GradScaler + grad_accum')
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')
opt.zero_grad()
t0 = time.time()
for step, b in enumerate(loader):
    ids = b['input_ids'].to(device); mk = b['attention_mask'].to(device); y = b['label'].to(device)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        lg = m(ids, mk)
        loss = F.binary_cross_entropy_with_logits(lg.view(-1), y) / 4  # grad_accum=4
    scaler.scale(loss).backward()
    if (step+1) % 4 == 0:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    print(f'   step {step+1}: loss={loss.item()*4:.4f}')
    if step >= 4: break
print(f'   5 steps in {time.time()-t0:.1f}s')

print('5. peak gpu mem:', f'{torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
print('SMOKE_OK')
