"""Masked native-row WPP models. Inputs are [B,T,R,2] with explicit masks."""
from __future__ import annotations
import torch
from torch import nn

def masked_mean(x, mask, dim):
 m=mask.to(x.dtype)
 while m.ndim<x.ndim: m=m.unsqueeze(-1)
 return (x*m).sum(dim)/m.sum(dim).clamp_min(1)

class NativeBase(nn.Module):
 def __init__(self,width,classes):
  super().__init__(); self.row=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width)); self.head=nn.Linear(width,classes)
 def rows(self,x,row_mask):
  # remove masked adversarial values before any biased/linear operation.
  x=x*row_mask[...,None].to(x.dtype); z=self.row(x)*row_mask[...,None].to(x.dtype)
  return masked_mean(z,row_mask,2)

class NativeResNet(NativeBase):
 def __init__(self,width,classes,blocks=3,dropout=.1):
  super().__init__(width,classes); self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=1),nn.GELU(),nn.Dropout(dropout),nn.Conv1d(width,width,3,padding=1)) for _ in range(blocks)]); self.norm=nn.LayerNorm(width)
 def forward(self,x,frame_mask=None,row_mask=None,pts=None):
  if row_mask is None: row_mask=torch.ones(x.shape[:3],device=x.device,dtype=torch.bool)
  if frame_mask is None: frame_mask=row_mask.any(-1)
  z=self.rows(x,row_mask)*frame_mask[...,None].to(x.dtype)
  z=z.transpose(1,2)
  for block in self.blocks:
   # convolutions have bias/receptive contamination, zero invalid points both sides.
   y=block(z)*frame_mask[:,None,:].to(z.dtype); z=(z+y)*frame_mask[:,None,:].to(z.dtype)
  z=self.norm(z.transpose(1,2))*frame_mask[...,None].to(z.dtype)
  return self.head(masked_mean(z,frame_mask,1))

class NativeTransformer(NativeBase):
 def __init__(self,width,classes,heads=4,layers=2,dropout=.1):
  super().__init__(width,classes); self.pos=nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width)); layer=nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*2,dropout=dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False); self.norm=nn.LayerNorm(width)
 def forward(self,x,frame_mask=None,row_mask=None,pts=None):
  if row_mask is None: row_mask=torch.ones(x.shape[:3],device=x.device,dtype=torch.bool)
  if frame_mask is None: frame_mask=row_mask.any(-1)
  z=self.rows(x,row_mask)
  if pts is None: pts=torch.arange(x.shape[1],device=x.device)[None].expand(x.shape[0],-1)
  # PTS position is structural and centered to avoid huge timestamp magnitude.
  # Subtract as Python integers before any float cast: float64 absolute int64 PTS
  # near 2**63 lose one-tick spacing, while tensor int64 subtraction can overflow.
  relative=[[int(value)-int(row[0]) for value in row] for row in pts.detach().cpu().tolist()]
  p=torch.tensor(relative,dtype=torch.float64,device=z.device); scale=p.abs().amax(1,keepdim=True).clamp_min(1); z=z+self.pos((p/scale).to(z.dtype)[...,None])
  z=z*frame_mask[...,None].to(z.dtype)
  z=self.encoder(z,src_key_padding_mask=~frame_mask)*frame_mask[...,None].to(z.dtype)
  z=self.norm(z)*frame_mask[...,None].to(z.dtype)
  return self.head(masked_mean(z,frame_mask,1))

def make_model(cfg, classes):
 m=cfg['model']
 if m['architecture']=='resnet': return NativeResNet(m['width'],classes,m['resnet_blocks'],m['dropout'])
 return NativeTransformer(m['d_model'],classes,m['heads'],m['layers'],m['dropout'])

def aggregate_logits(logits, durations):
 # Callers may supply already-normalized exact-duration ratios. Accumulate in fp64
 # so very large native durations never become fp16/fp32 infinities.
 d=torch.as_tensor(durations,dtype=torch.float64,device=logits.device)
 if not torch.isfinite(d).all() or d.sum() <= 0: raise ValueError('window duration weights must be finite and positive')
 d=d/d.sum()
 return (logits.to(torch.float64)*d[:,None]).sum(0).to(logits.dtype)
