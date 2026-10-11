"""Masked native-row WPP models. Inputs are [B,T,R,2] with explicit masks.

``NativeConvTran`` is a compact adaptation of ConvTran (Navid Fouladi et al.,
MIT): its tAPE and post-softmax eRPE mechanisms are retained, while its input
stem is adapted for ragged native WPP CTU rows rather than a fixed sensor grid.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x, mask, dim):
 m=mask.to(x.dtype)
 while m.ndim<x.ndim: m=m.unsqueeze(-1)
 return (x*m).sum(dim)/m.sum(dim).clamp_min(1)


def _masked_softmax(logits, mask, dim=-1):
 """Softmax with zero output for an entirely-invalid row (not NaNs)."""
 masked=logits.masked_fill(~mask, float('-inf'))
 maximum=masked.amax(dim=dim,keepdim=True)
 maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
 exp=torch.exp(masked-maximum)*mask.to(logits.dtype)
 return exp/exp.sum(dim=dim,keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


class NativeBase(nn.Module):
 def __init__(self,width,classes):
  super().__init__(); self.row=nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width)); self.head=nn.Linear(width,classes)
 def rows(self,x,row_mask):
  # Remove adversarial padded values before any biased/linear operation.
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
   # Convolutions have bias/receptive contamination, zero invalid points both sides.
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
  # This is intentionally the pre-ConvTran behavior; ConvTran owns its separate
  # mask-aware native-PTS helper below.
  relative=[[int(value)-int(row[0]) for value in row] for row in pts.detach().cpu().tolist()]
  p=torch.tensor(relative,dtype=torch.float64,device=z.device); scale=p.abs().amax(1,keepdim=True).clamp_min(1); z=z+self.pos((p/scale).to(z.dtype)[...,None])
  z=z*frame_mask[...,None].to(z.dtype)
  z=self.encoder(z,src_key_padding_mask=~frame_mask)*frame_mask[...,None].to(z.dtype)
  z=self.norm(z)*frame_mask[...,None].to(z.dtype)
  return self.head(masked_mean(z,frame_mask,1))


class TemporalAbsolutePositionEncoding(nn.Module):
 """ConvTran tAPE: sinusoidal position scaled by model width/max sequence length."""
 def __init__(self,d_model,max_length):
  super().__init__(); self.d_model=d_model; self.max_length=max_length
  position=torch.arange(max_length,dtype=torch.float32)[:,None]
  frequency=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.0)/d_model))
  table=torch.zeros(max_length,d_model)
  table[:,0::2]=torch.sin(position*frequency*(d_model/max_length))
  table[:,1::2]=torch.cos(position*frequency[:table[:,1::2].shape[1]]*(d_model/max_length))
  self.register_buffer('table',table,persistent=True)
 def forward(self,length,dtype):
  if length>self.max_length: raise ValueError(f'ConvTran received {length} frames, exceeding exact window bound {self.max_length}')
  return self.table[:length].to(dtype=dtype)


class EfficientRelativePositionAttention(nn.Module):
 """ConvTran eRPE attention with a learned offset/head term *after* softmax."""
 def __init__(self,d_model,heads,max_length,dropout):
  super().__init__()
  self.heads=heads; self.head_dim=d_model//heads; self.max_length=max_length
  self.qkv=nn.Linear(d_model,3*d_model); self.output=nn.Linear(d_model,d_model)
  # [relative offset, head]; finite because config derives max_length exactly.
  self.erpe=nn.Parameter(torch.zeros(2*max_length-1,heads))
  nn.init.trunc_normal_(self.erpe,std=.02)
  self.dropout=nn.Dropout(dropout)
 def forward(self,x,mask):
  b,t,d=x.shape
  if t>self.max_length: raise ValueError(f'ConvTran received {t} frames, exceeding exact window bound {self.max_length}')
  q,k,v=self.qkv(x).chunk(3,-1)
  def split(y): return y.view(b,t,self.heads,self.head_dim).transpose(1,2)
  q,k,v=split(q),split(k),split(v)
  scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
  key_mask=mask[:,None,None,:]
  attention=_masked_softmax(scores,key_mask)
  offsets=torch.arange(t,device=x.device)
  offset_index=(offsets[None,:]-offsets[:,None]+self.max_length-1)
  relative=self.erpe[offset_index].permute(2,0,1)[None] # B,H,query,key
  valid=(mask[:,None,:,None] & mask[:,None,None,:])
  # eRPE is deliberately after normalized attention, as in ConvTran.
  attention=(attention+relative*valid.to(attention.dtype))*valid.to(attention.dtype)
  attention=self.dropout(attention)
  z=(attention@v).transpose(1,2).reshape(b,t,d)
  return self.output(z)*mask[...,None].to(z.dtype)


class ConvTranBlock(nn.Module):
 def __init__(self,d_model,heads,max_length,ff_multiplier,dropout):
  super().__init__(); self.norm1=nn.LayerNorm(d_model); self.attention=EfficientRelativePositionAttention(d_model,heads,max_length,dropout); self.norm2=nn.LayerNorm(d_model); self.feedforward=nn.Sequential(nn.Linear(d_model,d_model*ff_multiplier),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model*ff_multiplier,d_model),nn.Dropout(dropout))
 def forward(self,x,mask):
  valid=mask[...,None].to(x.dtype)
  x=(x+self.attention(self.norm1(x)*valid,mask))*valid
  return (x+self.feedforward(self.norm2(x)*valid))*valid


class NativeConvTran(nn.Module):
 """Ragged-row ConvTran adaptation with local spatial-temporal convolution tokenizer."""
 def __init__(self,d_model,classes,heads=4,layers=2,ff_multiplier=2,kernel_size=3,max_frames=30,dropout=.1):
  super().__init__()
  self.max_frames=max_frames
  padding=kernel_size//2
  # Keep stages explicit: each biased convolution is immediately re-masked so
  # right/bottom padded cells cannot become nonzero neighbors for the next one.
  self.tokenizer1=nn.Conv2d(2,d_model,kernel_size,padding=padding)
  self.tokenizer2=nn.Conv2d(d_model,d_model,kernel_size,padding=padding)
  self.tokenizer_activation=nn.GELU()
  self.spatial_score=nn.Linear(d_model,1)
  self.tape=TemporalAbsolutePositionEncoding(d_model,max_frames)
  # Learned native-PTS embedding supplements tAPE and receives relative ticks.
  self.native_pts=nn.Sequential(nn.Linear(1,d_model),nn.Tanh(),nn.Linear(d_model,d_model))
  self.blocks=nn.ModuleList([ConvTranBlock(d_model,heads,max_frames,ff_multiplier,dropout) for _ in range(layers)])
  self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,classes)
 def _relative_pts(self,pts,frame_mask,dtype):
  # Python integer subtraction is intentional: it cannot lose/overflow int64 PTS precision.
  raw=pts.detach().cpu().tolist(); masks=frame_mask.detach().cpu().tolist(); rows=[]
  for row,valid in zip(raw,masks):
   first=next((int(value) for value,ok in zip(row,valid) if ok),0)
   rows.append([int(value)-first if ok else 0 for value,ok in zip(row,valid)])
  p=torch.tensor(rows,device=pts.device,dtype=torch.float64)
  p=p*frame_mask.to(p.dtype)
  scale=p.amax(1,keepdim=True).clamp_min(1)
  return (p/scale).to(dtype)[...,None]
 def forward(self,x,frame_mask=None,row_mask=None,pts=None):
  if row_mask is None: row_mask=torch.ones(x.shape[:3],device=x.device,dtype=torch.bool)
  if frame_mask is None: frame_mask=row_mask.any(-1)
  b,t,_,_=x.shape
  if t>self.max_frames: raise ValueError(f'ConvTran received {t} frames, exceeding exact window bound {self.max_frames}')
  if pts is None: pts=torch.arange(t,device=x.device,dtype=torch.int64)[None].expand(b,-1)
  # [B,T,R,2] -> local temporal/spatial WPP tokenizer. A false frame overrides
  # any row bits it carries. Re-masking between biased convolutions prevents an
  # invalid padded cell's bias activation contaminating its valid neighbor.
  effective_rows=row_mask & frame_mask[...,None]
  valid_rows=effective_rows[...,None].to(x.dtype)
  z=(x*valid_rows).permute(0,3,1,2)
  z=self.tokenizer_activation(self.tokenizer1(z))*valid_rows.permute(0,3,1,2)
  z=self.tokenizer_activation(self.tokenizer2(z))*valid_rows.permute(0,3,1,2)
  z=z.permute(0,2,3,1)
  scores=self.spatial_score(z).squeeze(-1)
  weights=_masked_softmax(scores,effective_rows,dim=2)
  frames=(z*weights[...,None]).sum(2)*frame_mask[...,None].to(z.dtype)
  frames=frames+self.tape(t,frames.dtype)[None]+self.native_pts(self._relative_pts(pts,frame_mask,frames.dtype))
  frames=frames*frame_mask[...,None].to(frames.dtype)
  for block in self.blocks: frames=block(frames,frame_mask)
  frames=self.norm(frames)*frame_mask[...,None].to(frames.dtype)
  return self.head(masked_mean(frames,frame_mask,1))


def make_model(cfg, classes):
 m=cfg['model']
 if m['architecture']=='resnet': return NativeResNet(m['width'],classes,m['resnet_blocks'],m['dropout'])
 if m['architecture']=='transformer': return NativeTransformer(m['d_model'],classes,m['heads'],m['layers'],m['dropout'])
 return NativeConvTran(m['d_model'],classes,m['convtran_heads'],m['convtran_layers'],m['convtran_ff_multiplier'],m['convtran_kernel_size'],m['convtran_max_frames'],m['dropout'])


def aggregate_logits(logits, durations):
 # Callers may supply already-normalized exact-duration ratios. Accumulate in fp64
 # so very large native durations never become fp16/fp32 infinities.
 d=torch.as_tensor(durations,dtype=torch.float64,device=logits.device)
 if not torch.isfinite(d).all() or d.sum() <= 0: raise ValueError('window duration weights must be finite and positive')
 d=d/d.sum()
 return (logits.to(torch.float64)*d[:,None]).sum(0).to(logits.dtype)
