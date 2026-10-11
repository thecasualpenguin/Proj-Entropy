import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from wpp_classifier.models import NativeResNet,NativeTransformer,NativeConvTran,aggregate_logits
class ModelTest(unittest.TestCase):
 def test_masked_garbage_invariant_and_grad(self):
  torch.manual_seed(1); x=torch.randn(1,3,3,2); mask=torch.tensor([[[1,1,0],[1,0,0],[1,1,0]]],dtype=torch.bool); altered=x.clone();altered[~mask]=1e6
  for model in (NativeResNet(8,4,1,0),NativeTransformer(8,4,2,1,0),NativeConvTran(8,4,2,1,2,3,3,0)):
   model.eval(); a=model(x,frame_mask=mask.any(-1),row_mask=mask);b=model(altered,frame_mask=mask.any(-1),row_mask=mask);self.assertTrue(torch.allclose(a,b,atol=1e-5));a.sum().backward();self.assertTrue(any(p.grad is not None for p in model.parameters()))
 def test_convtran_variable_rows_pts_all_invalid_and_bound(self):
  torch.manual_seed(2); model=NativeConvTran(8,3,2,1,2,3,3,0).eval()
  x=torch.randn(2,3,4,2); mask=torch.tensor([[[1,1,1,1],[1,1,0,0],[1,0,0,0]],[[0,0,0,0],[1,1,1,0],[1,1,0,0]]],dtype=torch.bool)
  pts=torch.tensor([[100,101,102],[99,100,101]])
  out=model(x,mask.any(-1),mask,pts); self.assertTrue(torch.isfinite(out).all())
  padded=x.clone(); padded[~mask]=123456.; self.assertTrue(torch.allclose(out,model(padded,mask.any(-1),mask,pts),atol=1e-5))
  # PTS embedding is learned and receives relative timestamps, not absolute origin.
  self.assertTrue(torch.allclose(out[0],model(x[:1],mask[:1].any(-1),mask[:1],pts[:1]+999999)[0],atol=1e-5))
  # Irregular native PTS changes the learned position input (while translation does not).
  self.assertFalse(torch.allclose(out[0],model(x[:1],mask[:1].any(-1),mask[:1],torch.tensor([[100,101,104]]))[0]))
  all_invalid=torch.zeros(1,3,4,dtype=torch.bool);self.assertTrue(torch.isfinite(model(x[:1],all_invalid.any(-1),all_invalid,torch.tensor([[4,5,6]])).all()))
  with self.assertRaisesRegex(ValueError,'exceeding exact window bound'): model(torch.randn(1,4,1,2),torch.ones(1,4,dtype=torch.bool),torch.ones(1,4,1,dtype=torch.bool))
 def test_convtran_compact_and_right_padded_tokenizers_are_invariant(self):
  torch.manual_seed(9); model=NativeConvTran(8,3,2,1,2,3,3,0).eval()
  compact=torch.randn(1,2,2,2); compact_rows=torch.ones(1,2,2,dtype=torch.bool); compact_pts=torch.tensor([[50,51]])
  expected=model(compact,compact_rows.any(-1),compact_rows,compact_pts)
  padded=torch.full((1,3,3,2),9e5); padded[:,:2,:2]=compact
  # The last frame deliberately claims live rows, but its frame mask makes all
  # of them invalid; the final column is ordinary right padding.
  rows=torch.tensor([[[1,1,0],[1,1,0],[1,1,1]]],dtype=torch.bool); frames=torch.tensor([[1,1,0]],dtype=torch.bool)
  padded.requires_grad_(); actual=model(padded,frames,rows,torch.tensor([[50,51,999999]]))
  self.assertTrue(torch.allclose(expected,actual,atol=1e-5));actual.sum().backward()
  effective=rows & frames[...,None]
  self.assertTrue(torch.equal(padded.grad[~effective],torch.zeros_like(padded.grad[~effective])))
 def test_all_models_handle_all_invalid_windows(self):
  x=torch.randn(1,2,3,2); rows=torch.zeros(1,2,3,dtype=torch.bool); frames=rows.any(-1)
  for model in (NativeResNet(8,2,1,0),NativeTransformer(8,2,2,1,0),NativeConvTran(8,2,2,1,2,3,3,0)):
   out=model(x,frames,rows,torch.tensor([[17,18]]));self.assertTrue(torch.isfinite(out).all());out.sum().backward()
 def test_duration_aggregation(self):
  z=torch.tensor([[1.,3.],[5.,1.]])
  self.assertTrue(torch.equal(aggregate_logits(z,[1,3]),torch.tensor([4.,1.5])))
 def test_extreme_duration_ratios_stay_finite(self):
  z=torch.tensor([[1.,3.],[5.,1.]])
  self.assertTrue(torch.isfinite(aggregate_logits(z,[1.0,1.0])).all())
 def test_transformer_pts_mask_behavior_matches_pre_convtran_baseline(self):
  torch.manual_seed(12); model=NativeTransformer(8,3,2,1,0).eval(); x=torch.randn(1,3,2,2)
  rows=torch.tensor([[[1,1],[1,1],[1,0]]],dtype=torch.bool); frames=torch.tensor([[1,0,1]],dtype=torch.bool); pts=torch.tensor([[100,10**9,104]])
  actual=model(x,frames,rows,pts)
  # Explicitly reproduce the established implementation: anchor at pts[:, 0]
  # and scale over every sequence position before temporal frame masking.
  z=model.rows(x,rows); relative=[[int(value)-int(row[0]) for value in row] for row in pts.tolist()]
  p=torch.tensor(relative,dtype=torch.float64); scale=p.abs().amax(1,keepdim=True).clamp_min(1)
  z=z+model.pos((p/scale).to(z.dtype)[...,None]);z=z*frames[...,None].to(z.dtype)
  z=model.encoder(z,src_key_padding_mask=~frames)*frames[...,None].to(z.dtype);z=model.norm(z)*frames[...,None].to(z.dtype)
  expected=model.head((z*frames[...,None]).sum(1)/frames.sum(1,keepdim=True))
  self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
 def test_translated_extreme_pts_preserve_relative_transformer_position(self):
  torch.manual_seed(4); model=NativeTransformer(8,3,2,1,0).eval(); x=torch.randn(1,3,2,2); mask=torch.ones(1,3,2,dtype=torch.bool)
  relative=torch.tensor([[0,1,2]],dtype=torch.int64); high=torch.tensor([[9223372036854775804,9223372036854775805,9223372036854775806]],dtype=torch.int64)
  self.assertTrue(torch.allclose(model(x,frame_mask=mask.any(-1),row_mask=mask,pts=relative),model(x,frame_mask=mask.any(-1),row_mask=mask,pts=high),atol=1e-6))
if __name__=='__main__':unittest.main()
