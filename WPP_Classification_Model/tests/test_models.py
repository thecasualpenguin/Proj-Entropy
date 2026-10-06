import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from wpp_classifier.models import NativeResNet,NativeTransformer,aggregate_logits
class ModelTest(unittest.TestCase):
 def test_masked_garbage_invariant_and_grad(self):
  torch.manual_seed(1); x=torch.randn(1,3,3,2); mask=torch.tensor([[[1,1,0],[1,0,0],[1,1,0]]],dtype=torch.bool); altered=x.clone();altered[~mask]=1e6
  for model in (NativeResNet(8,4,1,0),NativeTransformer(8,4,2,1,0)):
   model.eval(); a=model(x,frame_mask=mask.any(-1),row_mask=mask);b=model(altered,frame_mask=mask.any(-1),row_mask=mask);self.assertTrue(torch.allclose(a,b,atol=1e-5));a.sum().backward();self.assertTrue(any(p.grad is not None for p in model.parameters()))
 def test_duration_aggregation(self):
  z=torch.tensor([[1.,3.],[5.,1.]])
  self.assertTrue(torch.equal(aggregate_logits(z,[1,3]),torch.tensor([4.,1.5])))
 def test_extreme_duration_ratios_stay_finite(self):
  z=torch.tensor([[1.,3.],[5.,1.]])
  out=aggregate_logits(z,[1.0,1.0]) # pipeline passes bounded exact Fraction ratios
  self.assertTrue(torch.isfinite(out).all())
 def test_translated_extreme_pts_preserve_relative_transformer_position(self):
  torch.manual_seed(4); model=NativeTransformer(8,3,2,1,0).eval(); x=torch.randn(1,3,2,2); mask=torch.ones(1,3,2,dtype=torch.bool)
  relative=torch.tensor([[0,1,2]],dtype=torch.int64); high=torch.tensor([[9223372036854775804,9223372036854775805,9223372036854775806]],dtype=torch.int64)
  self.assertTrue(torch.allclose(model(x,frame_mask=mask.any(-1),row_mask=mask,pts=relative),model(x,frame_mask=mask.any(-1),row_mask=mask,pts=high),atol=1e-6))
if __name__=='__main__':unittest.main()
