import copy,random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from wpp_classifier.pipeline import optimizer_group_step

class Video:
 def __init__(self,label,x): self.label,self.x=label,x

def logits(model,video,cfg,dev):
 # Consume RNG so the retry test proves its snapshot was restored.
 return model(video.x.to(dev))+torch.rand((),device=dev)*0

class OptimizationTest(unittest.TestCase):
 def _run(self, injected=False):
  torch.manual_seed(123); np.random.seed(123); random.seed(123)
  model=torch.nn.Linear(2,3,bias=False); opt=torch.optim.SGD(model.parameters(),lr=.1)
  scaler=torch.amp.GradScaler('cuda',enabled=False); dev=torch.device('cpu')
  group=[Video(0,torch.tensor([1.,0.])),Video(2,torch.tensor([0.,1.])),Video(1,torch.tensor([1.,1.]))]
  weight=torch.tensor([1.,2.,4.])
  def injected_oom(attempt,seen):
   if injected and attempt==0 and seen==2: raise RuntimeError('CUDA out of memory: injected')
  result=optimizer_group_step(model,group,{'training':{'amp':False},'_oom_test_microbatch':1 if injected else 3},dev,weight,opt,scaler,logits,injected_oom)
  return model,result,torch.rand(())
 def test_weighted_remainder_matches_direct_video_numerator(self):
  torch.manual_seed(123); initial=torch.nn.Linear(2,3,bias=False).state_dict()
  baseline,result,_=self._run(); loss,correct,event,_=result
  reference=torch.nn.Linear(2,3,bias=False);reference.load_state_dict(initial); group=[Video(0,torch.tensor([1.,0.])),Video(2,torch.tensor([0.,1.])),Video(1,torch.tensor([1.,1.]))]; weight=torch.tensor([1.,2.,4.]); opt=torch.optim.SGD(reference.parameters(),lr=.1)
  terms=[torch.nn.functional.cross_entropy(reference(v.x)[None],torch.tensor([v.label]),reduction='none')[0]*weight[v.label] for v in group]
  direct=sum(terms)/len(group); direct.backward();opt.step()
  self.assertIsNone(event); self.assertAlmostEqual(loss,sum(float(x.detach()) for x in terms),6)
  for a,b in zip(baseline.parameters(),reference.parameters()): self.assertTrue(torch.allclose(a,b,atol=1e-7))
 def test_injected_oom_restores_rng_grads_and_retries_full_group(self):
  baseline,base_result,base_rng=self._run(False); retried,retry_result,retry_rng=self._run(True)
  self.assertEqual(base_result[:2],retry_result[:2]); self.assertEqual(retry_result[2],{'event':'oom_retry','videos':3,'retry_microbatch_size':1})
  self.assertTrue(torch.equal(base_rng,retry_rng))
  for a,b in zip(baseline.parameters(),retried.parameters()): self.assertTrue(torch.allclose(a,b,atol=1e-7))
if __name__=='__main__': unittest.main()
