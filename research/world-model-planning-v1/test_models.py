import unittest
import torch
from run import Direct,Energy
class ModelTests(unittest.TestCase):
 def test_shapes_counts_and_finite_gradients_invented_inputs(self):
  torch.set_num_threads(2)
  for cls,expected in [(Direct,7198),(Energy,7243)]:
   torch.manual_seed(999);m=cls();x=torch.zeros(2,16);c=torch.zeros(30,12)
   y=m(x,c);self.assertEqual(y.shape,(2,30));self.assertEqual(sum(p.numel() for p in m.parameters()),expected)
   torch.nn.functional.cross_entropy(y,torch.tensor([0,1])).backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_energy_output_is_negative_joint_score(self):
  m=Energy();x=torch.zeros(1,16);c=torch.zeros(30,12)
  with torch.no_grad():
   for p in m.parameters():p.zero_()
   m.net[-1].bias.fill_(2)
  self.assertTrue(torch.equal(m(x,c),torch.full((1,30),-2.)))
if __name__=='__main__':unittest.main()
