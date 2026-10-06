import unittest
import numpy as np
from world import make_world,make_plan,split,step,policy,episode,probabilities
class WorldTests(unittest.TestCase):
 def test_key_door_and_collision(self):
  self.assertEqual(step((1,2,0),1),(1,2,0))
  self.assertEqual(step((1,2,1),1),(2,2,1))
  self.assertEqual(step((0,3,0),2),(0,4,1))
  self.assertEqual(step((1,0,1),1),(1,0,1))
  self.assertEqual(step((0,0,0),0),(0,0,0))
 def test_reachable_support_and_split(self):
  s,t=make_world();tr,te=split(s)
  self.assertEqual((len(s),t.shape,len(tr),len(te)),(30,(30,4),96,24))
  self.assertFalse(set(tr)&set(te));self.assertEqual(set(tr)|set(te),set(range(120)))
  self.assertEqual([sum(i%4==a for i in te) for a in range(4)],[6]*4)
  self.assertIn(s.index((0,3,0))*4+2,te);self.assertIn(s.index((1,4,0))*4+3,tr)
  self.assertIn(s.index((1,2,1))*4+1,te);self.assertIn(s.index((3,2,1))*4+3,tr)
 def test_oracle_long_horizon_positive_control(self):
  p=make_plan();s=p['states'];t=np.array(p['transitions']);P=np.eye(30)[t]
  for case in p['planning_cases']:
   pi,ts=policy(P,s,case['goal'],16);e=episode(t,s,case['start'],case['goal'],pi,ts)
   self.assertTrue(e['success']);self.assertEqual(e['steps'],case['optimal_steps'])
 def test_probability_shift_and_invalid(self):
  x=np.arange(24,dtype=float).reshape(2,3,4)
  np.testing.assert_allclose(probabilities(x),probabilities(x+100));np.testing.assert_allclose(probabilities(x).sum(-1),1)
  with self.assertRaises(ValueError):probabilities(x*np.nan)
 def test_cycle_and_tie_report(self):
  s,t=make_world();e=episode(t,s,s.index((0,0,0)),(4,4),np.zeros(30,dtype=int),np.full(30,4))
  self.assertTrue(e['cycle']);self.assertFalse(e['success']);self.assertEqual(e['tie_counts'],[4])
if __name__=='__main__':unittest.main()
