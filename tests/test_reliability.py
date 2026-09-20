import unittest
import numpy as np
from reliability_common import calibration,group_bootstrap,summary


class ReliabilityTests(unittest.TestCase):
    def test_group_bootstrap_preserves_mixed_groups(self):
        y=np.array([0,1,0,1]); groups=np.array([0,0,1,1]); pred=np.array([0,0,1,1])
        draws=group_bootstrap(y,groups,[pred,pred],resamples=1000)
        self.assertTrue(np.allclose(draws,.5))
        self.assertTrue(np.array_equal(draws[:,0],draws[:,1]))

    def test_perfect_predictions_interval(self):
        y=np.array([0,0,1,1]); draws=group_bootstrap(y,np.arange(4),[y],resamples=100)
        self.assertTrue(np.array_equal(draws,np.ones((100,1))))

    def test_calibration_definition(self):
        result=calibration(np.array([0,1]),np.array([.2,.8]))
        self.assertAlmostEqual(result['brier'],.04)
        self.assertAlmostEqual(result['log_loss'],-np.log(.8))
        self.assertAlmostEqual(result['ece_15_equal_width'],.2)

    def test_confusion_convention(self):
        r=summary(np.array([0,0,1,1]),np.array([0,1,0,1]))
        self.assertEqual(r['false_positive_rise'],1)
        self.assertEqual(r['false_negative_rise'],1)
        self.assertEqual(r['balanced_accuracy'],.5)


if __name__=='__main__': unittest.main()
