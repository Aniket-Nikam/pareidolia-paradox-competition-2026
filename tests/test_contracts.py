import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from ml.preprocessing import normalize_solar_azimuth
from ml.submission import validate_submission
from validate_submission import check


class ContractTests(unittest.TestCase):
    def test_rotation_sign(self):
        a=np.zeros((9,9),dtype=np.uint8); a[1,4]=255
        b=np.asarray(normalize_solar_azimuth(Image.fromarray(a),90,border_mode='reflect',resample=Image.Resampling.NEAREST))
        self.assertEqual(b[4,7],255)
        self.assertEqual(b[4,1],0)

    def test_no_new_black_fill(self):
        for angle in [0,37,45,90,213,359.9]:
            a=np.asarray(normalize_solar_azimuth(Image.new('L',(256,256),127),angle,border_mode='reflect'))
            self.assertEqual(a.shape,(256,256))
            self.assertTrue((a==127).all())

    def test_fractional_labels_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission(pd.DataFrame({'image_id':['a.png'],'label':[0.9]}),expected_rows=1)

    def test_order_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission(pd.DataFrame({'image_id':['b','a'],'label':[0,1]}),expected_rows=2,expected_ids=['a','b'])

    def test_csv_roundtrip_and_bad_header(self):
        # Schema-only fixtures are not training or competition predictions.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); ids=[f'eval_{i:05d}.png' for i in range(1,2001)]
            pd.DataFrame({'image_id':ids,'sun_azimuth_angle':0}).to_csv(root/'metadata.csv',index=False)
            pd.DataFrame({'image_id':ids,'label':np.zeros(2000,dtype=int)}).to_csv(root/'example.csv',index=False)
            self.assertTrue(check(root/'example.csv',root/'metadata.csv')['valid'])
            pd.DataFrame({'image_id':ids,'label':np.zeros(2000,dtype=int)}).to_csv(root/'example.csv',index=True)
            with self.assertRaises(ValueError): check(root/'example.csv',root/'metadata.csv')


if __name__=='__main__': unittest.main()
