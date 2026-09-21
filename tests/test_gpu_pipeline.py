import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gpu.common import ImagePolicy, build_model, preprocess_pil


class GpuPipelineTests(unittest.TestCase):
    def test_spatial_policies_have_fixed_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "sample.png"
            Image.fromarray(np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)).save(source)
            for angle in [0, 45, 123.4, 359.9]:
                for policy in ["reflect", "crop180", "circle180"]:
                    result = preprocess_pil(source, angle, ImagePolicy(policy))
                    self.assertEqual(result.size, (224, 224))

    def test_circular_mask_boundary_is_angle_independent(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / "sample.png"
            Image.fromarray(np.full((256, 256), 255, dtype=np.uint8)).save(source)
            first = np.asarray(preprocess_pil(source, 0, ImagePolicy("circle180"))) > 0
            second = np.asarray(preprocess_pil(source, 45, ImagePolicy("circle180"))) > 0
            self.assertTrue(np.array_equal(first, second))

    def test_complete_network_is_trainable(self):
        for architecture in ["efficientnet_b0", "resnet18"]:
            model = build_model(architecture, pretrained=False)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
