import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from traffic_forecasting import multihorizon_cuda as cuda
from traffic_forecasting import multihorizon_mps as mps
from traffic_forecasting import multihorizon_study as study


class CUDAAdapterTests(unittest.TestCase):
    def test_import_does_not_modify_active_studies(self):
        self.assertIsNot(study.build_model, mps.build_model)
        self.assertNotIn('traffic_forecasting/multihorizon_cuda.py', study.CODE_FILES)

    def test_cuda_protocol_preserves_all_model_trials_and_budgets(self):
        root = Path(__file__).resolve().parents[2] / 'configs/forecasting'
        original = json.loads((root / 'multihorizon.json').read_text())
        server = json.loads((root / 'multihorizon_cuda.json').read_text())
        for key in ['datasets', 'seeds', 'trials', 'epochs', 'min_epochs',
                    'patience', 'batch_size', 'train_stride', 'eval_stride']:
            self.assertEqual(server[key], original[key], key)
        self.assertEqual(server['device'], 'cuda')
        self.assertEqual(len(server['datasets']) * len(server['seeds']) * len(server['trials']), 42)

    def test_cuda_unavailable_fails_without_cpu_fallback(self):
        with patch.object(cuda.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                cuda.install()

    def test_deterministic_workspace_must_be_set_before_python(self):
        with patch.object(cuda.torch.cuda, 'is_available', return_value=True), \
             patch.object(cuda.torch.cuda, 'device_count', return_value=1), \
             patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': 'invalid'}):
            with self.assertRaisesRegex(RuntimeError, 'CUBLAS_WORKSPACE_CONFIG'):
                cuda.install()


if __name__ == '__main__':
    unittest.main()
