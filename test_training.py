import unittest
from unittest.mock import patch
from training import select_device


class DeviceTests(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self):
        with patch('training.torch.cuda.is_available', return_value=False):
            self.assertEqual(select_device('auto'), 'cpu')
            with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
                select_device('cuda')

    def test_available_cuda_and_explicit_cpu(self):
        with patch('training.torch.cuda.is_available', return_value=True):
            self.assertEqual(select_device('auto'), 'cuda')
            self.assertEqual(select_device('cuda'), 'cuda')
            self.assertEqual(select_device('cpu'), 'cpu')


if __name__ == '__main__':
    unittest.main()
