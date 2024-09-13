import unittest

import torch


class TestPackageVersions(unittest.TestCase):
    def test_pytorch_version(self):
        assert torch.__version__ in ["2.2.2", "2.2.2+cu121"]
