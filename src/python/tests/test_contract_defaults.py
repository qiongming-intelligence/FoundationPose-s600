from __future__ import annotations

import argparse
import unittest

from foundationpose_s600_tools.export.contract import DEFAULT_SCORE_PAIRS, build_partitions, export_dims


class ContractDefaultTests(unittest.TestCase):
    def test_default_score_pairs_include_l20_and_l32(self) -> None:
        self.assertEqual(DEFAULT_SCORE_PAIRS, (20, 32))

    def test_default_partitions_include_l20_and_l32(self) -> None:
        args = argparse.Namespace(c_in=6, image_size=160, rot_dim=3)
        parts = build_partitions(export_dims(args), DEFAULT_SCORE_PAIRS)

        self.assertIn("refine_net", parts)
        self.assertIn("score_net_L20", parts)
        self.assertIn("score_net_L32", parts)

    def test_refine_batch_partition_uses_distinct_name_and_shape(self) -> None:
        args = argparse.Namespace(c_in=6, image_size=160, rot_dim=3)
        parts = build_partitions(export_dims(args), [], refine_batches=[1, 32])

        self.assertIn("refine_net", parts)
        self.assertIn("refine_net_N32", parts)
        self.assertEqual(parts["refine_net_N32"].onnx_name, "foundationpose_refine_net_N32.onnx")


if __name__ == "__main__":
    unittest.main()
