from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from foundationpose_s600_tools.runtime.hrt import TensorSpec
from foundationpose_s600_tools.runtime.refine_bpu import RefineNetBpu
from foundationpose_s600_tools.runtime.score_bpu import ScoreNetBpu


class FakeScoreRunner:
    def __init__(self, l: int = 4, tail: tuple[int, ...] = (2, 3, 3)) -> None:
        self.input_specs = (
            TensorSpec("A", (l, *tail)),
            TensorSpec("B", (l, *tail)),
        )
        self.output_specs = (TensorSpec("score_logit", (1, l)),)
        self.calls: list[dict[str, np.ndarray]] = []

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        l = self.output_specs[0].shape[1]
        return {"score_logit": np.arange(l, dtype=np.float32).reshape(1, l)}


class FakeRefineRunner:
    def __init__(self, n: int = 1, tail: tuple[int, ...] = (2, 3, 3), rot_dim: int = 6) -> None:
        self.input_specs = (
            TensorSpec("A", (n, *tail)),
            TensorSpec("B", (n, *tail)),
        )
        self.output_specs = (
            TensorSpec("trans", (1, 3)),
            TensorSpec("rot", (1, rot_dim)),
        )
        self.calls: list[dict[str, np.ndarray]] = []

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        rot_dim = self.output_specs[1].shape[1]
        return {
            "trans": np.ones((1, 3), dtype=np.float32) * len(self.calls),
            "rot": np.ones((1, rot_dim), dtype=np.float32) * (10 + len(self.calls)),
        }


class RuntimeAdapterTests(unittest.TestCase):
    def test_score_strict_exact_l_runs_once(self) -> None:
        runner = FakeScoreRunner(l=4)
        adapter = ScoreNetBpu(runner=runner, chunk_size=4, mode="strict")
        a = np.zeros((4, 2, 3, 3), dtype=np.float32)
        b = np.ones((4, 2, 3, 3), dtype=np.float32)

        out = adapter.predict_numpy(a, b)

        self.assertEqual(out.shape, (1, 4))
        self.assertTrue(np.allclose(out, [[0, 1, 2, 3]]))
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0]["A"].shape, (4, 2, 3, 3))

    def test_score_strict_shorter_input_fails_before_runner_call(self) -> None:
        runner = FakeScoreRunner(l=4)
        adapter = ScoreNetBpu(runner=runner, chunk_size=4, mode="strict")
        a = np.zeros((3, 2, 3, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "requires exactly 4 candidates"):
            adapter.predict_numpy(a, a)

        self.assertEqual(runner.calls, [])

    def test_score_smoke_pad_shorter_input_pads_once_and_clips_output(self) -> None:
        runner = FakeScoreRunner(l=4)
        adapter = ScoreNetBpu(runner=runner, chunk_size=4, mode="smoke_pad", pad_mode="repeat_last")
        a = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3)
        b = a + 100

        out = adapter.predict_numpy(a, b)

        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(np.allclose(out, [[0, 1, 2]]))
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0]["A"].shape, (4, 2, 3, 3))
        self.assertTrue(np.allclose(runner.calls[0]["A"][3], a[2]))
        self.assertTrue(np.allclose(runner.calls[0]["B"][3], b[2]))

    def test_score_smoke_pad_rejects_oversized_input_instead_of_chunking(self) -> None:
        runner = FakeScoreRunner(l=4)
        adapter = ScoreNetBpu(runner=runner, chunk_size=4, mode="smoke_pad")
        a = np.zeros((5, 2, 3, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "at most 4 candidates"):
            adapter.predict_numpy(a, a)

        self.assertEqual(runner.calls, [])

    def test_score_contract_tail_shape_is_not_hard_coded(self) -> None:
        runner = FakeScoreRunner(l=4, tail=(4, 5, 6))
        adapter = ScoreNetBpu(runner=runner, chunk_size=4)
        ok = np.zeros((4, 4, 5, 6), dtype=np.float32)
        bad = np.zeros((4, 6, 160, 160), dtype=np.float32)

        self.assertEqual(adapter.predict_numpy(ok, ok).shape, (1, 4))
        with self.assertRaisesRegex(ValueError, "tail shape"):
            adapter.predict_numpy(bad, bad)

    def test_score_chunk_size_must_match_contract_l(self) -> None:
        runner = FakeScoreRunner(l=4)

        with self.assertRaisesRegex(ValueError, "does not match .* contract L=4"):
            ScoreNetBpu(runner=runner, chunk_size=3)

    def test_refine_rot_dim_six_is_preserved(self) -> None:
        runner = FakeRefineRunner(rot_dim=6)
        adapter = RefineNetBpu(runner=runner)
        a = np.zeros((2, 2, 3, 3), dtype=np.float32)

        out = adapter.predict_numpy(a, a)

        self.assertEqual(out["trans"].shape, (2, 3))
        self.assertEqual(out["rot"].shape, (2, 6))
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0]["A"].shape, (1, 2, 3, 3))

    def test_refine_contract_tail_shape_is_not_hard_coded(self) -> None:
        runner = FakeRefineRunner(tail=(4, 5, 6), rot_dim=3)
        adapter = RefineNetBpu(runner=runner)
        ok = np.zeros((1, 4, 5, 6), dtype=np.float32)
        bad = np.zeros((1, 6, 160, 160), dtype=np.float32)

        self.assertEqual(adapter.predict_numpy(ok, ok)["rot"].shape, (1, 3))
        with self.assertRaisesRegex(ValueError, "tail shape"):
            adapter.predict_numpy(bad, bad)

    def test_refine_rejects_non_unit_compiled_batch(self) -> None:
        runner = FakeRefineRunner(n=2)

        with self.assertRaisesRegex(ValueError, "expects an N=1 compiled contract"):
            RefineNetBpu(runner=runner)

    def test_board_demo_installs_cpu_patch_before_upstream_imports(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_s600_board_hybrid_demo.py"
        text = script.read_text(encoding="utf-8")
        self.assertLess(text.index("        patch_torch_cpu_cuda_compat()"), text.index("from estimater import"))


if __name__ == "__main__":
    unittest.main()
