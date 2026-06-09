from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_s600_board_hybrid_demo.py"
_SPEC = importlib.util.spec_from_file_location("run_s600_board_hybrid_demo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
board_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(board_demo)


try:
    import torch
except Exception:  # pragma: no cover - exercised only in minimal non-runtime envs
    torch = None


class PlaceholderRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            self.skipTest("torch is required for placeholder renderer smoke tests")

    def _mesh(self) -> dict[str, object]:
        return {
            "pos": torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [6.0, 1.0, 1.0],
                    [1.0, 6.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            "faces": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "vertex_color": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        }

    def test_zbuffer_update_writes_and_prefers_nearer_depth(self) -> None:
        depth = np.zeros((4, 4), dtype=np.float32)
        xyz = np.zeros((4, 4, 3), dtype=np.float32)
        color = np.zeros((4, 4, 3), dtype=np.float32)
        inside = np.array([[True, True], [False, True]])
        far_xyz = np.full((2, 2, 3), 2.0, dtype=np.float32)
        far_color = np.full((2, 2, 3), 0.25, dtype=np.float32)

        wrote = board_demo._update_placeholder_zbuffer(
            depth,
            xyz,
            color,
            xmin=1,
            ymin=1,
            inside=inside,
            z_map=np.full((2, 2), 2.0, dtype=np.float32),
            xyz_map=far_xyz,
            color_map=far_color,
        )

        self.assertEqual(wrote, 3)
        self.assertEqual(int(np.count_nonzero(depth)), 3)
        self.assertTrue(np.allclose(color[1, 1], [0.25, 0.25, 0.25]))

        farther = board_demo._update_placeholder_zbuffer(
            depth,
            xyz,
            color,
            xmin=1,
            ymin=1,
            inside=np.ones((1, 1), dtype=bool),
            z_map=np.array([[3.0]], dtype=np.float32),
            xyz_map=np.full((1, 1, 3), 3.0, dtype=np.float32),
            color_map=np.full((1, 1, 3), 0.75, dtype=np.float32),
        )
        self.assertEqual(farther, 0)
        self.assertAlmostEqual(float(depth[1, 1]), 2.0)
        self.assertTrue(np.allclose(color[1, 1], [0.25, 0.25, 0.25]))

        nearer = board_demo._update_placeholder_zbuffer(
            depth,
            xyz,
            color,
            xmin=1,
            ymin=1,
            inside=np.ones((1, 1), dtype=bool),
            z_map=np.array([[1.0]], dtype=np.float32),
            xyz_map=np.full((1, 1, 3), 1.0, dtype=np.float32),
            color_map=np.full((1, 1, 3), 0.5, dtype=np.float32),
        )
        self.assertEqual(nearer, 1)
        self.assertAlmostEqual(float(depth[1, 1]), 1.0)
        self.assertTrue(np.allclose(color[1, 1], [0.5, 0.5, 0.5]))

    def test_points_only_renderer_returns_expected_shapes_and_xyz_extra(self) -> None:
        extra: dict[str, object] = {}
        color, depth, normal = board_demo.render_placeholder_cpu(
            K=np.eye(3, dtype=np.float32),
            H=8,
            W=8,
            ob_in_cams=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
            output_size=(8, 8),
            extra=extra,
            mesh_tensors=self._mesh(),
            max_faces=0,
            splat_radius=0,
        )

        self.assertEqual(tuple(color.shape), (1, 8, 8, 3))
        self.assertEqual(tuple(depth.shape), (1, 8, 8))
        self.assertIsNone(normal)
        self.assertEqual(color.dtype, torch.float32)
        self.assertEqual(depth.dtype, torch.float32)
        self.assertIn("xyz_map", extra)
        self.assertEqual(tuple(extra["xyz_map"].shape), (1, 8, 8, 3))
        self.assertGreater(int(torch.count_nonzero(depth)), 0)

    def test_triangle_renderer_fills_more_pixels_than_points_only(self) -> None:
        kwargs = dict(
            K=np.eye(3, dtype=np.float32),
            H=8,
            W=8,
            ob_in_cams=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
            output_size=(8, 8),
            mesh_tensors=self._mesh(),
            splat_radius=0,
        )
        _, points_depth, _ = board_demo.render_placeholder_cpu(max_faces=0, **kwargs)
        _, tri_depth, _ = board_demo.render_placeholder_cpu(max_faces=1, **kwargs)

        self.assertGreater(int(torch.count_nonzero(tri_depth)), int(torch.count_nonzero(points_depth)))
        self.assertEqual(tuple(tri_depth.shape), (1, 8, 8))

    def test_depth2xyzmap_fast_matches_reference_formula(self) -> None:
        depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        K = np.array([[2.0, 0.0, 0.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        xyz = board_demo.depth2xyzmap_fast(depth, K)

        expected = np.array(
            [
                [[0.0, 0.0, 0.0], [0.25, -0.25, 1.0]],
                [[-0.5, 0.0, 2.0], [0.75, 0.0, 3.0]],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(xyz, expected))

    def test_depth2xyzmap_fast_reuses_same_frame_cache(self) -> None:
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        K = np.eye(3, dtype=np.float32)

        xyz0 = board_demo.depth2xyzmap_fast(depth, K)
        xyz1 = board_demo.depth2xyzmap_fast(depth, K)
        xyz2 = board_demo.depth2xyzmap_fast(depth.copy(), K)

        self.assertIs(xyz0, xyz1)
        self.assertIsNot(xyz0, xyz2)
        self.assertTrue(np.allclose(xyz0, xyz2))

    def _batch_for_inline_transform(self):
        batch = type("Batch", (), {})()
        batch.rgbAs = torch.full((1, 3, 1, 2), 255.0, dtype=torch.float32)
        batch.rgbBs = torch.full((1, 3, 1, 2), 127.5, dtype=torch.float32)
        batch.poseA = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
        batch.poseA[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        batch.Ks = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        batch.mesh_diameters = torch.tensor([4.0], dtype=torch.float32)
        batch.xyz_mapAs = torch.tensor([[[[1.0, 1.0]], [[2.0, 2.0]], [[4.0, 0.0005]]]], dtype=torch.float32)
        batch.xyz_mapBs = torch.tensor([[[[3.0, 1.0]], [[2.0, 2.0]], [[3.0, 0.0005]]]], dtype=torch.float32)
        return batch

    def test_inline_precomputed_xyz_transform_matches_pair_normalization(self) -> None:
        batch = self._batch_for_inline_transform()

        out = board_demo._inline_precomputed_xyz_transform(batch, {"normalize_xyz": True}, invalid_z_threshold=0.001)

        self.assertIs(out, batch)
        self.assertTrue(torch.allclose(batch.rgbAs, torch.ones_like(batch.rgbAs)))
        self.assertTrue(torch.allclose(batch.rgbBs, torch.full_like(batch.rgbBs, 0.5)))
        expected_a = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[0.5, 0.0]]]], dtype=torch.float32)
        expected_b = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(batch.xyz_mapAs, expected_a))
        self.assertTrue(torch.allclose(batch.xyz_mapBs, expected_b))

    def test_inline_precomputed_xyz_transform_uses_score_invalid_threshold(self) -> None:
        pair_batch = self._batch_for_inline_transform()
        pair_batch.poseA[0, :3, 3] = 0
        pair_batch.mesh_diameters[:] = 2
        pair_batch.xyz_mapAs[:] = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[0.05, 0.2]]]], dtype=torch.float32)
        pair_batch.xyz_mapBs[:] = pair_batch.xyz_mapAs
        score_batch = self._batch_for_inline_transform()
        score_batch.poseA[0, :3, 3] = 0
        score_batch.mesh_diameters[:] = 2
        score_batch.xyz_mapAs[:] = pair_batch.xyz_mapAs.clone()
        score_batch.xyz_mapBs[:] = pair_batch.xyz_mapAs.clone()

        board_demo._inline_precomputed_xyz_transform(pair_batch, {"normalize_xyz": True}, invalid_z_threshold=0.001)
        board_demo._inline_precomputed_xyz_transform(score_batch, {"normalize_xyz": True}, invalid_z_threshold=0.1)

        self.assertAlmostEqual(float(pair_batch.xyz_mapAs[0, 2, 0, 0]), 0.05)
        self.assertAlmostEqual(float(pair_batch.xyz_mapAs[0, 2, 0, 1]), 0.2)
        self.assertAlmostEqual(float(score_batch.xyz_mapAs[0, 2, 0, 0]), 0.0)
        self.assertAlmostEqual(float(score_batch.xyz_mapAs[0, 2, 0, 1]), 0.2)


if __name__ == "__main__":
    unittest.main()
