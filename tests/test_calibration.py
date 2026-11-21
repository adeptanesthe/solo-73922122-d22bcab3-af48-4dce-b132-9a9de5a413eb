"""核心标定逻辑测试：排序枚举、成功/失败/多解歧义、结果留存语义。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.calibration import (
    BoardSpec,
    CameraParams,
    build_order_variants,
    run_calibration,
)

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_calibration.json"


@pytest.fixture(scope="module")
def sample():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    left = CameraParams(
        np.asarray(data["left_camera"]["camera_matrix"], np.float64),
        np.asarray(data["left_camera"]["dist_coeffs"], np.float64),
    )
    right = CameraParams(
        np.asarray(data["right_camera"]["camera_matrix"], np.float64),
        np.asarray(data["right_camera"]["dist_coeffs"], np.float64),
    )
    board = BoardSpec(**data["board"])
    return data, left, right, board


def test_order_variants_rect_board_has_four_unique():
    variants = build_order_variants(7, 5)
    # 非方形棋盘：8 个对称中有 4 个会改变网格尺寸而被去重，剩 4 种合法排序
    assert {v["name"] for v in variants} == {
        "identity", "flip_rows", "flip_cols", "rotate_180"
    }
    ident = next(v for v in variants if v["name"] == "identity")
    np.testing.assert_array_equal(ident["index"], np.arange(35))


def test_order_variants_square_board_has_eight():
    variants = build_order_variants(6, 6)
    assert len(variants) == 8
    # 所有索引都是合法置换
    for v in variants:
        assert sorted(v["index"].tolist()) == list(range(36))


def test_successful_calibration_recovers_ground_truth(sample):
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    outcome = run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960)
    )
    assert outcome.success, outcome.reason
    assert outcome.accepted.order_name == "flip_cols"
    assert outcome.passing_candidates == ["flip_cols"]

    gt = data["_ground_truth"]
    R_gt = np.asarray(gt["R_left_to_right"])
    T_gt = np.asarray(gt["T_left_to_right"])
    np.testing.assert_allclose(outcome.accepted.R, R_gt, atol=2e-3)
    np.testing.assert_allclose(outcome.accepted.T.ravel(), T_gt, atol=0.5)

    # 每对图都达标，且记录了所采用的排序
    assert len(outcome.accepted.pairs) == 5
    assert all(p.passed and p.order_name == "flip_cols" for p in outcome.accepted.pairs)
    assert all(p.per_pair_rms <= 0.6 for p in outcome.accepted.pairs)
    # 独立解算与共用外参应高度一致（单对 PnP 受 0.12px 噪声影响，容差 1°）
    assert all(p.rot_deg < 1.0 for p in outcome.accepted.pairs)


def test_no_variant_passes_with_noisy_pair(sample):
    data, left, right, board = sample
    # 含 pair_06（大噪声），严格阈值下没有任何排序能让所有图对达标
    ids = [p["pair_id"] for p in data["image_pairs"]]
    outcome = run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960)
    )
    assert not outcome.success
    assert outcome.passing_candidates == []
    assert "没有任何一种角点排序" in outcome.reason
    # 诊断信息仍然完整给出
    assert len(outcome.all_candidates) == 4
    best = outcome.all_candidates[0]
    assert best.order_name == "flip_cols"  # 残差最小的仍是正确排序
    assert any(not p.passed for p in best.pairs)


def test_loose_threshold_allows_multiple_variants(sample):
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:4]]
    # 阈值极松（>= 错误排序的最差图对残差）：多种排序同时“达标”
    # -> 多解歧义，拒绝更新
    outcome = run_calibration(
        left, right, board, data["image_pairs"], ids, 100.0, (1280, 960)
    )
    assert not outcome.success
    assert len(outcome.passing_candidates) >= 2
    assert "种排序" in outcome.reason


def test_min_three_pairs_required(sample):
    data, left, right, board = sample
    with pytest.raises(ValueError, match="至少需要选择 3 对"):
        run_calibration(
            left, right, board, data["image_pairs"],
            ["pair_01", "pair_02"], 0.6, (1280, 960),
        )


def test_corner_count_mismatch_rejected(sample):
    data, left, right, board = sample
    broken = json.loads(json.dumps(data["image_pairs"]))
    broken[0]["right_pts"] = broken[0]["right_pts"][:-3]
    with pytest.raises(ValueError, match="角点数"):
        run_calibration(
            left, right, board, broken,
            [p["pair_id"] for p in broken[:4]], 0.6, (1280, 960),
        )


def test_wrong_order_produces_large_residual(sample):
    """对照实验：人为把右图按 identity 解释（忽略真实倒序），误差必须显著更大。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    outcome = run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960)
    )
    by_name = {c.order_name: c for c in outcome.all_candidates}
    assert by_name["flip_cols"].overall_rms < by_name["identity"].overall_rms / 50


def test_synthetic_rotation_180_order_detected():
    """自造 rotate_180 倒序数据，验证枚举能识别 identity/flip 之外的排序族。"""
    rng = np.random.default_rng(7)
    rows, cols, sq = 6, 5, 20.0
    n = rows * cols
    board = BoardSpec(rows, cols, sq)
    obj = board.object_points()
    K = np.array([[800, 0, 500], [0, 800, 400], [0, 0, 1]], np.float64)
    cam = CameraParams(K, np.zeros(5))
    R_true = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))[0]
    T_true = np.array([[-90.0], [0.0], [0.0]])

    idx = np.empty(n, int)
    for r in range(rows):
        for c in range(cols):
            idx[r * cols + c] = (rows - 1 - r) * cols + (cols - 1 - c)

    pairs = []
    for k in range(4):
        rvec = np.array([0.1 * k, 0.2 * (k - 1), 0.05])
        tvec = np.array([[20 * k], [10 * k], [800 + 50 * k]], np.float64)
        ql, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
        Rr = R_true @ cv2.Rodrigues(rvec)[0]
        qr, _ = cv2.projectPoints(
            obj, cv2.Rodrigues(Rr)[0], R_true @ tvec + T_true, K, np.zeros(5)
        )
        ql = ql.reshape(-1, 2) + rng.normal(0, 0.08, (n, 2))
        qr = qr.reshape(-1, 2)[idx] + rng.normal(0, 0.08, (n, 2))
        pairs.append({
            "pair_id": f"s{k}",
            "left_pts": ql.tolist(),
            "right_pts": qr.tolist(),
        })

    outcome = run_calibration(
        cam, cam, board, pairs, [p["pair_id"] for p in pairs], 0.5, (1000, 800)
    )
    assert outcome.success, outcome.reason
    assert outcome.accepted.order_name == "rotate_180"
