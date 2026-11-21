"""核心标定逻辑测试：排序枚举、成功/失败/多解歧义、结果留存语义。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.calibration import (
    BoardSpec,
    CalibrationCache,
    CameraParams,
    build_order_variants,
    build_solution_cache_key,
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


# ---- 复用与失效（缓存与左相机姿态共享） ------------------------------------


def _snapshot(outcome):
    """把一次结果序列化成可直接比较的纯数据（含 R/T、逐对 RMSE 与达标判定）。"""
    def cand(c):
        if c is None:
            return None
        return {
            "order_name": c.order_name,
            "order_description": c.order_description,
            "overall_rms": c.overall_rms,
            "R": None if c.R is None else np.asarray(c.R).tolist(),
            "T": None if c.T is None else np.asarray(c.T).tolist(),
            "passed": c.passed,
            "error": c.error,
            "pairs": [
                {
                    "pair_id": p.pair_id,
                    "order_name": p.order_name,
                    "per_pair_rms": p.per_pair_rms,
                    "left_rms": p.left_rms,
                    "right_rms": p.right_rms,
                    "point_count": p.point_count,
                    "passed": p.passed,
                    "independent_rms": p.independent_rms,
                    "rot_deg": p.rot_deg,
                    "trans_angle_deg": p.trans_angle_deg,
                    "trans_scale_ratio": p.trans_scale_ratio,
                }
                for p in c.pairs
            ],
        }

    return {
        "success": outcome.success,
        "reason": outcome.reason,
        "selected_pair_ids": outcome.selected_pair_ids,
        "threshold": outcome.threshold,
        "passing_candidates": outcome.passing_candidates,
        "accepted": cand(outcome.accepted),
        "all_candidates": [cand(c) for c in outcome.all_candidates],
    }


def _copy_pairs(data):
    return json.loads(json.dumps(data["image_pairs"]))


def _threshold_independent(snapshot):
    """去掉随像素上限变化的 ``passed`` 字段，用于比较复用结果。"""
    def strip(c):
        if c is None:
            return None
        out = {k: v for k, v in c.items() if k not in ("passed", "pairs")}
        out["pairs"] = [
            {k: v for k, v in p.items() if k != "passed"} for p in c["pairs"]
        ]
        return out

    return {
        "selected_pair_ids": snapshot["selected_pair_ids"],
        "accepted": strip(snapshot["accepted"]),
        "all_candidates": [strip(c) for c in snapshot["all_candidates"]],
    }


def test_cache_reuse_across_thresholds_matches_cold_run(sample):
    """同一批输入反复调阈值：命中缓存的结果必须与冷算完全一致。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    cache = CalibrationCache()
    cold = _snapshot(run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960)
    ))
    warm = _snapshot(run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960), cache=cache
    ))
    assert warm == cold  # 首次填充缓存也不得改变结果
    cold_fixed = _threshold_independent(cold)
    for threshold in (0.75, 1.0, 3.0, 10.0, 100.0):
        got = _snapshot(run_calibration(
            left, right, board, data["image_pairs"], ids, threshold,
            (1280, 960), cache=cache,
        ))
        # 与未启用缓存的冷算逐字段一致（含达标判定与原因）
        assert got == _snapshot(run_calibration(
            left, right, board, data["image_pairs"], ids, threshold,
            (1280, 960),
        ))
        # 与阈值无关的数值在命中缓存时必须完全相同
        assert _threshold_independent(got) == cold_fixed


def test_threshold_change_avoids_redundant_opencv_solves(sample, monkeypatch):
    """仅改像素上限：缓存命中后不应再调用 solvePnP/stereoCalibrate。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    cache = CalibrationCache()
    run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                    (1280, 960), cache=cache)

    calls = {"pnp": 0, "stereo": 0, "project": 0}
    real_pnp, real_stereo, real_project = (
        cv2.solvePnP, cv2.stereoCalibrate, cv2.projectPoints
    )
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (
        calls.__setitem__("pnp", calls["pnp"] + 1), real_pnp(*a, **k))[1])
    monkeypatch.setattr(cv2, "stereoCalibrate", lambda *a, **k: (
        calls.__setitem__("stereo", calls["stereo"] + 1), real_stereo(*a, **k))[1])
    monkeypatch.setattr(cv2, "projectPoints", lambda *a, **k: (
        calls.__setitem__("project", calls["project"] + 1),
        real_project(*a, **k))[1])

    for threshold in (0.61, 0.8, 2.0, 50.0):
        run_calibration(left, right, board, data["image_pairs"], ids, threshold,
                        (1280, 960), cache=cache)
    assert calls == {"pnp": 0, "stereo": 0, "project": 0}


def test_left_camera_pose_solved_once_per_pair_within_a_run(sample, monkeypatch):
    """冷算时左姿只按图对解一次，由全部候选排序共享（不再是 排序数 × 图对数）。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    variants = build_order_variants(board.rows, board.cols)
    real_pnp = cv2.solvePnP
    call_args: list[tuple[int, bool]] = []

    def counting_pnp(obj, pts, K, dist, *args, **kwargs):
        # 左相机是传入的第 1 个相机：按内参主点区分左右
        is_left = np.array_equal(np.asarray(K), left.K)
        call_args.append((len(pts), is_left))
        return real_pnp(obj, pts, K, dist, *args, **kwargs)

    monkeypatch.setattr(cv2, "solvePnP", counting_pnp)
    run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                    (1280, 960), cache=CalibrationCache())

    left_calls = [n for n, is_left in call_args if is_left]
    right_calls = [n for n, is_left in call_args if not is_left]
    # 左姿：每对图仅 1 次；右姿：每对图 × 每种排序各 1 次（独立解算）
    assert left_calls == [board.corner_count] * len(ids)
    assert right_calls == [board.corner_count] * (len(ids) * len(variants))


def test_cache_key_changes_on_input_selection_and_config(sample):
    """换批 / 改选图对 / 改相机、棋盘、图对坐标或尺寸都会更换缓存键。"""
    data, left, right, board = sample
    pairs = _copy_pairs(data)
    ids = [p["pair_id"] for p in pairs[:5]]
    base = build_solution_cache_key(
        left, right, board, pairs, ids, (1280, 960), "sess"
    )

    def key(**kw):
        args = dict(left=left, right=right, board=board, pairs=pairs,
                    selected_ids=ids, image_size=(1280, 960), cache_seed="sess")
        args.update(kw)
        return build_solution_cache_key(**args)

    assert key() == base  # 相同输入产生稳定指纹
    # 改选图对
    assert key(selected_ids=ids[:4]) != base
    assert key(selected_ids=ids[:4] + [pairs[5]["pair_id"]]) != base
    # 换批（session seed 变化）
    assert key(cache_seed="other") != base
    # 相机内参、畸变、棋盘参数、图像尺寸
    assert key(left=CameraParams(left.K * 1.01, left.dist)) != base
    assert key(right=CameraParams(right.K, right.dist + 0.01)) != base
    assert key(board=BoardSpec(board.rows, board.cols, board.square_size + 1)) != base
    assert key(image_size=(1280, 961)) != base
    # 任意图对坐标变化（含未选中的 pair_06）
    moved = _copy_pairs(data)
    moved[5]["right_pts"][0][0] += 0.25
    assert key(pairs=moved) != base
    moved = _copy_pairs(data)
    moved[0]["left_pts"][3][1] -= 0.25
    assert key(pairs=moved) != base


def test_input_change_busts_cache_and_new_result_is_used(sample, monkeypatch):
    """缓存填充后修改右图坐标，必须重算而不是回放旧结果。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    cache = CalibrationCache()
    first = run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                            (1280, 960), cache=cache)
    assert first.success

    modified = _copy_pairs(data)
    modified[0]["right_pts"][0][0] += 15.0

    calls = {"stereo": 0}
    real_stereo = cv2.stereoCalibrate
    monkeypatch.setattr(cv2, "stereoCalibrate", lambda *a, **k: (
        calls.__setitem__("stereo", calls["stereo"] + 1), real_stereo(*a, **k))[1])
    second = run_calibration(left, right, board, modified, ids, 0.6,
                             (1280, 960), cache=cache)
    assert calls["stereo"] > 0
    assert second.all_candidates[0].pairs[0].per_pair_rms != pytest.approx(
        first.all_candidates[0].pairs[0].per_pair_rms
    )


def test_selection_change_busts_cache(sample, monkeypatch):
    """改选图对必须重算（左姿与立体标定都依赖所选子集）。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    cache = CalibrationCache()
    run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                    (1280, 960), cache=cache)

    calls = {"stereo": 0, "pnp": 0}
    real_stereo, real_pnp = cv2.stereoCalibrate, cv2.solvePnP
    monkeypatch.setattr(cv2, "stereoCalibrate", lambda *a, **k: (
        calls.__setitem__("stereo", calls["stereo"] + 1), real_stereo(*a, **k))[1])
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (
        calls.__setitem__("pnp", calls["pnp"] + 1), real_pnp(*a, **k))[1])

    run_calibration(left, right, board, data["image_pairs"], ids[:4], 0.6,
                    (1280, 960), cache=cache)
    assert calls["stereo"] > 0 and calls["pnp"] > 0


@pytest.mark.parametrize("threshold", [0.6, 100.0])
def test_failed_and_ambiguous_runs_are_stable_when_cached(sample, threshold):
    """失败/多解情形下，缓存命中前后的判定、原因与上次结果语义必须一致。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"]]
    cache = CalibrationCache()
    cold = _snapshot(run_calibration(
        left, right, board, data["image_pairs"], ids, threshold, (1280, 960)
    ))
    assert cold["success"] is False
    for _ in range(3):
        warm = _snapshot(run_calibration(
            left, right, board, data["image_pairs"], ids, threshold,
            (1280, 960), cache=cache,
        ))
        assert warm == cold


def test_cached_arrays_are_isolated_from_caller_mutation(sample):
    """调用方修改返回的 R/T 不得污染缓存中的复用结果。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:5]]
    cache = CalibrationCache()
    first = run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                            (1280, 960), cache=cache)
    expected_R = first.accepted.R.copy()
    first.accepted.R[:] = 0.0
    first.accepted.T[:] = 0.0

    second = run_calibration(left, right, board, data["image_pairs"], ids, 0.6,
                             (1280, 960), cache=cache)
    np.testing.assert_array_equal(second.accepted.R, expected_R)


def test_cache_lru_eviction_and_clear():
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
    ids = [p["pair_id"] for p in data["image_pairs"][:4]]
    cache = CalibrationCache(max_entries=1)
    # 阈值不进缓存键，连续改阈值只产生 1 个条目；再换选图对才产生第 2 个
    for threshold in (0.6, 0.7, 0.8):
        run_calibration(left, right, board, data["image_pairs"], ids, threshold,
                        (1280, 960), cache=cache)
    assert len(cache) == 1
    run_calibration(left, right, board, data["image_pairs"], ids[:3], 0.6,
                    (1280, 960), cache=cache)
    assert len(cache) == 1  # 容量为 1，新键淘汰旧条目
    assert cache.misses == 2


def test_cache_key_excludes_threshold(sample):
    """像素上限不属于解算输入，键中不应包含（由阈值后置应用保证）。"""
    data, left, right, board = sample
    pairs = _copy_pairs(data)
    ids = [p["pair_id"] for p in pairs[:4]]
    a = run_calibration(left, right, board, pairs, ids, 0.6, (1280, 960),
                        cache=(cache := CalibrationCache()))
    b = run_calibration(left, right, board, pairs, ids, 999.0, (1280, 960),
                        cache=cache)
    assert len(cache) == 1
    assert cache.hits == 1 and cache.misses == 1
    # 阈值只改变达标判定，不改变 R/T 与逐对 RMSE 数值
    assert a.accepted is not None or b.accepted is not None
    by_name_a = {c.order_name: c for c in a.all_candidates}
    by_name_b = {c.order_name: c for c in b.all_candidates}
    assert set(by_name_a) == set(by_name_b)
    for name, ca in by_name_a.items():
        cb = by_name_b[name]
        assert ca.overall_rms == cb.overall_rms
        assert [(p.per_pair_rms, p.left_rms, p.right_rms) for p in ca.pairs] == \
            [(p.per_pair_rms, p.left_rms, p.right_rms) for p in cb.pairs]


def test_stereo_failure_is_not_passing_and_stable_cached(sample, monkeypatch):
    """立体标定失败时该排序不得算作达标；缓存命中前后判定一致。"""
    data, left, right, board = sample
    ids = [p["pair_id"] for p in data["image_pairs"][:4]]

    def failing_stereo(*a, **k):
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(cv2, "stereoCalibrate", failing_stereo)
    cache = CalibrationCache()
    cold = _snapshot(run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960),
        cache=cache,
    ))
    assert cold["success"] is False
    assert cold["passing_candidates"] == []
    assert all(c["passed"] is False for c in cold["all_candidates"])
    assert all(c["error"] and "立体标定失败" in c["error"]
               for c in cold["all_candidates"])

    # 恢复 OpenCV 后仍走同一缓存键：失败事实被复用，不会被“修好”而误判达标
    monkeypatch.setattr(cv2, "stereoCalibrate", lambda *a, **k: pytest.fail(
        "缓存命中时不应再调用 stereoCalibrate"))
    warm = _snapshot(run_calibration(
        left, right, board, data["image_pairs"], ids, 0.6, (1280, 960),
        cache=cache,
    ))
    assert warm == cold
    assert cache.misses == 1 and cache.hits == 1
