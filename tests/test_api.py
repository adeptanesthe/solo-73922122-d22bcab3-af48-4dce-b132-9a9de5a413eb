"""FastAPI 端到端测试：上传、标定成功/失败、上次成功结果留存与批次切换。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app.calibration import CalibrationCache
from app.main import STORE, app

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_calibration.json"


@pytest.fixture(autouse=True)
def _fresh_store():
    """每个用例从干净的进程内状态开始，避免缓存计数跨用例累积。"""
    STORE.session_id = None
    STORE.raw = None
    STORE.last_success = None
    STORE.cache = CalibrationCache(max_entries=16)
    yield


def _upload_raw(raw: dict) -> dict:
    resp = client.post(
        "/api/upload",
        files={"file": ("sample_calibration.json", json.dumps(raw).encode(),
                        "application/json")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _upload() -> dict:
    with SAMPLE.open("rb") as fh:
        resp = client.post(
            "/api/upload",
            files={"file": ("sample_calibration.json", fh, "application/json")},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _calibrate(raw: dict, ids: list[str], threshold: float):
    body = {
        "data": {
            **raw,
            "selected_pair_ids": ids,
            "pixel_threshold": threshold,
        }
    }
    return client.post("/api/calibrate", json=body)


def test_health_and_index():
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200 and "双目" in page.text
    js = client.get("/static/app.js")
    assert js.status_code == 200


def test_upload_rejects_bad_json():
    resp = client.post(
        "/api/upload",
        files={"file": ("bad.json", b"{not json", "application/json")},
    )
    assert resp.status_code == 422


def test_full_flow_success_then_failure_keeps_last_result():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    meta = _upload()
    assert len(meta["image_pairs"]) == 6
    assert STORE.last_success is None  # 新批次必须清空旧结果

    good_ids = [p["pair_id"] for p in raw["image_pairs"][:5]]
    ok = _calibrate(raw, good_ids, 0.6)
    assert ok.status_code == 200
    ok_body = ok.json()
    assert ok_body["success"] is True
    assert ok_body["accepted"]["order_name"] == "flip_cols"
    assert ok_body["accepted"]["selected_pair_ids"] == good_ids
    assert len(ok_body["accepted"]["pairs"]) == 5

    # 上次成功结果已留存，且记录使用了哪些图对
    kept = client.get("/api/last-success").json()["last_success"]
    assert kept["result"]["selected_pair_ids"] == good_ids

    # 失败标定：含大噪声 pair_06，严格阈值
    all_ids = [p["pair_id"] for p in raw["image_pairs"]]
    fail = _calibrate(raw, all_ids, 0.6)
    assert fail.status_code == 200
    fail_body = fail.json()
    assert fail_body["success"] is False
    assert fail_body["accepted"] is None
    assert "没有任何一种角点排序" in fail_body["reason"]
    # 关键语义：失败后上次成功结果仍在，且标明所用图对
    assert fail_body["last_success"]["result"]["selected_pair_ids"] == good_ids
    assert client.get("/api/last-success").json()["last_success"] is not None


def test_ambiguous_calibration_does_not_overwrite():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    good_ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    _calibrate(raw, good_ids, 0.6)  # 先建立成功结果

    loose = _calibrate(raw, good_ids, 100.0)
    body = loose.json()
    assert body["success"] is False
    assert len(body["passing_candidates"]) >= 2
    # 多解时同样保留上次成功结果
    assert body["last_success"] is not None
    assert body["last_success"]["result"]["order_name"] == "flip_cols"


def test_reupload_resets_last_success():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    good_ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    assert _calibrate(raw, good_ids, 0.6).json()["success"] is True
    assert client.get("/api/last-success").json()["last_success"] is not None

    # 重新上传一批 -> 上一批结果清空
    _upload()
    assert client.get("/api/last-success").json()["last_success"] is None


def test_less_than_three_pairs_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    resp = _calibrate(raw, ["pair_01", "pair_02"], 0.6)
    assert resp.status_code == 422


def test_invalid_threshold_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    resp = _calibrate(
        raw, [p["pair_id"] for p in raw["image_pairs"][:4]], 0
    )
    assert resp.status_code == 422


def test_upload_with_only_two_pairs_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    raw["image_pairs"] = raw["image_pairs"][:2]
    resp = client.post(
        "/api/upload",
        files={"file": ("short.json", json.dumps(raw).encode(), "application/json")},
    )
    assert resp.status_code == 422


def test_non_square_camera_matrix_rejected():
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    raw["left_camera"]["camera_matrix"] = [[1, 0, 0], [0, 1, 0]]  # 2x3
    resp = client.post(
        "/api/upload",
        files={"file": ("badk.json", json.dumps(raw).encode(), "application/json")},
    )
    # 上传阶段模型校验通过形状字段，3x3 形状在标定入口拦截
    assert resp.status_code == 200
    cal = _calibrate(
        raw, [p["pair_id"] for p in raw["image_pairs"][:4]], 0.6
    )
    assert cal.status_code == 422
    assert "3x3" in cal.json()["detail"]


# ---- 复用与失效（端到端） ---------------------------------------------------


def _count_opencv(monkeypatch):
    calls = {"pnp": 0, "stereo": 0, "project": 0}
    real = (cv2.solvePnP, cv2.stereoCalibrate, cv2.projectPoints)

    def wrap(name, fn):
        def inner(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return inner

    monkeypatch.setattr(cv2, "solvePnP", wrap("pnp", real[0]))
    monkeypatch.setattr(cv2, "stereoCalibrate", wrap("stereo", real[1]))
    monkeypatch.setattr(cv2, "projectPoints", wrap("project", real[2]))
    return calls


def _calibrate_ok(raw: dict, ids: list[str], threshold: float) -> dict:
    resp = _calibrate(raw, ids, threshold)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _strip_passed(candidates: list[dict]) -> list[dict]:
    """去掉随阈值变化的 passed 字段，只比较与阈值无关的解算数值。"""
    return [
        {
            **{k: v for k, v in c.items() if k != "pairs"},
            "pairs": [
                {k: v for k, v in p.items() if k != "passed"} for p in c["pairs"]
            ],
        }
        for c in candidates
    ]


def test_threshold_changes_reuse_work_within_session(monkeypatch):
    """同一批次内反复拖阈值：不再重复解算，结果与首次冷算一致。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    ids = [p["pair_id"] for p in raw["image_pairs"][:5]]

    calls = _count_opencv(monkeypatch)
    first = _calibrate_ok(raw, ids, 0.6)
    assert calls["stereo"] > 0
    assert first["success"] is True
    baseline = {
        "R": first["accepted"]["R"],
        "T": first["accepted"]["T"],
        "pairs": first["accepted"]["pairs"],
        "candidates": _strip_passed(first["candidates"]),
    }
    first_calls = dict(calls)
    assert STORE.cache.misses == 1

    for threshold in (0.65, 0.9, 2.0, 50.0):
        body = _calibrate_ok(raw, ids, threshold)
        assert body["accepted"] is not None
        assert body["accepted"]["R"] == baseline["R"]
        assert body["accepted"]["T"] == baseline["T"]
        # 逐对 RMSE 复用，达标判定按新阈值重新套用
        assert [(p["pair_id"], p["per_pair_rms"], p["passed"])
                for p in body["accepted"]["pairs"]] == [
            (p["pair_id"], p["per_pair_rms"], p["per_pair_rms"] <= threshold)
            for p in baseline["pairs"]
        ]
        assert _strip_passed(body["candidates"]) == baseline["candidates"]

    # 阈值命中缓存：除首次冷算外没有新的 OpenCV 调用
    assert calls == first_calls
    assert STORE.cache.hits == 4 and STORE.cache.misses == 1


def test_threshold_change_with_failure_keeps_last_success(monkeypatch):
    """阈值调到无解时仍复用解算结果，但失败语义与上次成功结果不变。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    good_ids = [p["pair_id"] for p in raw["image_pairs"][:5]]
    assert _calibrate_ok(raw, good_ids, 0.6)["success"] is True

    calls = _count_opencv(monkeypatch)
    all_ids = [p["pair_id"] for p in raw["image_pairs"]]
    fail = _calibrate_ok(raw, all_ids, 0.6)  # 改选图对 -> 必须冷算
    assert calls["stereo"] > 0
    assert fail["success"] is False
    assert fail["last_success"]["result"]["selected_pair_ids"] == good_ids
    cold_calls = dict(calls)

    # 同一失败输入下改阈值：复用本次失败解算，不产生新的 OpenCV 调用
    again = _calibrate_ok(raw, all_ids, 0.7)
    assert calls == cold_calls
    assert again["success"] is False
    assert again["passing_candidates"] == fail["passing_candidates"]
    assert again["candidates"] == fail["candidates"]
    assert again["last_success"]["result"]["selected_pair_ids"] == good_ids


def test_reselection_invalidates_cache():
    """换选图对必须重算，且返回所选子集对应的结果。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    ids5 = [p["pair_id"] for p in raw["image_pairs"][:5]]
    ids4 = ids5[:4]
    five = _calibrate_ok(raw, ids5, 0.6)
    four = _calibrate_ok(raw, ids4, 0.6)
    assert five["accepted"]["selected_pair_ids"] == ids5
    assert four["accepted"]["selected_pair_ids"] == ids4
    assert len(four["accepted"]["pairs"]) == 4
    assert STORE.cache.misses == 2


def test_reupload_clears_cache():
    """换批后缓存与上次成功结果都清空，旧批结果不会被回放。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    ids = [p["pair_id"] for p in raw["image_pairs"][:5]]
    _calibrate_ok(raw, ids, 0.6)
    assert len(STORE.cache) == 1

    _upload()
    assert len(STORE.cache) == 0
    assert STORE.last_success is None

    # 即便内容相同，新批次也重新冷算一次（批次隔离）
    body = _calibrate_ok(raw, ids, 0.6)
    assert body["success"] is True
    assert STORE.cache.misses == 2


def test_changed_input_json_invalidates_cached_result():
    """输入 JSON 中右图坐标改动后，接口必须返回基于新数据的误差。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    # 只选前 4 对（不含大噪声 pair_06），阈值 0.6 时稳定成功
    ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    before = _calibrate_ok(raw, ids, 0.6)
    assert before["success"] is True

    changed = json.loads(json.dumps(raw))
    # 只挪动一个角点：仍能唯一达标，但误差数值必须体现新输入
    changed["image_pairs"][0]["right_pts"][0][0] += 3.0
    after = _calibrate_ok(changed, ids, 0.6)

    def rms(body):
        return {c["order_name"]: c["max_pair_rms"] for c in body["candidates"]}

    assert rms(after) != rms(before)
    assert after["accepted"] is not None
    assert after["accepted"]["pairs"][0]["right_rms"] != \
        before["accepted"]["pairs"][0]["right_rms"]
    assert STORE.cache.misses == 2


def test_cache_stats_do_not_leak_threshold_into_key():
    """阈值不是解算输入：多次改阈值只占一个缓存条目。"""
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    _upload()
    ids = [p["pair_id"] for p in raw["image_pairs"][:4]]
    for threshold in (0.6, 0.7, 0.8, 0.9):
        _calibrate_ok(raw, ids, threshold)
    assert len(STORE.cache) == 1
    assert STORE.cache.misses == 1 and STORE.cache.hits == 3
