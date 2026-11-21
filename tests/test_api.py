"""FastAPI 端到端测试：上传、标定成功/失败、上次成功结果留存与批次切换。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import STORE, app

client = TestClient(app)
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_calibration.json"


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
