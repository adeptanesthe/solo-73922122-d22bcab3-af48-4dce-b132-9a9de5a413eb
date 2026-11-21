"""FastAPI 网页后端：上传标定 JSON、执行外参标定、留存上次成功结果。"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .calibration import (
    BoardSpec,
    CalibrationCache,
    CalibrationOutcome,
    CameraParams,
    CandidateResult,
    PairResult,
    run_calibration,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="双目相机外参标定", version="1.0.0")

# ---- 请求/响应模型 ----------------------------------------------------------


class IntrinsicInput(BaseModel):
    camera_matrix: list[list[float]] = Field(..., description="3x3 内参矩阵 K")
    dist_coeffs: list[float] = Field(..., description="畸变系数（去畸变数据可全 0）")


class BoardInput(BaseModel):
    rows: int = Field(..., gt=1, description="棋盘格内角点行数")
    cols: int = Field(..., gt=1, description="棋盘格内角点列数")
    square_size: float = Field(..., gt=0, description="格距（毫米）")


class ImagePairInput(BaseModel):
    pair_id: str
    left_pts: list[list[float]] = Field(..., description="左图角点 [[x,y],...]")
    right_pts: list[list[float]] = Field(..., description="右图角点 [[x,y],...]")
    image_size: Optional[list[int]] = Field(
        None, description="[width, height]，缺省时由角点范围推断"
    )


class CalibrationPayload(BaseModel):
    left_camera: IntrinsicInput
    right_camera: IntrinsicInput
    board: BoardInput
    image_pairs: list[ImagePairInput] = Field(..., min_length=3, max_length=6)
    selected_pair_ids: list[str] = Field(..., min_length=3, max_length=6)
    pixel_threshold: float = Field(..., gt=0)


class CalibrateRequest(BaseModel):
    data: CalibrationPayload


# ---- 进程内状态 -------------------------------------------------------------


class Store:
    """留存本批上传数据与上次成功标定结果。"""

    def __init__(self) -> None:
        self.session_id: Optional[str] = None
        self.raw: Optional[dict] = None
        self.last_success: Optional[dict] = None
        # 与阈值无关的标定解算缓存，按输入内容寻址；换批时清空
        self.cache = CalibrationCache(max_entries=16)

    def reset_with(self, raw: dict) -> None:
        # 新一批数据：清掉上一批的成功结果与解算缓存
        self.session_id = uuid.uuid4().hex[:12]
        self.raw = raw
        self.last_success = None
        self.cache.clear()


STORE = Store()


# ---- 序列化 -----------------------------------------------------------------


def _pair_to_dict(pr: PairResult) -> dict[str, Any]:
    return {
        "pair_id": pr.pair_id,
        "order_name": pr.order_name,
        "per_pair_rms": round(pr.per_pair_rms, 6),
        "left_rms": round(pr.left_rms, 6),
        "right_rms": round(pr.right_rms, 6),
        "point_count": pr.point_count,
        "passed": pr.passed,
        "independent_rms": None if pr.independent_rms is None else round(pr.independent_rms, 6),
        "rotation_delta_deg": None if pr.rot_deg is None else round(pr.rot_deg, 4),
        "translation_angle_deg": None if pr.trans_angle_deg is None else round(pr.trans_angle_deg, 4),
        "translation_scale_ratio": None if pr.trans_scale_ratio is None else round(pr.trans_scale_ratio, 4),
    }


def _candidate_brief(c: CandidateResult) -> dict[str, Any]:
    return {
        "order_name": c.order_name,
        "order_description": c.order_description,
        "overall_rms": None if math.isinf(c.overall_rms) else round(c.overall_rms, 6),
        "passed": c.passed,
        "error": c.error,
        "max_pair_rms": round(max((p.per_pair_rms for p in c.pairs), default=float("inf")), 6),
        "pairs": [
            {"pair_id": p.pair_id, "per_pair_rms": round(p.per_pair_rms, 6), "passed": p.passed}
            for p in c.pairs
        ],
    }


def _outcome_to_dict(outcome: CalibrationOutcome) -> dict[str, Any]:
    accepted = None
    if outcome.accepted is not None:
        c = outcome.accepted
        accepted = {
            "order_name": c.order_name,
            "order_description": c.order_description,
            "overall_rms": round(c.overall_rms, 6),
            "R": np.round(c.R, 8).tolist(),
            "T": np.round(c.T, 8).tolist(),
            "T_flat": np.round(c.T.ravel(), 8).tolist(),
            "pairs": [_pair_to_dict(p) for p in c.pairs],
            "selected_pair_ids": outcome.selected_pair_ids,
            "threshold": outcome.threshold,
        }
    return {
        "success": outcome.success,
        "reason": outcome.reason,
        "selected_pair_ids": outcome.selected_pair_ids,
        "threshold": outcome.threshold,
        "passing_candidates": outcome.passing_candidates,
        "accepted": accepted,
        "candidates": [_candidate_brief(c) for c in outcome.all_candidates],
    }


def _build_inputs(payload: CalibrationPayload) -> tuple[
    CameraParams, CameraParams, BoardSpec, list[dict], tuple[int, int]
]:
    Kl = np.asarray(payload.left_camera.camera_matrix, dtype=np.float64)
    Kr = np.asarray(payload.right_camera.camera_matrix, dtype=np.float64)
    if Kl.shape != (3, 3) or Kr.shape != (3, 3):
        raise HTTPException(422, "camera_matrix 必须是 3x3 矩阵")
    dl = np.asarray(payload.left_camera.dist_coeffs, dtype=np.float64).ravel()
    dr = np.asarray(payload.right_camera.dist_coeffs, dtype=np.float64).ravel()

    pairs: list[dict] = []
    sizes: list[tuple[int, int]] = []
    for p in payload.image_pairs:
        lp = np.asarray(p.left_pts, dtype=np.float64)
        rp = np.asarray(p.right_pts, dtype=np.float64)
        if lp.ndim != 2 or lp.shape[1] != 2 or rp.ndim != 2 or rp.shape[1] != 2:
            raise HTTPException(422, f"图对 {p.pair_id} 的角点必须是 [[x,y],...] 形式")
        if len(lp) != len(rp):
            raise HTTPException(422, f"图对 {p.pair_id} 左右角点数量不一致")
        pairs.append({
            "pair_id": p.pair_id,
            "left_pts": lp.tolist(),
            "right_pts": rp.tolist(),
        })
        if p.image_size is not None:
            w, h = p.image_size
            if w <= 0 or h <= 0:
                raise HTTPException(422, f"图对 {p.pair_id} 的 image_size 非法")
            sizes.append((int(w), int(h)))

    if sizes:
        image_size = sizes[0]
    else:
        all_pts = np.concatenate(
            [np.asarray(p["left_pts"]).reshape(-1, 2) for p in pairs]
            + [np.asarray(p["right_pts"]).reshape(-1, 2) for p in pairs],
            axis=0,
        )
        image_size = (
            int(math.ceil(all_pts[:, 0].max())) + 1,
            int(math.ceil(all_pts[:, 1].max())) + 1,
        )

    return (
        CameraParams(Kl, dl),
        CameraParams(Kr, dr),
        BoardSpec(payload.board.rows, payload.board.cols, payload.board.square_size),
        pairs,
        image_size,
    )


# ---- 路由 -------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """接收标定 JSON，解析并留存为本批数据，返回前端所需的回显内容。"""
    try:
        raw = json.loads((await file.read()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"无法解析 JSON 文件：{exc}") from exc

    try:
        payload = CalibrationPayload(**{
            **raw,
            # 上传阶段不强制预选，给默认值以通过模型校验，标定前可在页面调整
            "selected_pair_ids": raw.get("selected_pair_ids")
            or [p["pair_id"] for p in raw.get("image_pairs", [])],
            "pixel_threshold": raw.get("pixel_threshold", 1.0),
        })
    except Exception as exc:
        raise HTTPException(422, f"JSON 内容校验失败：{exc}") from exc

    STORE.reset_with(raw)
    return JSONResponse({
        "session_id": STORE.session_id,
        "board": payload.board.model_dump(),
        "left_camera": payload.left_camera.model_dump(),
        "right_camera": payload.right_camera.model_dump(),
        "image_pairs": [
            {
                "pair_id": p.pair_id,
                "point_count": len(p.left_pts),
                "image_size": p.image_size,
                "left_bounds": _bounds(p.left_pts),
                "right_bounds": _bounds(p.right_pts),
            }
            for p in payload.image_pairs
        ],
        "selected_pair_ids": payload.selected_pair_ids,
        "pixel_threshold": payload.pixel_threshold,
        "last_success": None,
    })


def _bounds(pts: list[list[float]]) -> dict[str, float]:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return {
        "x_min": round(float(arr[:, 0].min()), 2),
        "x_max": round(float(arr[:, 0].max()), 2),
        "y_min": round(float(arr[:, 1].min()), 2),
        "y_max": round(float(arr[:, 1].max()), 2),
    }


@app.post("/api/calibrate")
def calibrate(req: CalibrateRequest) -> JSONResponse:
    payload = req.data
    left, right, board, pairs, image_size = _build_inputs(payload)
    try:
        outcome = run_calibration(
            left, right, board, pairs,
            payload.selected_pair_ids,
            payload.pixel_threshold,
            image_size,
            cache=STORE.cache,
            cache_seed=STORE.session_id,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    body = _outcome_to_dict(outcome)
    if outcome.success and outcome.accepted is not None:
        STORE.last_success = {
            "session_id": STORE.session_id,
            "result": body["accepted"],
        }
    body["last_success"] = STORE.last_success
    return JSONResponse(body)


@app.get("/api/last-success")
def last_success() -> JSONResponse:
    return JSONResponse({"last_success": STORE.last_success})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
