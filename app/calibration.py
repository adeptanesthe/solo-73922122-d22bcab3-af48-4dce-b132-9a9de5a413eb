"""双目外参标定核心逻辑（与 Web 框架解耦，便于单元测试）。

业务规则
========
1. 输入为已去畸变的棋盘格角点像素坐标（左右各一组，顺序可能整体不一致）。
2. 左图角点顺序为基准；右图的点列可能是棋盘格 8 种对称排序中的任意一种
   （旋转/翻转，维修员描述的“整体倒序”是其中之一）。
3. 对每种候选排序，用所选全部图对做一次 ``stereoCalibrate``（固定内参），
   得到“同一组外参” R,T 及每对图在该外参下的重投影误差。
4. 仅当 *恰好一种* 排序满足：所有被选图对的逐对 RMSE 都不超过像素误差上限，
   才接受该排序并产出新结果。零种达标 -> 报错并保留上次成功结果；
   多种达标 -> 数据不足以区分排序，同样不更新。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ---- 棋盘格对称排序（D4 群，共 8 种） -------------------------------------

# 每个元素为 (名称, 行变换, 列变换)；变换作用于“排序后的目标网格坐标”，
# 目标 (r', c') 处的点取自源 (r, c) 处。
_SYMMETRIES = [
    ("identity",            lambda r, c, R, C: (r, c)),
    ("flip_rows",           lambda r, c, R, C: (R - 1 - r, c)),
    ("flip_cols",           lambda r, c, R, C: (r, C - 1 - c)),
    ("transpose",           lambda r, c, R, C: (c, r)),
    ("anti_transpose",      lambda r, c, R, C: (C - 1 - c, R - 1 - r)),
    ("rotate_180",          lambda r, c, R, C: (R - 1 - r, C - 1 - c)),
    ("rotate_90",           lambda r, c, R, C: (C - 1 - c, r)),
    ("rotate_270",          lambda r, c, R, C: (c, R - 1 - r)),
]


def build_order_variants(rows: int, cols: int) -> list[dict]:
    """生成棋盘格全部对称排序的重排索引。

    返回 ``[{"name": str, "index": ndarray(N,), "description": str}]``，
    ``index[k]`` 表示：排序后第 k 个点 = 原始点列中的第 index[k] 个点。
    方形棋盘存在重复映射，自动去重。
    """
    n = rows * cols
    seen: set[bytes] = set()
    variants: list[dict] = []
    descriptions = {
        "identity": "与左图同序",
        "flip_rows": "行倒序（上下翻转）",
        "flip_cols": "列倒序（左右翻转，即整体倒序）",
        "transpose": "转置",
        "anti_transpose": "反对角转置",
        "rotate_180": "旋转 180°（完全逆序）",
        "rotate_90": "旋转 90°",
        "rotate_270": "旋转 270°",
    }
    for name, fn in _SYMMETRIES:
        idx = np.empty(n, dtype=np.int64)
        valid = True
        for rp in range(rows):
            for cp in range(cols):
                r, c = fn(rp, cp, rows, cols)
                # 转置类对称仅在方形棋盘上是合法网格置换，非方形时越界 -> 丢弃
                if not (0 <= r < rows and 0 <= c < cols):
                    valid = False
                    break
                idx[rp * cols + cp] = r * cols + c
            if not valid:
                break
        if not valid:
            continue
        key = idx.tobytes()
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            {"name": name, "index": idx, "description": descriptions[name]}
        )
    return variants


# ---- 数据结构 ---------------------------------------------------------------


@dataclass
class CameraParams:
    K: np.ndarray  # 3x3
    dist: np.ndarray  # (5,) 或 (14,) 等，去畸变数据通常为 0


@dataclass
class BoardSpec:
    rows: int
    cols: int
    square_size: float

    @property
    def corner_count(self) -> int:
        return self.rows * self.cols

    def object_points(self) -> np.ndarray:
        """棋盘格局部坐标系下的三维角点，z=0，单位与格距一致。"""
        pts = np.zeros((self.corner_count, 3), np.float32)
        pts[:, :2] = np.array(
            [
                (c * self.square_size, r * self.square_size)
                for r in range(self.rows)
                for c in range(self.cols)
            ],
            np.float32,
        )
        return pts


@dataclass
class PairResult:
    pair_id: str
    order_name: str
    per_pair_rms: float          # 共用外参下该对的左右合并 RMSE（像素）
    left_rms: float
    right_rms: float
    point_count: int
    passed: bool
    independent_rms: Optional[float] = None  # 该对单独解算的相对位姿残差
    rot_deg: Optional[float] = None          # 独立相对旋转与共用外参的偏差（度）
    trans_angle_deg: Optional[float] = None  # 平移方向偏差（度）
    trans_scale_ratio: Optional[float] = None


@dataclass
class CandidateResult:
    order_name: str
    order_description: str
    overall_rms: float
    R: Optional[np.ndarray]
    T: Optional[np.ndarray]
    pairs: list[PairResult]
    passed: bool
    error: Optional[str] = None


@dataclass
class CalibrationOutcome:
    success: bool
    reason: Optional[str]
    selected_pair_ids: list[str]
    threshold: float
    accepted: Optional[CandidateResult] = None
    all_candidates: list[CandidateResult] = field(default_factory=list)
    passing_candidates: list[str] = field(default_factory=list)


# ---- 几何计算 ---------------------------------------------------------------


def _rms(projected: np.ndarray, observed: np.ndarray) -> float:
    diff = projected.reshape(-1, 2) - observed.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _project(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
             cam: CameraParams) -> np.ndarray:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
    return proj.reshape(-1, 2)


def solve_pair_relative_pose(obj: np.ndarray, left_pts: np.ndarray,
                             right_pts: np.ndarray, left: CameraParams,
                             right: CameraParams) -> tuple[np.ndarray, np.ndarray, float]:
    """单对图独立求左->右相对位姿（PnP + 对极几何），并返回重投影 RMSE。"""
    ok_l, rvec_l, tvec_l = cv2.solvePnP(obj, left_pts, left.K, left.dist,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    ok_r, rvec_r, tvec_r = cv2.solvePnP(obj, right_pts, right.K, right.dist,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not (ok_l and ok_r):
        raise RuntimeError("solvePnP 失败")
    R_l, _ = cv2.Rodrigues(rvec_l)
    R_r, _ = cv2.Rodrigues(rvec_r)
    R_rel = R_r @ R_l.T
    T_rel = tvec_r - R_rel @ tvec_l
    # 独立解算的重投影残差（两侧 PnP 残差合并）
    rms = math.sqrt(
        (_rms(_project(obj, rvec_l, tvec_l, left), left_pts) ** 2
         + _rms(_project(obj, rvec_r, tvec_r, right), right_pts) ** 2)
        / 2.0
    )
    return R_rel, T_rel, rms


def _stereo_calibrate(obj: np.ndarray, left_pts: list[np.ndarray],
                      right_pts: list[np.ndarray], left: CameraParams,
                      right: CameraParams, image_size: tuple[int, int]
                      ) -> tuple[float, np.ndarray, np.ndarray]:
    """固定内参的立体标定，返回整体 RMS、R、T。兼容 OpenCV 4.x/5.x。"""
    ret = cv2.stereoCalibrate(
        [obj] * len(left_pts),
        [p.astype(np.float32) for p in left_pts],
        [p.astype(np.float32) for p in right_pts],
        left.K.astype(np.float64),
        left.dist.astype(np.float64),
        right.K.astype(np.float64),
        right.dist.astype(np.float64),
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-9),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    # OpenCV 4.x: 10 元组（末尾 perViewErrors）；5.x: 9 元组
    rms = float(ret[0])
    R = np.asarray(ret[5], dtype=np.float64)
    T = np.asarray(ret[6], dtype=np.float64)
    return rms, R, T


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_t = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def evaluate_variant(order_name: str, order_desc: str, index: np.ndarray,
                     obj: np.ndarray, pairs: list[dict],
                     left: CameraParams, right: CameraParams,
                     image_size: tuple[int, int],
                     threshold: float) -> CandidateResult:
    """评估一种右图排序在所选图对上的表现。"""
    left_pts = [np.asarray(p["left_pts"], np.float32).reshape(-1, 2) for p in pairs]
    right_pts = [
        np.asarray(p["right_pts"], np.float32).reshape(-1, 2)[index] for p in pairs
    ]

    pair_results: list[PairResult] = []
    try:
        overall_rms, R, T = _stereo_calibrate(
            obj, left_pts, right_pts, left, right, image_size
        )
    except cv2.error as exc:  # 几何上无法收敛（如排序错误）
        return CandidateResult(
            order_name=order_name, order_description=order_desc,
            overall_rms=float("inf"), R=None, T=None, pairs=[],
            passed=False, error=f"立体标定失败：{exc}",
        )

    # 用共用外参逐对恢复右相机位姿并计算每对 RMSE
    for p, lp, rp in zip(pairs, left_pts, right_pts):
        ok_l, rvec_l, tvec_l = cv2.solvePnP(obj, lp, left.K, left.dist,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
        R_l, _ = cv2.Rodrigues(rvec_l)
        R_r = R @ R_l
        t_r = R @ tvec_l + T
        rvec_r, _ = cv2.Rodrigues(R_r)

        proj_l = _project(obj, rvec_l, tvec_l, left)
        proj_r = _project(obj, rvec_r, t_r, right)
        l_rms = _rms(proj_l, lp)
        r_rms = _rms(proj_r, rp)
        pair_rms = math.sqrt((l_rms ** 2 + r_rms ** 2) / 2.0)
        passed = pair_rms <= threshold

        # 单对独立解算，用于检查“共用同一组外参”的物理一致性
        rot_deg = trans_ang = trans_ratio = None
        ind_rms: Optional[float] = None
        try:
            Ri, Ti, ind_rms = solve_pair_relative_pose(
                obj, lp, rp, left, right
            )
            rot_deg = _rotation_angle_deg(Ri.T @ R)
            t_unit = T.ravel() / (np.linalg.norm(T) + 1e-12)
            ti_unit = Ti.ravel() / (np.linalg.norm(Ti) + 1e-12)
            trans_ang = float(np.degrees(
                np.arccos(np.clip(float(np.dot(t_unit, ti_unit)), -1.0, 1.0))
            ))
            trans_ratio = float(np.linalg.norm(Ti) / (np.linalg.norm(T) + 1e-12))
        except (cv2.error, RuntimeError):
            pass

        pair_results.append(PairResult(
            pair_id=p["pair_id"], order_name=order_name,
            per_pair_rms=pair_rms, left_rms=l_rms, right_rms=r_rms,
            point_count=len(lp), passed=passed,
            independent_rms=ind_rms, rot_deg=rot_deg,
            trans_angle_deg=trans_ang, trans_scale_ratio=trans_ratio,
        ))

    return CandidateResult(
        order_name=order_name, order_description=order_desc,
        overall_rms=overall_rms, R=R, T=T, pairs=pair_results,
        passed=all(pr.passed for pr in pair_results),
    )


def run_calibration(left: CameraParams, right: CameraParams, board: BoardSpec,
                    pairs: list[dict], selected_ids: list[str],
                    threshold: float, image_size: tuple[int, int]
                    ) -> CalibrationOutcome:
    """主入口：枚举排序 → 联合标定 → 唯一性判定。"""
    if threshold <= 0:
        raise ValueError("像素误差上限必须为正数")
    selected = [p for p in pairs if p["pair_id"] in set(selected_ids)]
    if len(selected) != len(set(selected_ids)) or len(selected) < 3:
        raise ValueError("至少需要选择 3 对图，且所选图对必须都存在")

    n_expected = board.corner_count
    for p in selected:
        for side in ("left_pts", "right_pts"):
            pts = np.asarray(p[side], np.float64).reshape(-1, 2)
            if len(pts) != n_expected:
                raise ValueError(
                    f"图对 {p['pair_id']} 的 {side} 角点数 {len(pts)} "
                    f"与棋盘格 {board.rows}x{board.cols}={n_expected} 不符"
                )

    obj = board.object_points()
    variants = build_order_variants(board.rows, board.cols)
    results = [
        evaluate_variant(
            v["name"], v["description"], v["index"], obj, selected,
            left, right, image_size, threshold,
        )
        for v in variants
    ]
    results.sort(key=lambda c: c.overall_rms)
    passing = [c for c in results if c.passed]

    if len(passing) == 1:
        return CalibrationOutcome(
            success=True, reason=None,
            selected_pair_ids=[p["pair_id"] for p in selected],
            threshold=threshold, accepted=passing[0],
            all_candidates=results,
            passing_candidates=[c.order_name for c in passing],
        )

    if not passing:
        reason = (
            "没有任何一种角点排序能让全部所选图对在同一组外参下达到"
            f" {threshold:g} 像素的误差上限（已尝试 {len(results)} 种排序）。"
            "可能原因：右图角点并非整体同序错位、角点检测偏差过大、"
            "图对不同步或内参/棋盘参数填写错误。"
        )
    else:
        names = "、".join(c.order_name for c in passing)
        reason = (
            f"存在 {len(passing)} 种排序（{names}）在当前 {threshold:g} 像素"
            "上限下同时达标，数据不足以唯一确定右图角点顺序。"
            "请收紧误差上限、增加图对数量或换用姿态差异更大的图对后重试。"
        )
    return CalibrationOutcome(
        success=False, reason=reason,
        selected_pair_ids=[p["pair_id"] for p in selected],
        threshold=threshold, all_candidates=results,
        passing_candidates=[c.order_name for c in passing],
    )
