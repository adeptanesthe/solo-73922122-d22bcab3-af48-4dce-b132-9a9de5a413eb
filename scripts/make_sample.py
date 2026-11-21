#!/usr/bin/env python3
"""生成可直接试标的双目角点样例 JSON。

数据链路（保证物理一致性）：
  已知左右内参、左->右 R,T 与每个位姿的棋盘格外参，用 projectPoints
  正投影得到“已去畸变”的角点像素坐标，并注入小量检测噪声。

样例设计：
  - 6 对同步图，board = 7x5 内角点，格距 25 mm；
  - 右图角点列整体为“列倒序”（flip_cols，维修现场最常见的倒序情形）；
  - pair_05 注入很大的噪声，用于演示“没有排序达标 → 保留上次结果”；
  - 前 5 对任选 >=3 对且阈值取 0.6 px 时可唯一标定成功。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROWS, COLS = 7, 5
SQUARE_SIZE = 25.0
IMG_W, IMG_H = 1280, 960


def object_points() -> np.ndarray:
    pts = np.zeros((ROWS * COLS, 3), np.float32)
    pts[:, :2] = np.array(
        [(c * SQUARE_SIZE, r * SQUARE_SIZE)
         for r in range(ROWS) for c in range(COLS)],
        np.float32,
    )
    return pts


def flip_cols_index() -> np.ndarray:
    """右图相对左图的列倒序重排索引（排序后第 k 点取自原 index[k]）。"""
    idx = np.empty(ROWS * COLS, dtype=np.int64)
    for r in range(ROWS):
        for c in range(COLS):
            idx[r * COLS + c] = r * COLS + (COLS - 1 - c)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="输出 JSON 路径")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    obj = object_points()

    fx_l, fy_l, cx_l, cy_l = 900.0, 905.0, 645.0, 482.0
    fx_r, fy_r, cx_r, cy_r = 895.0, 900.0, 638.0, 480.0
    K_l = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]], np.float64)
    K_r = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]], np.float64)
    dist_zero = [0.0] * 5  # 已去畸变

    # 左 -> 右 的真实相对位姿（-8° yaw，基线 120 mm，轻微俯仰）
    R_true = cv2.Rodrigues(np.array([0.012, -0.1396, 0.006]))[0]
    T_true = np.array([[-120.0], [2.5], [6.0]], np.float64)

    # 6 个位姿：棋盘格相对左相机，深度 850~1150 mm，姿态有差异
    poses = [
        (np.array([0.10, 0.25, 0.05]), np.array([-60.0, -40.0, 900.0])),
        (np.array([-0.12, -0.20, 0.10]), np.array([40.0, 30.0, 980.0])),
        (np.array([0.25, 0.05, -0.15]), np.array([10.0, -80.0, 1050.0])),
        (np.array([-0.20, 0.30, 0.08]), np.array([-90.0, 50.0, 880.0])),
        (np.array([0.05, -0.35, -0.10]), np.array([70.0, 60.0, 1120.0])),
        (np.array([0.30, 0.15, 0.20]), np.array([-30.0, 20.0, 1000.0])),
    ]

    idx_flip = flip_cols_index()
    pairs = []
    for i, (rod, tvec) in enumerate(poses):
        rvec = rod.astype(np.float64)
        tvec = tvec.reshape(3, 1).astype(np.float64)

        q_l, _ = cv2.projectPoints(obj, rvec, tvec, K_l, np.zeros(5))
        q_l = q_l.reshape(-1, 2)

        R_w = cv2.Rodrigues(rvec)[0]
        R_r = R_true @ R_w
        t_r = R_true @ tvec + T_true
        q_r, _ = cv2.projectPoints(
            obj, cv2.Rodrigues(R_r)[0], t_r, K_r, np.zeros(5)
        )
        q_r = q_r.reshape(-1, 2)

        # 检测噪声：正常对 0.12 px；最后一对 1.8 px 的大噪声（模拟抖动/失焦）
        sigma = 1.8 if i == 5 else 0.12
        q_l += rng.normal(0, sigma, q_l.shape)
        q_r += rng.normal(0, sigma, q_r.shape)

        # 右图列表整体列倒序（现场更换相机后常见的顺序不一致）
        q_r = q_r[idx_flip]

        assert q_l[:, 0].min() >= 0 and q_l[:, 0].max() < IMG_W
        assert q_l[:, 1].min() >= 0 and q_l[:, 1].max() < IMG_H
        pairs.append({
            "pair_id": f"pair_{i + 1:02d}",
            "image_size": [IMG_W, IMG_H],
            "left_pts": [[round(float(x), 4), round(float(y), 4)] for x, y in q_l],
            "right_pts": [[round(float(x), 4), round(float(y), 4)] for x, y in q_r],
        })

    payload = {
        "description": (
            "双目外参标定样例：右图角点为整体列倒序(flip_cols)；"
            "pair_06 含大噪声用于失败演示。前 5 对、阈值 0.6px 可成功。"
        ),
        "left_camera": {
            "camera_matrix": K_l.tolist(),
            "dist_coeffs": dist_zero,
        },
        "right_camera": {
            "camera_matrix": K_r.tolist(),
            "dist_coeffs": dist_zero,
        },
        "board": {"rows": ROWS, "cols": COLS, "square_size": SQUARE_SIZE},
        "image_pairs": pairs,
        "selected_pair_ids": [p["pair_id"] for p in pairs[:4]],
        "pixel_threshold": 0.6,
        "_ground_truth": {
            "note": "仅用于人工核对，标定算法不读取该字段",
            "R_left_to_right": np.round(R_true, 8).tolist(),
            "T_left_to_right": np.round(T_true.ravel(), 6).tolist(),
            "right_point_order": "flip_cols",
            "noisy_pair": "pair_06",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out} with {len(pairs)} pairs; board {ROWS}x{COLS}, sq={SQUARE_SIZE}mm")


if __name__ == "__main__":
    main()
