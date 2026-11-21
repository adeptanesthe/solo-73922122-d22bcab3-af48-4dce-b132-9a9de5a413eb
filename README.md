# 双目相机外参标定演示（FastAPI + OpenCV）

双目检测工位更换右相机后，重新标定 **左相机 → 右相机** 的相对位姿（旋转 R、平移 T）。
维修员在网页上传一份 JSON（两台相机内参、棋盘格行列数与格距、3～6 对同步图的
去畸变角点），勾选至少 3 对图并设定像素误差上限即可试标。

## 核心判定规则

1. **左图角点顺序为基准**。右图角点列表可能与左图整体顺序不一致（整体倒序、
   旋转等），后端枚举棋盘格的全部对称排序（非方形 4 种、方形 8 种）逐一验证。
2. 对每种候选排序，用所勾选的全部图对做一次固定内参的立体标定
   （`cv2.stereoCalibrate` + `CALIB_FIX_INTRINSIC`），得到 **同一组外参 R,T**
   及每对图在该外参下的逐对重投影 RMSE。
3. 仅当 **恰好一种** 排序满足“所有被选图对 RMSE 均 ≤ 像素误差上限”时，
   才接受并更新结果：
   - **零种达标** → 报错说明原因，不更新，保留本批上次成功结果；
   - **多种达标** → 数据不足以唯一确定顺序，不更新，保留上次成功结果；
   - 页面始终标明上次成功结果 **使用了哪些图对、哪种排序、误差多少**。
4. 重新上传一批新数据时，上一批的成功结果自动清空。

## 解算复用与缓存失效

维修员常反复勾选图对、拖动像素误差上限。标定流程据此做了两级复用：

1. **单次请求内共享左相机姿态**：左图角点顺序是基准，其 `solvePnP` 结果与
   右图排序无关，每对图只解算一次，供全部候选排序复用（原先每种排序重算一遍）。
2. **跨请求缓存与阈值无关的解算结果**：立体标定 R/T、逐对 RMSE、独立解算
   一致性等只取决于输入数据。`CalibrationCache` 以输入内容指纹为键缓存这些
   结果，调整像素上限时只重新套用“逐对 RMSE ≤ 上限”的达标判定与唯一性结论，
   不再调用 OpenCV。

缓存键覆盖左右相机内参与畸变、棋盘行列/格距、**全部**图对坐标（含未勾选的）、
所选图对集合、图像尺寸与批次号。因此换批、改选图对，或输入 JSON 中任何标定
数据变化都会立即失效并重算；重新上传会直接清空缓存与上次成功结果。

复用前后的 R/T、逐对 RMSE、唯一达标判定，以及失败/多解时保留的上次成功结果
完全一致（`tests/` 中有逐字段回归与失效测试）。缓存仅保存在进程内存、容量
16 条（LRU），多副本或重启后重建。

## 目录结构

```
.
├── app/
│   ├── calibration.py        # 核心算法：排序枚举、立体标定、逐对误差、唯一性判定
│   ├── main.py               # FastAPI 路由、JSON 校验、上次结果留存
│   └── static/
│       ├── index.html        # 单页前端（原生 JS）
│       ├── app.js
│       ├── style.css
│       └── sample_calibration.json   # 可一键载入的试标样例
├── samples/
│   └── sample_calibration.json       # 同一份样例（curl 上传用）
├── scripts/make_sample.py    # 样例生成脚本（真实投影 + 噪声，含真值 _ground_truth）
├── tests/                    # pytest：核心算法 21 项 + API 端到端 15 项
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 用 Docker Compose 启动（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

页面上点 **“载入内置示例”** 即可直接试标，无需自己准备数据。

命令行上传样例：

```bash
curl -F "file=@samples/sample_calibration.json;type=application/json" \
  http://localhost:8000/api/upload
```

## 本地直接运行（无需 Docker）

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 输入 JSON 格式

```json
{
  "left_camera":  { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [0,0,0,0,0] },
  "right_camera": { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist_coeffs": [0,0,0,0,0] },
  "board": { "rows": 7, "cols": 5, "square_size": 25.0 },
  "image_pairs": [
    {
      "pair_id": "pair_01",
      "image_size": [1280, 960],
      "left_pts":  [[x, y], "...35 个角点"],
      "right_pts": [[x, y], "...顺序可能整体倒序"]
    }
  ],
  "selected_pair_ids": ["pair_01", "pair_02", "pair_03", "pair_04"],
  "pixel_threshold": 0.6
}
```

- 角点为 **已去畸变** 的像素坐标；`dist_coeffs` 可传全 0。
- `image_pairs` 数量 3～6；`selected_pair_ids` 至少 3 个；`pixel_threshold > 0`。
- `image_size` 可省略，缺省时由角点坐标范围推断。

## 内置样例说明

`scripts/make_sample.py` 用已知内参与已知左→右 R/T 正投影生成 6 对图：

| 图对 | 用途 |
|---|---|
| pair_01～pair_05 | 正常数据（0.12 px 检测噪声），右图角点为 **整体列倒序 flip_cols** |
| pair_06 | 注入 1.8 px 大噪声，用于演示“无排序达标 → 保留上次结果” |

推荐试标路径：

1. 选前 4 或前 5 对、阈值 **0.6 px** → 成功，唯一排序 `flip_cols`，
   R/T 与 JSON 内 `_ground_truth` 字段高度一致；
2. 勾选全部 6 对（含 pair_06）、阈值 0.6 → 无排序达标，页面报错且
   保留第 1 步的成功结果并标明所用图对；
3. 选前 4 对、阈值改成 **100 px** → 3 种排序同时达标（歧义），拒绝更新。

重新生成样例：

```bash
python scripts/make_sample.py samples/sample_calibration.json
cp samples/sample_calibration.json app/static/sample_calibration.json
```

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/upload` | multipart 上传标定 JSON，建立新批次并清空旧结果 |
| POST | `/api/calibrate` | 执行标定，返回 R/T、每对排序与误差、全部候选排序诊断 |
| GET  | `/api/last-success` | 查看本批上次成功结果（含所用图对） |
| GET  | `/api/health` | 健康检查 |

## 运行测试

```bash
pytest tests/ -v
```

覆盖范围：排序枚举与真值恢复、失败/多解语义；以及缓存复用与失效——
仅改阈值不触发 OpenCV 求解、冷算左姿每对只解一次、改选图对/改 JSON/换批
均重算，复用前后 R/T、逐对 RMSE、达标判定与上次成功结果逐字段一致。

## 备注

- 结果保存在进程内存中（演示用途）；多副本部署或重启后不持久化。
- 解算缓存同样只在进程内，容量 16 条 LRU；换批（重新上传）时清空。
- 缓存按内容指纹校验，不依赖批次内“图对 ID 不变即数据不变”的假设：
  直接调用 `/api/calibrate` 传入改动过的 JSON 也会正确失效。
- OpenCV 4.x 与 5.x 的 `stereoCalibrate` 返回签名不同，代码已兼容
  （5.x 无 perViewErrors，逐对误差在应用层重投影计算）。
