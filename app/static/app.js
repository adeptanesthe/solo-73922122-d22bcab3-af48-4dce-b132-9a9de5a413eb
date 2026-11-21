/* 双目外参标定前端逻辑（原生 JS，无构建步骤） */

const state = {
  payload: null,      // 上传的完整 JSON
  threshold: 0.6,
};

const $ = (id) => document.getElementById(id);

function fmt(v, digits = 5) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return "∞";
  return Number(v).toFixed(digits);
}

function matrixHtml(mat) {
  const rows = mat.map((r) => r.map((x) => fmt(x, 6).padStart(11)).join("  ")).join("\n");
  return `<div class="matrix">${rows}</div>`;
}

// ---- 上传与回显 -------------------------------------------------------------

$("fileInput").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const resp = await fetch("/api/upload", { method: "POST", body: fd });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "上传失败");
    state.payload = JSON.parse(await file.text());
    renderUploaded(body);
  } catch (err) {
    alert(`上传失败：${err.message}`);
  }
});

$("loadSampleBtn").addEventListener("click", async () => {
  try {
    const resp = await fetch("/static/sample_calibration.json");
    if (!resp.ok) throw new Error("示例文件缺失");
    const raw = await resp.json();
    const fd = new FormData();
    fd.append("file", new Blob([JSON.stringify(raw)], { type: "application/json" }),
      "sample_calibration.json");
    const up = await fetch("/api/upload", { method: "POST", body: fd });
    const body = await up.json();
    if (!up.ok) throw new Error(body.detail || "示例载入失败");
    state.payload = raw;
    renderUploaded(body);
  } catch (err) {
    alert(`示例载入失败：${err.message}`);
  }
});

function renderUploaded(meta) {
  $("sessionInfo").textContent = `批次 ${meta.session_id} · ${meta.image_pairs.length} 对图`;
  $("metaJson").textContent = JSON.stringify({
    board: meta.board,
    left_camera: meta.left_camera,
    right_camera: meta.right_camera,
  }, null, 2);
  $("uploadedMeta").classList.remove("hidden");
  $("thresholdInput").value = meta.pixel_threshold ?? 0.6;

  const tbody = $("pairTable").querySelector("tbody");
  tbody.innerHTML = "";
  meta.image_pairs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="checkbox" class="pairChk" value="${p.pair_id}" ${i < 4 ? "checked" : ""}></td>
      <td>${p.pair_id}</td>
      <td>${p.point_count}</td>
      <td>x ${p.left_bounds.x_min} ~ ${p.left_bounds.x_max}<br>y ${p.left_bounds.y_min} ~ ${p.left_bounds.y_max}</td>
      <td>x ${p.right_bounds.x_min} ~ ${p.right_bounds.x_max}<br>y ${p.right_bounds.y_min} ~ ${p.right_bounds.y_max}</td>`;
    tbody.appendChild(tr);
  });
  document.querySelectorAll(".pairChk").forEach((el) =>
    el.addEventListener("change", updateSelection));
  updateSelection();
  $("configCard").classList.remove("hidden");
  $("runCard").classList.remove("hidden");
  renderLastSuccess(meta.last_success);
  $("resultCard").classList.add("hidden");
}

function selectedIds() {
  return [...document.querySelectorAll(".pairChk:checked")].map((el) => el.value);
}

function updateSelection() {
  const n = selectedIds().length;
  $("selectedCount").innerHTML = n >= 3
    ? `已选 <b>${n}</b> 对，可以标定`
    : `已选 <b class="fail">${n}</b> 对，至少需要 3 对`;
}

// ---- 执行标定 ---------------------------------------------------------------

$("runBtn").addEventListener("click", async () => {
  const ids = selectedIds();
  if (ids.length < 3) {
    alert("至少勾选 3 对图");
    return;
  }
  const threshold = parseFloat($("thresholdInput").value);
  if (!(threshold > 0)) {
    alert("像素误差上限必须为正数");
    return;
  }
  const requestPayload = {
    data: {
      ...state.payload,
      selected_pair_ids: ids,
      pixel_threshold: threshold,
    },
  };
  $("runStatus").textContent = "计算中…";
  $("runBtn").disabled = true;
  try {
    const resp = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || "标定请求失败");
    renderResult(body);
    renderLastSuccess(body.last_success);
  } catch (err) {
    alert(`标定失败：${err.message}`);
  } finally {
    $("runBtn").disabled = false;
    $("runStatus").textContent = "";
  }
});

// ---- 结果渲染 ---------------------------------------------------------------

function pairTableHtml(pairs, threshold) {
  const head = `
    <tr><th>图对</th><th>采用排序</th><th>该对 RMSE(px)</th>
    <th>左/右 RMSE(px)</th><th>独立解算偏差<br>ΔR / ΔT方向 / 尺度比</th>
    <th>是否达标(≤${threshold})</th></tr>`;
  const rows = pairs.map((p) => `
    <tr>
      <td>${p.pair_id}</td>
      <td>${p.order_name}</td>
      <td>${fmt(p.per_pair_rms, 4)}</td>
      <td>${fmt(p.left_rms, 4)} / ${fmt(p.right_rms, 4)}</td>
      <td>${fmt(p.rotation_delta_deg, 3)}° / ${fmt(p.translation_angle_deg, 3)}° / ${fmt(p.translation_scale_ratio, 3)}</td>
      <td><span class="tag ${p.passed ? "ok" : "bad"}">${p.passed ? "达标" : "超标"}</span></td>
    </tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function candidateTableHtml(candidates, threshold) {
  const head = `
    <tr><th>候选排序</th><th>含义</th><th>联合 RMSE(px)</th>
    <th>最差图对 RMSE(px)</th><th>结论</th></tr>`;
  const rows = candidates.map((c) => `
    <tr>
      <td>${c.order_name}</td>
      <td>${c.order_description}</td>
      <td>${c.overall_rms === null ? "失败" : fmt(c.overall_rms, 4)}</td>
      <td>${fmt(c.max_pair_rms, 4)}</td>
      <td><span class="tag ${c.passed ? "ok" : "bad"}">${c.passed ? "全部达标" : "未达标"}</span>
          ${c.error ? `<div class="muted">${c.error}</div>` : ""}</td>
    </tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function renderResult(body) {
  const card = $("resultCard");
  card.classList.remove("hidden");
  let html = "";

  if (body.success && body.accepted) {
    const a = body.accepted;
    html += `
      <div class="banner ok">
        标定成功：唯一达标排序为 <b>${a.order_name}</b>（${a.order_description}），
        使用图对：<b>${a.selected_pair_ids.join("、")}</b>，
        联合重投影 RMSE = <b>${fmt(a.overall_rms, 4)} px</b>（上限 ${body.threshold} px）。
      </div>
      <div class="grid2">
        <div>
          <h3>旋转 R（左相机 → 右相机）</h3>
          ${matrixHtml(a.R)}
        </div>
        <div>
          <h3>平移 T（左相机 → 右相机）</h3>
          ${matrixHtml(a.T)}
          <p class="muted">向量形式：[${a.T_flat.map((x) => fmt(x, 4)).join(", ")}]</p>
        </div>
      </div>
      <h3>各图对采用的顺序与误差</h3>
      ${pairTableHtml(a.pairs, body.threshold)}`;
  } else {
    html += `
      <div class="banner err">
        <b>本次未更新标定结果。</b><br>${body.reason}
      </div>`;
    if (body.last_success) {
      html += `<div class="banner warn">
        本批上次成功结果仍然保留（见下方“本批上次成功结果”卡片），
        其使用图对：<b>${body.last_success.result.selected_pair_ids.join("、")}</b>。
      </div>`;
    } else {
      html += `<div class="banner warn">本批尚无成功结果，因此没有可保留的外参。</div>`;
    }
  }

  html += `<details class="meta" open><summary>全部候选排序评估（${body.candidates.length} 种）</summary>
    ${candidateTableHtml(body.candidates, body.threshold)}</details>`;
  $("resultBody").innerHTML = html;
}

function renderLastSuccess(last) {
  const card = $("lastSuccessCard");
  if (!last) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");
  const a = last.result;
  $("lastSuccessMeta").innerHTML =
    `批次 ${last.session_id} · 排序 <b>${a.order_name}</b>（${a.order_description}）·
     使用图对：<b>${a.selected_pair_ids.join("、")}</b> ·
     联合 RMSE ${fmt(a.overall_rms, 4)} px（当时上限 ${a.threshold} px）`;
  $("lastSuccessBody").innerHTML = `
    <div class="grid2">
      <div><h3>R</h3>${matrixHtml(a.R)}</div>
      <div><h3>T</h3>${matrixHtml(a.T)}</div>
    </div>
    ${pairTableHtml(a.pairs, a.threshold)}`;
}
