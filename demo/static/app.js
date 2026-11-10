// DocuLens live demo: thin client over /api/options and /api/run.
//
// Two modes, one endpoint. "Upload images" posts each chosen file to /api/run
// and accumulates a results table; "Controlled corpus" runs one generated page
// under a chosen degradation. Neither does any image processing here -- every
// number and every picture comes from the server, which calls src/doculens.
"use strict";

const $ = (id) => document.getElementById(id);
const CLEAN = "none";

let OPTIONS = null;

/** Uploaded files waiting to be processed. */
let QUEUE = [];
/** One entry per processed upload: { name, size, data }. */
let RESULTS = [];

function fmt(value, digits = 4) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function signed(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  const text = Number(value).toFixed(digits);
  return Number(value) > 0 ? `+${text}` : text.replace("-", "−");
}

function fillSelect(select, values, selected) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = String(value.value ?? value);
    option.textContent = String(value.label ?? value);
    select.appendChild(option);
  }
  if (selected !== undefined && values.some((v) => String(v.value ?? v) === String(selected))) {
    select.value = String(selected);
  }
}

function openLightbox(src, caption) {
  $("lightbox-img").src = src;
  $("lightbox-caption").textContent = caption;
  $("lightbox").hidden = false;
}

/** Build a clickable stage filmstrip into `container`. */
function buildFilmstrip(container, stages) {
  container.innerHTML = "";
  stages.forEach((stage, index) => {
    const card = document.createElement("div");
    card.className = "shot";
    const img = document.createElement("img");
    img.src = stage.png;
    img.alt = stage.name;
    const name = document.createElement("div");
    name.className = "n";
    name.textContent = `${String(index).padStart(2, "0")} ${stage.name}`;
    const params = document.createElement("div");
    params.className = "p";
    params.textContent = stage.params || "";
    card.append(img, name, params);
    card.addEventListener("click", () => openLightbox(stage.png, `${index} — ${stage.name}`));
    container.appendChild(card);
  });
}

// ===================================================================
// Upload mode
// ===================================================================

function setUpStatus(message, isError = false) {
  const el = $("up-status");
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
}

function kb(bytes) {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} kB`;
}

function renderQueue() {
  const list = $("queue-list");
  list.innerHTML = "";
  QUEUE.forEach((file, index) => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${file.name} · ${kb(file.size)}`;
    const remove = document.createElement("button");
    remove.className = "x";
    remove.textContent = "×";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      QUEUE.splice(index, 1);
      renderQueue();
    });
    item.append(label, remove);
    list.appendChild(item);
  });
  $("queue").hidden = QUEUE.length === 0;
  $("up-run").disabled = QUEUE.length === 0;
  if (QUEUE.length) {
    setUpStatus(`${QUEUE.length} file(s) ready. Press "Run full pipeline".`);
  }
}

function addFiles(fileList) {
  for (const file of fileList) {
    if (!QUEUE.some((f) => f.name === file.name && f.size === file.size)) QUEUE.push(file);
  }
  renderQueue();
}

/** Classify one result into a verdict class and label. */
function verdictOf(data) {
  const base = data.baseline;
  const pipe = data.result;
  if (data.scored) {
    const delta = pipe.cer - base.cer;
    if (Math.abs(delta) < 1e-9) return { cls: "same", text: "no change" };
    return delta < 0
      ? { cls: "better", text: "pipeline helps" }
      : { cls: "worse", text: "pipeline hurts" };
  }
  if (base.confidence === null || pipe.confidence === null) {
    return { cls: "same", text: "not comparable" };
  }
  const delta = pipe.confidence - base.confidence;
  if (Math.abs(delta) < 0.05) return { cls: "same", text: "no change" };
  return delta > 0
    ? { cls: "better", text: "more confident" }
    : { cls: "worse", text: "less confident" };
}

function cell(row, text, className = "") {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
}

function appendSummaryRow(entry) {
  const { name, size, data } = entry;
  const base = data.baseline;
  const pipe = data.result;
  const verdict = verdictOf(data);
  const row = document.createElement("tr");

  const first = cell(row, name, "file");
  if (!data.scored) {
    const tag = document.createElement("span");
    tag.className = "pill";
    tag.textContent = "no ground truth";
    tag.title = "CER needs a reference text; this page is described by OCR confidence instead.";
    first.appendChild(tag);
  }
  cell(row, `${data.width}×${data.height}`, "dim");
  cell(row, fmt(base.cer), "num");
  cell(row, fmt(pipe.cer), "num");
  cell(row, data.scored ? signed(pipe.cer - base.cer) : "—", `num ${verdict.cls}`);
  cell(row, fmt(base.confidence, 1), "num");
  cell(row, fmt(pipe.confidence, 1), "num");
  cell(
    row,
    base.confidence !== null && pipe.confidence !== null
      ? signed(pipe.confidence - base.confidence, 1)
      : "—",
    "num",
  );
  cell(row, pipe.n_words === null ? "—" : String(pipe.n_words), "num");
  cell(row, String(Math.round(pipe.preprocess_ms)), "num");
  cell(row, verdict.text, `verdict-cell ${verdict.cls}`);

  row.addEventListener("click", () => {
    const target = $(`detail-${entry.index}`);
    if (target) {
      target.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
  $("up-rows").appendChild(row);
}

function mean(values) {
  const usable = values.filter((v) => v !== null && v !== undefined);
  return usable.length ? usable.reduce((a, b) => a + b, 0) / usable.length : null;
}

function renderFooter() {
  const scored = RESULTS.filter((entry) => entry.data.scored);
  const foot = $("up-foot");
  foot.innerHTML = "";
  if (!RESULTS.length) return;

  const row = document.createElement("tr");
  cell(row, `mean over ${RESULTS.length} file(s)`, "file");
  cell(row, "", "");
  cell(row, fmt(mean(scored.map((e) => e.data.baseline.cer))), "num");
  cell(row, fmt(mean(scored.map((e) => e.data.result.cer))), "num");
  const baseMean = mean(scored.map((e) => e.data.baseline.cer));
  const pipeMean = mean(scored.map((e) => e.data.result.cer));
  cell(
    row,
    baseMean === null || pipeMean === null ? "—" : signed(pipeMean - baseMean),
    `num ${pipeMean < baseMean ? "better" : pipeMean > baseMean ? "worse" : "same"}`,
  );
  const baseConf = mean(RESULTS.map((e) => e.data.baseline.confidence));
  const pipeConf = mean(RESULTS.map((e) => e.data.result.confidence));
  cell(row, fmt(baseConf, 1), "num");
  cell(row, fmt(pipeConf, 1), "num");
  cell(row, baseConf === null || pipeConf === null ? "—" : signed(pipeConf - baseConf, 1), "num");
  cell(row, "", "");
  cell(row, String(Math.round(mean(RESULTS.map((e) => e.data.result.preprocess_ms)) || 0)), "num");
  cell(row, "", "");
  foot.appendChild(row);

  const unscored = RESULTS.length - scored.length;
  $("up-note").textContent =
    `CER is an exact edit distance against the generated reference text, so it is only ` +
    `available for the ${scored.length} file(s) whose name matches a corpus document. ` +
    (unscored
      ? `The other ${unscored} have no reference text: for those, confidence is Tesseract's ` +
        `own mean word confidence (0-100), which says how sure the recogniser is, not how ` +
        `right it is. `
      : "") +
    `Click a row to open its stage-by-stage breakdown.`;
}

function appendDetail(entry) {
  const { data, index } = entry;
  const details = document.createElement("details");
  details.className = "detail";
  details.id = `detail-${index}`;
  details.open = index === 0;

  const summary = document.createElement("summary");
  const verdict = verdictOf(data);
  summary.innerHTML =
    `<strong>${entry.name}</strong> <span class="dim">${data.label}</span>` +
    `<span class="chip ${verdict.cls}">${verdict.text}</span>`;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "detail-body";

  const board = document.createElement("div");
  board.className = "scoreboard";
  board.innerHTML = `
    <article class="card baseline">
      <h2>Baseline <span class="tag">no preprocessing</span></h2>
      <div class="metrics">
        <div class="metric"><span class="val">${fmt(data.baseline.cer)}</span><span class="key">CER</span></div>
        <div class="metric"><span class="val">${fmt(data.baseline.confidence, 1)}</span><span class="key">confidence</span></div>
        <div class="metric"><span class="val sm">${Math.round(data.baseline.ocr_ms)}</span><span class="key">OCR ms</span></div>
      </div>
    </article>
    <div class="arrow ${verdict.cls}"><span>${verdict.text}</span></div>
    <article class="card pipeline">
      <h2>${data.pipeline}</h2>
      <div class="metrics">
        <div class="metric"><span class="val">${fmt(data.result.cer)}</span><span class="key">CER</span></div>
        <div class="metric"><span class="val">${fmt(data.result.confidence, 1)}</span><span class="key">confidence</span></div>
        <div class="metric"><span class="val sm">${Math.round(data.result.ocr_ms)}</span><span class="key">OCR ms</span></div>
      </div>
    </article>`;
  body.appendChild(board);

  const stagesHead = document.createElement("h3");
  stagesHead.innerHTML = `Stage by stage <span class="hint">${data.stage_names.length} images · click to enlarge</span>`;
  const strip = document.createElement("div");
  strip.className = "filmstrip";
  buildFilmstrip(strip, data.stages);
  body.append(stagesHead, strip);

  const texts = document.createElement("div");
  texts.className = "texts";
  const columns = [
    ["Ground truth", data.ground_truth ?? "(no reference text for this page)"],
    ["Baseline OCR", data.baseline.text || "(nothing recognised)"],
    [`${data.pipeline} OCR`, data.result.text || "(nothing recognised)"],
  ];
  for (const [heading, content] of columns) {
    const article = document.createElement("article");
    const h3 = document.createElement("h3");
    h3.textContent = heading;
    const pre = document.createElement("pre");
    pre.textContent = content;
    article.append(h3, pre);
    texts.appendChild(article);
  }
  body.appendChild(texts);

  details.appendChild(body);
  $("up-details").appendChild(details);
}

async function processFile(file, pipelineName) {
  const form = new FormData();
  form.append("file", file);
  form.append("pipeline", pipelineName);
  const response = await fetch("/api/run", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);
  return data;
}

async function runBatch() {
  const pipelineName = $("up-pipeline").value;
  $("up-run").disabled = true;
  $("up-clear").disabled = true;
  RESULTS = [];
  $("up-rows").innerHTML = "";
  $("up-foot").innerHTML = "";
  $("up-details").innerHTML = "";
  $("up-summary-wrap").hidden = false;

  const started = performance.now();
  const failures = [];
  for (let i = 0; i < QUEUE.length; i += 1) {
    const file = QUEUE[i];
    setUpStatus(`Processing ${i + 1} of ${QUEUE.length}: ${file.name} — ${pipelineName} …`);
    try {
      const data = await processFile(file, pipelineName);
      const entry = { name: file.name, size: file.size, data, index: RESULTS.length };
      RESULTS.push(entry);
      appendSummaryRow(entry);
      appendDetail(entry);
      renderFooter();
    } catch (error) {
      failures.push(`${file.name}: ${error.message}`);
    }
  }
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  $("up-run").disabled = false;
  $("up-clear").disabled = false;
  if (failures.length) {
    setUpStatus(`${RESULTS.length} processed in ${seconds}s. Failed — ${failures.join("; ")}`, true);
  } else {
    setUpStatus(
      `Done: ${RESULTS.length} file(s) through ${pipelineName} in ${seconds}s ` +
        `(2 OCR passes per file, baseline and pipeline).`,
    );
  }
}

const CSV_COLUMNS = [
  ["file", (e) => e.name],
  ["width", (e) => e.data.width],
  ["height", (e) => e.data.height],
  ["pipeline", (e) => e.data.pipeline],
  ["scored", (e) => e.data.scored],
  ["baseline_cer", (e) => e.data.baseline.cer],
  ["pipeline_cer", (e) => e.data.result.cer],
  ["baseline_wer", (e) => e.data.baseline.wer],
  ["pipeline_wer", (e) => e.data.result.wer],
  ["baseline_confidence", (e) => e.data.baseline.confidence],
  ["pipeline_confidence", (e) => e.data.result.confidence],
  ["baseline_words", (e) => e.data.baseline.n_words],
  ["pipeline_words", (e) => e.data.result.n_words],
  ["preprocess_ms", (e) => Math.round(e.data.result.preprocess_ms)],
  ["baseline_ocr_ms", (e) => Math.round(e.data.baseline.ocr_ms)],
  ["pipeline_ocr_ms", (e) => Math.round(e.data.result.ocr_ms)],
  ["verdict", (e) => verdictOf(e.data).text],
];

function downloadCsv() {
  if (!RESULTS.length) return;
  const escape = (value) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const lines = [CSV_COLUMNS.map(([header]) => header).join(",")];
  for (const entry of RESULTS) {
    lines.push(CSV_COLUMNS.map(([, get]) => escape(get(entry))).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "doculens_upload_results.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

// ===================================================================
// Corpus mode
// ===================================================================

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
}

function refreshLevels() {
  const degradation = $("degradation").value;
  const levels = (OPTIONS.degradations[degradation] || [0]).map((n) => ({
    value: n,
    label: degradation === CLEAN ? "n/a" : `level ${n}`,
  }));
  fillSelect($("level"), levels, levels[levels.length - 1]?.value);
  $("level").disabled = degradation === CLEAN;
}

function renderScores(data) {
  const base = data.baseline;
  const pipe = data.result;

  $("base-cer").textContent = fmt(base.cer);
  $("base-wer").textContent = fmt(base.wer);
  $("base-ms").textContent = Math.round(base.ocr_ms);
  $("pipe-cer").textContent = fmt(pipe.cer);
  $("pipe-wer").textContent = fmt(pipe.wer);
  $("pipe-ms").textContent = Math.round(pipe.ocr_ms);
  $("pipeline-title").textContent = data.pipeline;
  $("pipe-text-title").textContent = `${data.pipeline} OCR`;

  const verdict = $("verdict");
  const text = $("verdict-text");
  verdict.classList.remove("better", "worse", "same");

  const delta = pipe.cer - base.cer;
  if (Math.abs(delta) < 1e-9) {
    verdict.classList.add("same");
    text.textContent = "no change";
  } else if (delta < 0) {
    const drop = base.cer > 0 ? ` (−${Math.round((-delta / base.cer) * 100)}%)` : "";
    verdict.classList.add("better");
    text.textContent = `CER ${fmt(base.cer)} → ${fmt(pipe.cer)}${drop}\npipeline helps`;
  } else {
    verdict.classList.add("worse");
    text.textContent = `CER ${fmt(base.cer)} → ${fmt(pipe.cer)}\npipeline hurts here`;
  }
  $("scoreboard").hidden = false;
}

function renderTexts(data) {
  $("gt").textContent = data.ground_truth ?? "(no reference text for this page)";
  $("base-text").textContent = data.baseline.text || "(nothing recognised)";
  $("pipe-text").textContent = data.result.text || "(nothing recognised)";
  $("text-wrap").hidden = false;
}

async function run() {
  const button = $("run");
  button.disabled = true;
  setStatus("Degrading, preprocessing and running OCR twice…");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doc_id: $("doc").value,
        degradation: $("degradation").value,
        level: Number($("level").value || 0),
        pipeline: $("pipeline").value,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);

    renderScores(data);
    $("condition").textContent = `${data.label} · ${data.pipeline}`;
    buildFilmstrip($("stages"), data.stages);
    $("stages-wrap").hidden = false;
    renderTexts(data);
    setStatus(
      `${data.label} · ${data.pipeline} · preprocessing ${Math.round(
        data.result.preprocess_ms,
      )} ms · click any stage to enlarge`,
    );
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}

// ===================================================================
// Startup and wiring
// ===================================================================

function showView(mode) {
  const upload = mode === "upload";
  $("view-upload").hidden = !upload;
  $("view-corpus").hidden = upload;
  $("tab-upload").classList.toggle("active", upload);
  $("tab-corpus").classList.toggle("active", !upload);
  $("tab-upload").setAttribute("aria-selected", String(upload));
  $("tab-corpus").setAttribute("aria-selected", String(!upload));
}

async function loadOptions() {
  const response = await fetch("/api/options");
  OPTIONS = await response.json();

  fillSelect($("pipeline"), OPTIONS.pipelines, OPTIONS.default_pipeline);
  fillSelect($("up-pipeline"), OPTIONS.pipelines, OPTIONS.default_pipeline);
  $("engine").textContent = `tesseract ${OPTIONS.tesseract_config}`;

  if (!OPTIONS.corpus_ready) {
    setStatus("No corpus found. Run `make data` to enable this tab.", true);
    $("run").disabled = true;
    return;
  }
  fillSelect(
    $("doc"),
    OPTIONS.documents.map((d) => ({
      value: d.doc_id,
      label: `${d.doc_id} — ${d.doc_type} (${d.split})`,
    })),
  );
  fillSelect($("degradation"), Object.keys(OPTIONS.degradations), OPTIONS.default_degradation);
  refreshLevels();
  setStatus("Pick a document and a degradation, then run the pipeline.");
}

$("tab-upload").addEventListener("click", () => showView("upload"));
$("tab-corpus").addEventListener("click", () => showView("corpus"));

$("dropzone").addEventListener("click", () => $("files").click());
$("dropzone").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("files").click();
  }
});
$("files").addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = "";
});
for (const type of ["dragenter", "dragover"]) {
  $("dropzone").addEventListener(type, (event) => {
    event.preventDefault();
    $("dropzone").classList.add("over");
  });
}
for (const type of ["dragleave", "drop"]) {
  $("dropzone").addEventListener(type, (event) => {
    event.preventDefault();
    $("dropzone").classList.remove("over");
  });
}
$("dropzone").addEventListener("drop", (event) => {
  if (event.dataTransfer?.files?.length) addFiles(event.dataTransfer.files);
});
$("up-run").addEventListener("click", runBatch);
$("up-clear").addEventListener("click", () => {
  QUEUE = [];
  RESULTS = [];
  renderQueue();
  $("up-rows").innerHTML = "";
  $("up-foot").innerHTML = "";
  $("up-details").innerHTML = "";
  $("up-summary-wrap").hidden = true;
  setUpStatus("Add one or more document images, then run the pipeline.");
});
$("up-csv").addEventListener("click", downloadCsv);

$("degradation").addEventListener("change", refreshLevels);
$("run").addEventListener("click", run);
$("lightbox-close").addEventListener("click", () => ($("lightbox").hidden = true));
$("lightbox").addEventListener("click", (event) => {
  if (event.target.id === "lightbox") $("lightbox").hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("lightbox").hidden = true;
});

renderQueue();
loadOptions().catch((error) => setUpStatus(`could not load options: ${error.message}`, true));
