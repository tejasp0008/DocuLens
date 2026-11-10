// DocuLens live demo: thin client over /api/options and /api/run.
"use strict";

const $ = (id) => document.getElementById(id);
const CLEAN = "none";

let OPTIONS = null;

function setStatus(message, isError = false) {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
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

function refreshLevels() {
  const degradation = $("degradation").value;
  const levels = (OPTIONS.degradations[degradation] || [0]).map((n) => ({
    value: n,
    label: degradation === CLEAN ? "n/a" : `level ${n}`,
  }));
  fillSelect($("level"), levels, levels[levels.length - 1]?.value);
  $("level").disabled = degradation === CLEAN;
}

async function loadOptions() {
  const response = await fetch("/api/options");
  OPTIONS = await response.json();

  if (!OPTIONS.corpus_ready) {
    setStatus("No corpus found. Run `make data` first, then reload this page.", true);
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
  fillSelect($("pipeline"), OPTIONS.pipelines, OPTIONS.default_pipeline);
  refreshLevels();
  $("engine").textContent = `tesseract ${OPTIONS.tesseract_config}`;
  setStatus("Pick a document and a degradation, then run the pipeline.");
}

function fmt(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(4);
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

  if (base.cer === null || pipe.cer === null) {
    // An uploaded page has no ground truth, so no CER can be computed.
    verdict.classList.add("same");
    text.textContent = "no ground truth\nfor an uploaded page";
    $("scoreboard").hidden = false;
    return;
  }
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

function renderStages(stages) {
  const strip = $("stages");
  strip.innerHTML = "";
  stages.forEach((stage, index) => {
    const card = document.createElement("div");
    card.className = "shot";
    card.innerHTML = `
      <img src="${stage.png}" alt="${stage.name}">
      <div class="n">${String(index).padStart(2, "0")} ${stage.name}</div>
      <div class="p">${stage.params || ""}</div>`;
    card.addEventListener("click", () => openLightbox(stage.png, `${index} — ${stage.name}`));
    strip.appendChild(card);
  });
  $("stages-wrap").hidden = false;
}

function renderCondition(data) {
  $("condition").textContent = `${data.label} · ${data.pipeline}`;
}

function renderTexts(data) {
  $("gt").textContent = data.ground_truth ?? "(uploaded page — no ground truth available)";
  $("base-text").textContent = data.baseline.text || "(nothing recognised)";
  $("pipe-text").textContent = data.result.text || "(nothing recognised)";
  $("text-wrap").hidden = false;
}

function openLightbox(src, caption) {
  $("lightbox-img").src = src;
  $("lightbox-caption").textContent = caption;
  $("lightbox").hidden = false;
}

async function run() {
  const button = $("run");
  button.disabled = true;
  setStatus("Degrading, preprocessing and running OCR twice…");

  const file = $("file").files[0];
  let request;
  if (file) {
    const form = new FormData();
    form.append("file", file);
    form.append("pipeline", $("pipeline").value);
    request = fetch("/api/run", { method: "POST", body: form });
  } else {
    request = fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doc_id: $("doc").value,
        degradation: $("degradation").value,
        level: Number($("level").value || 0),
        pipeline: $("pipeline").value,
      }),
    });
  }

  try {
    const response = await request;
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);

    renderScores(data);
    renderCondition(data);
    renderStages(data.stages);
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

$("degradation").addEventListener("change", refreshLevels);
$("run").addEventListener("click", run);
$("file").addEventListener("change", () => {
  const chosen = Boolean($("file").files[0]);
  for (const id of ["doc", "degradation", "level"]) $(id).disabled = chosen;
  if (!chosen) refreshLevels();
  setStatus(chosen ? "Uploaded page selected — CER needs ground truth, so it will show n/a." : "");
});
$("lightbox-close").addEventListener("click", () => ($("lightbox").hidden = true));
$("lightbox").addEventListener("click", (event) => {
  if (event.target.id === "lightbox") $("lightbox").hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("lightbox").hidden = true;
  if (event.key === "Enter" && !$("run").disabled) run();
});

loadOptions().catch((error) => setStatus(`could not load options: ${error.message}`, true));
