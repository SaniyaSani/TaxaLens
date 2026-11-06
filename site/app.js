const config = window.TAXALENS_CONFIG || {};
const apiUrl = String(config.apiUrl || "").replace(/\/$/, "");
const input = document.querySelector("#specimen");
const preview = document.querySelector("#preview");
const dropCopy = document.querySelector("#drop-copy");
const dropZone = document.querySelector("#drop-zone");
const button = document.querySelector("#identify");
const statusNode = document.querySelector("#status");
let selectedFile = null;

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    statusNode.textContent = "Please choose a JPG, PNG or WEBP image.";
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  dropCopy.style.display = "none";
  button.disabled = false;
  statusNode.textContent = file.name;
}

input.addEventListener("change", () => setFile(input.files[0]));
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add("drag");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove("drag");
}));
dropZone.addEventListener("drop", event => setFile(event.dataTransfer.files[0]));

function probability(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) : 0;
}

function renderRank(target, rows = []) {
  target.replaceChildren();
  if (!rows.length) {
    target.textContent = "No supported candidate";
    return;
  }
  rows.slice(0, 5).forEach(row => {
    const score = probability(row.probability);
    const item = document.createElement("div");
    item.className = "taxon-row";
    item.innerHTML = `<div><span>${escapeHtml(row.taxon || "—")}</span><strong>${(score * 100).toFixed(1)}%</strong></div><div class="mini-meter"><span style="width:${score * 100}%"></span></div>`;
    target.append(item);
  });
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function renderGuidance(guide = {}) {
  const verification = document.querySelector("#verification");
  const stepsNode = document.querySelector("#steps");
  const resourcesNode = document.querySelector("#resources");
  const steps = guide.steps || [];
  const resources = guide.resources || [];
  stepsNode.replaceChildren();
  resourcesNode.replaceChildren();

  steps.forEach(step => {
    const card = document.createElement("article");
    card.className = "step";
    card.innerHTML = `<h3>${escapeHtml(step.title || step.character || "Inspect")}</h3><p>${escapeHtml(step.inspect || step.instruction || step.description || "")}</p>${step.why ? `<p><em>Why: ${escapeHtml(step.why)}</em></p>` : ""}`;
    stepsNode.append(card);
  });
  resources.forEach(resource => {
    const link = document.createElement("a");
    link.className = "resource";
    link.href = resource.url || "#";
    link.target = "_blank";
    link.rel = "noopener";
    link.innerHTML = `<strong>${escapeHtml(resource.title || "Identification resource")}</strong><small>${escapeHtml(resource.scope || resource.verification || "")}</small>`;
    resourcesNode.append(link);
  });
  verification.hidden = !(steps.length || resources.length);
}

function renderNeighbours(rows = []) {
  const section = document.querySelector("#neighbour-section");
  const gallery = document.querySelector("#neighbours");
  gallery.replaceChildren();
  rows.slice(0, 8).forEach(row => {
    const url = row.image_url || row.url || row.source_url;
    if (!url) return;
    const figure = document.createElement("figure");
    figure.className = "neighbour";
    figure.innerHTML = `<img src="${escapeHtml(url)}" alt="Similar reference specimen" loading="lazy"><figcaption>${escapeHtml(row.species || row.genus || row.family || "Reference specimen")}</figcaption>`;
    gallery.append(figure);
  });
  section.hidden = !gallery.children.length;
}

function renderResult(payload) {
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#result").hidden = false;
  const order = payload.order || {};
  const accepted = payload.accepted || {};
  const predictions = payload.predictions || {};
  const selectedOrder = order.selected_order || order.predicted_order || "Uncertain";
  const confidence = probability(order.confidence);

  document.querySelector("#accepted-taxon").textContent = accepted.taxon || "Further evidence required";
  document.querySelector("#accepted-rank").textContent = accepted.rank ? `Accepted at ${accepted.rank} level` : "No rank crossed the acceptance threshold";
  document.querySelector("#order-badge").textContent = selectedOrder;
  document.querySelector("#router-confidence").textContent = `${(confidence * 100).toFixed(1)}%`;
  document.querySelector("#router-meter").style.width = `${confidence * 100}%`;
  renderRank(document.querySelector("#family-list"), predictions.family);
  renderRank(document.querySelector("#genus-list"), predictions.genus);
  renderRank(document.querySelector("#species-list"), predictions.species);
  renderGuidance(payload.guided_verification);
  renderNeighbours(payload.similar_specimens);
}

button.addEventListener("click", async () => {
  if (!selectedFile) return;
  if (!apiUrl || apiUrl.includes("YOUR-TAXALENS-API")) {
    statusNode.textContent = "The API address still needs to be configured in site/config.js.";
    return;
  }

  button.disabled = true;
  button.textContent = "Looking closely…";
  statusNode.textContent = "Encoding the image and comparing reference specimens…";
  const form = new FormData();
  form.append("image", selectedFile);
  form.append("mode", document.querySelector('input[name="order"]:checked').value);

  try {
    const response = await fetch(`${apiUrl}/v1/identify`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Inference request failed");
    renderResult(payload);
    statusNode.textContent = "Identification complete.";
  } catch (error) {
    statusNode.textContent = `Could not identify this image: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Identify specimen";
  }
});

