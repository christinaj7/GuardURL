/**
 * GuardURL — Frontend Dashboard Logic
 * Handles real-time API communication, SVG gauge animation,
 * SHAP waterfall rendering, adaptive RL display, and model metrics hub.
 */

const API_BASE = ""; // Relative path to FastAPI backend

document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  loadModelMetrics();
  loadHistory();
});

// Tab Switching
function switchTab(tabName) {
  const tabs = ["scanner", "models", "history"];
  tabs.forEach(t => {
    const btn = document.getElementById(`tab-t`);
    const view = document.getElementById(`view-${t}`);
    const navBtn = document.getElementById(`tab-${t}`);

    if (t === tabName) {
      view.classList.remove("hidden");
      if (navBtn) navBtn.classList.add("active");
    } else {
      view.classList.add("hidden");
      if (navBtn) navBtn.classList.remove("active");
    }
  });

  if (tabName === "models") {
    loadModelMetrics();
  } else if (tabName === "history") {
    loadHistory();
  }
}

// Health Check
async function checkBackendHealth() {
  const statusText = document.getElementById("system-status-text");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      statusText.textContent = `AI Core Online (${data.model_type})`;
    } else {
      statusText.textContent = "AI Core Offline";
    }
  } catch (e) {
    statusText.textContent = "API Disconnected";
  }
}

// Quick Sample Filler
function fillAndScan(url) {
  const input = document.getElementById("url-input");
  input.value = url;
  switchTab("scanner");
  handleScan(new Event("submit"));
}

// Scan Submission
async function handleScan(event) {
  if (event) event.preventDefault();

  const input = document.getElementById("url-input");
  const url = input.value.trim();
  const errorCard = document.getElementById("error-card");
  const errorMsg = document.getElementById("error-msg");
  const loader = document.getElementById("scan-loader");
  const resultsContainer = document.getElementById("results-container");
  const scanBtn = document.getElementById("scan-btn");

  if (!url) return;

  // Reset UI
  errorCard.classList.add("hidden");
  resultsContainer.classList.add("hidden");
  loader.classList.remove("hidden");
  scanBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url, enable_rl: true })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "URL analysis failed. Please verify input.");
    }

    renderResults(data);
    loadHistory(); // update audit trail

  } catch (err) {
    errorMsg.textContent = err.message || "An unexpected error occurred.";
    errorCard.classList.remove("hidden");
  } finally {
    loader.classList.add("hidden");
    scanBtn.disabled = false;
  }
}

// Render Results
function renderResults(data) {
  const resultsContainer = document.getElementById("results-container");
  resultsContainer.classList.remove("hidden");

  const isPhishing = (data.prediction === "PHISHING");
  const isUncertain = (data.confidence_level === "UNCERTAIN");

  // 1. Verdict Card
  const verdictCard = document.getElementById("verdict-card");
  const verdictTitle = document.getElementById("verdict-title");
  const resultUrl = document.getElementById("result-url-text");
  const verdictShield = document.getElementById("verdict-shield");
  const shieldIcon = document.getElementById("shield-icon");

  verdictCard.className = "verdict-card glass-panel" + (isPhishing ? " phishing" : (isUncertain ? " uncertain" : ""));
  verdictTitle.className = "verdict-title" + (isPhishing ? " phishing" : (isUncertain ? " uncertain" : ""));
  verdictTitle.textContent = data.prediction;
  resultUrl.textContent = data.url;

  verdictShield.className = "verdict-shield" + (isPhishing ? " phishing" : (isUncertain ? " uncertain" : ""));

  if (isPhishing) {
    shieldIcon.innerHTML = `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>`;
  } else if (isUncertain) {
    shieldIcon.innerHTML = `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><circle cx="12" cy="12" r="3"/>`;
  } else {
    shieldIcon.innerHTML = `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>`;
  }

  // 2. Confidence Gauge
  const gaugeCircle = document.getElementById("gauge-circle");
  const gaugeText = document.getElementById("gauge-text");
  const confLevel = document.getElementById("confidence-level-tag");

  const confPercent = Math.round(data.confidence * 100);
  gaugeText.textContent = `${confPercent}%`;
  confLevel.textContent = data.confidence_level;

  gaugeCircle.className = "circle" + (isPhishing ? " phishing" : (isUncertain ? " uncertain" : ""));
  // Smoothly trigger stroke animation
  setTimeout(() => {
    gaugeCircle.setAttribute("stroke-dasharray", `${confPercent}, 100`);
  }, 50);

  // 3. Risk Card
  const riskBadge = document.getElementById("risk-badge");
  const latencyTag = document.getElementById("latency-tag");

  riskBadge.className = "risk-badge" + (isPhishing ? " phishing" : (isUncertain ? " uncertain" : ""));
  riskBadge.textContent = `${data.risk_level} RISK`;
  latencyTag.textContent = `Inference: ${data.latency_ms || 1.2}ms`;

  // 4. Important Factors (WHY?)
  const factorsList = document.getElementById("factors-list");
  factorsList.innerHTML = "";

  (data.important_factors || []).forEach(factor => {
    const li = document.createElement("li");
    li.className = "factor-item" + (isPhishing ? " phish" : " legit");
    li.innerHTML = `
      <span class="factor-icon">${isPhishing ? "✕" : "✓"}</span>
      <span class="factor-text">${factor}</span>
    `;
    factorsList.appendChild(li);
  });

  // 5. Adaptive RL Decision Layer
  const rl = data.adaptive_rl_assessment;
  if (rl) {
    document.getElementById("rl-action-text").textContent = rl.action;
    document.getElementById("rl-tier-text").textContent = `${rl.confidence_tier} CONFIDENCE`;
    document.getElementById("rl-threat-text").textContent = rl.threat_flags_detected ? "Acute Threat Signatures Detected" : "Normal Baseline Profile";
    document.getElementById("rl-rationale-text").textContent = rl.rationale;

    const addBox = document.getElementById("rl-additional-analysis-box");
    if (rl.additional_analysis) {
      addBox.classList.remove("hidden");
      const notes = rl.additional_analysis;
      document.getElementById("rl-additional-notes").innerHTML = `
        <p><strong>Baseline Legit Prob:</strong> ${(notes.initial_prob_legitimate * 100).toFixed(1)}%</p>
        <p><strong>Adjusted Legit Prob:</strong> ${(notes.adjusted_prob_legitimate * 100).toFixed(1)}% (Penalty: -${(notes.risk_penalties_applied * 100).toFixed(0)}%)</p>
        <p><em>${notes.refinement_notes}</em></p>
      `;
    } else {
      addBox.classList.add("hidden");
    }
  }

  // 6. SHAP Waterfall Bars
  renderShapBars(data.feature_explanations || []);

  // 7. Complete 34 Feature Grid
  renderFeatureGrid(data.raw_features || {});

  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Render SHAP Waterfall Horizontal Bars
function renderShapBars(explanations) {
  const container = document.getElementById("shap-bars");
  container.innerHTML = "";

  if (!explanations || explanations.length === 0) {
    container.innerHTML = `<p class="text-dim">No SHAP attributions available.</p>`;
    return;
  }

  // Find max impact for percentage scaling
  const maxImpact = Math.max(...explanations.map(e => e.abs_impact), 0.01);

  explanations.forEach(exp => {
    const isLegit = (exp.pushes_toward === "LEGITIMATE");
    const barWidth = Math.min(100, Math.max(10, (exp.abs_impact / maxImpact) * 100));

    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <div class="shap-feat-name" title="${exp.feature}">
        ${exp.feature.replace(/_/g, " ")}
      </div>
      <div class="shap-track">
        <div class="shap-bar ${isLegit ? 'pushes-legit' : 'pushes-phish'}" style="width: ${barWidth}%"></div>
      </div>
      <div class="shap-value-tag ${isLegit ? 'pushes-legit' : 'pushes-phish'}">
        ${isLegit ? "+" : ""}${exp.shap_value.toFixed(3)}
      </div>
    `;
    container.appendChild(row);
  });
}

// Render Feature Grid
function renderFeatureGrid(features) {
  const grid = document.getElementById("feature-grid");
  grid.innerHTML = "";

  Object.entries(features).forEach(([key, val]) => {
    const item = document.createElement("div");
    item.className = "feature-grid-item";

    let displayVal = val;
    if (typeof val === "number") {
      displayVal = Number.isInteger(val) ? val : val.toFixed(3);
    }

    item.innerHTML = `
      <span class="feature-grid-key">${key.replace(/_/g, " ")}</span>
      <span class="feature-grid-val">${displayVal}</span>
    `;
    grid.appendChild(item);
  });
}

function toggleFeatureGrid() {
  const content = document.getElementById("feature-grid-content");
  const icon = document.getElementById("feature-toggle-icon");
  if (content.classList.contains("hidden")) {
    content.classList.remove("hidden");
    icon.textContent = "▲";
  } else {
    content.classList.add("hidden");
    icon.textContent = "▼";
  }
}

// Load Model Metrics Hub
async function loadModelMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) return;
    const data = await res.json();
    const models = data.models_comparison;

    const tbody = document.getElementById("metrics-table-body");
    const cardsGrid = document.getElementById("models-cards-grid");

    if (!models || Object.keys(models).length === 0) return;

    tbody.innerHTML = "";
    cardsGrid.innerHTML = "";

    const winningModel = data.metadata?.model_name || "XGBoost";
    const pill = document.getElementById("selected-model-pill");
    if (pill) pill.textContent = `Winning Model: ${winningModel}`;

    Object.entries(models).forEach(([name, m]) => {
      const isWinner = (name === winningModel);

      // Table row
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong> ${isWinner ? '⭐' : ''}</td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.phishing_metrics.recall * 100).toFixed(2)}%</td>
        <td>${(m.phishing_metrics.f1_score * 100).toFixed(2)}%</td>
        <td>${(m.phishing_metrics.false_negative_rate * 100).toFixed(2)}%</td>
        <td>${(m.f1_macro * 100).toFixed(2)}%</td>
        <td>${m.roc_auc.toFixed(4)}</td>
        <td>${m.train_time_seconds || '<1'}s</td>
      `;
      tbody.appendChild(tr);

      // Card
      const card = document.createElement("div");
      card.className = "model-card glass-panel" + (isWinner ? " winner" : "");
      card.innerHTML = `
        <div class="model-card-header">
          <span class="model-card-title">${name}</span>
          <span class="model-badge">${isWinner ? "Selected Best Model" : "Evaluated"}</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">Accuracy:</span>
          <span class="model-stat-value">${(m.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">Phishing Detection Recall:</span>
          <span class="model-stat-value">${(m.phishing_metrics.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">Phishing F1-Score:</span>
          <span class="model-stat-value">${(m.phishing_metrics.f1_score * 100).toFixed(2)}%</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">False Negative Rate (Missed):</span>
          <span class="model-stat-value">${(m.phishing_metrics.false_negative_rate * 100).toFixed(2)}%</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">False Positive Rate (Alarms):</span>
          <span class="model-stat-value">${(m.phishing_metrics.false_positive_rate * 100).toFixed(2)}%</span>
        </div>
        <div class="model-stat-row">
          <span class="model-stat-label">ROC-AUC:</span>
          <span class="model-stat-value">${m.roc_auc.toFixed(4)}</span>
        </div>
      `;
      cardsGrid.appendChild(card);
    });

  } catch (e) {
    console.error("Failed to load model metrics:", e);
  }
}

// Load History
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) return;
    const data = await res.json();
    const history = data.history || [];

    const tbody = document.getElementById("history-table-body");
    if (history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center empty-state">No URLs scanned yet in this session.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    history.forEach((item, idx) => {
      const isPhish = (item.prediction === "PHISHING");
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${item.url}">${item.url}</td>
        <td style="color: ${isPhish ? 'var(--color-phish)' : 'var(--color-legit)'}; font-weight: 700;">${item.prediction}</td>
        <td>${item.confidence_percentage}</td>
        <td><span class="chip-dot" style="display:inline-block; margin-right:4px; background:${isPhish ? 'var(--color-phish)' : 'var(--color-legit)'};"></span>${item.risk_level}</td>
        <td>${item.latency_ms || 1.2}ms</td>
        <td>${item.timestamp}</td>
        <td><button class="small-btn" onclick="fillAndScan('${item.url}')">Re-scan</button></td>
      `;
      tbody.appendChild(tr);
    });

  } catch (e) {
    console.error("Failed to load history:", e);
  }
}

function clearHistory() {
  const tbody = document.getElementById("history-table-body");
  tbody.innerHTML = `<tr><td colspan="8" class="text-center empty-state">History cleared.</td></tr>`;
}
