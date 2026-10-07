// =====================================================================
// HYBRID AI ENGINE ENTERPRISE CONTROL CENTER JAVASCRIPT
// =====================================================================

let currentTenant = "fintech_corp";
let currentTenantStatus = null;

// Preset Payloads
const PRESETS = {
  fintech_corp: {
    cold: {
      account_age_days: 14,
      transaction_velocity_24h: 6500.0,
      failed_login_attempts: 1,
      risk_score_tier: "high"
    },
    ml: {
      account_age_days: 180,
      transaction_velocity_24h: 1200.0,
      failed_login_attempts: 0,
      risk_score_tier: "low"
    },
    nl: {
      natural_language_query: "What is the policy for new accounts under 30 days old?"
    }
  },
  ecommerce_inc: {
    cold: {
      days_since_last_purchase: 52,
      average_order_value: 140.0,
      support_tickets_opened: 3,
      loyalty_tier: "vip"
    },
    ml: {
      days_since_last_purchase: 12,
      average_order_value: 85.0,
      support_tickets_opened: 0,
      loyalty_tier: "standard"
    },
    nl: {
      natural_language_query: "What is the retention rule for VIP customers who haven't purchased in 45 days?"
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initTenantSelector();
  initTabs();
  initSandbox();
  initMLStudio();
  initRAGManager();
  initSchemaEditor();
});

// Toast notification helper
function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3500);
}

// 1. Health & Pinecone Check
async function initHealthCheck() {
  const badge = document.getElementById("pineconeStatusBadge");
  const text = document.getElementById("pineconeStatusText");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.pinecone_connected) {
      text.textContent = `Pinecone Live: Connected (${data.pinecone_index})`;
      badge.className = "status-pill badge-emerald";
    } else {
      text.textContent = `Pinecone: Serverless Index Active`;
      badge.className = "status-pill";
    }
  } catch (err) {
    text.textContent = "API Disconnected";
    badge.className = "status-pill badge-rose";
  }
}

// 2. Tenant Management
async function initTenantSelector() {
  const select = document.getElementById("tenantSelect");
  const refreshBtn = document.getElementById("refreshTenantBtn");

  select.addEventListener("change", (e) => {
    currentTenant = e.target.value;
    loadTenantData();
  });

  refreshBtn.addEventListener("click", () => {
    loadTenantData();
    showToast(`Refreshed tenant partition: ${currentTenant}`);
  });

  // Load initial tenant
  loadTenantData();
}

async function loadTenantData() {
  // Update labels
  document.getElementById("activeTenantBadge").textContent = currentTenant;
  document.getElementById("tenantPartitionId").textContent = currentTenant;
  document.getElementById("pineconeNamespaceName").textContent = `namespace: ${currentTenant}`;
  document.getElementById("vectorSearchNamespaceBadge").textContent = `namespace: ${currentTenant}`;

  try {
    const res = await fetch(`/api/v1/tenants/${currentTenant}/status`, {
      headers: { "X-Tenant-ID": currentTenant }
    });
    const result = await res.json();
    if (result.success) {
      currentTenantStatus = result.data;
      updateHeaderHealthCards(currentTenantStatus);
      loadDatasetPreview();
      loadModelMetrics();
      loadSchemaDefinition();
      loadPreset("cold");
    }
  } catch (err) {
    console.error("Failed to load tenant status:", err);
  }
}

function updateHeaderHealthCards(status) {
  // 1. Pinecone Vectors
  const vCount = status.pinecone ? status.pinecone.vector_count : 0;
  document.getElementById("pineconeVectorCount").textContent = `${vCount} vectors`;

  // 2. Model Status Card
  const modelText = document.getElementById("modelStatusText");
  const modelSub = document.getElementById("modelDetailsSubtext");
  const modelBadge = document.getElementById("modelStatusBadge");

  if (status.has_trained_model) {
    modelText.textContent = "Trained & Active";
    modelSub.textContent = `LightGBM Booster (Iter: ${status.model_metadata.best_iteration})`;
    modelBadge.className = "badge badge-emerald";
    modelBadge.textContent = "LightGBM Ready";
  } else {
    modelText.textContent = "Cold-Start Active";
    modelSub.textContent = "Routing fallback to Pinecone RAG";
    modelBadge.className = "badge badge-amber";
    modelBadge.textContent = "Cold-Start";
  }

  // 3. Tabular samples (updated after preview loads)
}

// 3. Tabs Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.getAttribute("data-tab"));
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// 4. Live Sandbox & Inference
function initSandbox() {
  const payloadInput = document.getElementById("queryPayloadInput");
  const btnPredict = document.getElementById("btnExecutePredict");

  document.getElementById("btnPresetCold").addEventListener("click", () => loadPreset("cold"));
  document.getElementById("btnPresetML").addEventListener("click", () => loadPreset("ml"));
  document.getElementById("btnPresetNL").addEventListener("click", () => loadPreset("nl"));

  btnPredict.addEventListener("click", async () => {
    let payload;
    try {
      payload = JSON.parse(payloadInput.value);
    } catch (err) {
      showToast("Invalid JSON syntax in payload editor.");
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = "<span>⏳</span> Evaluating...";

    try {
      const res = await fetch("/api/v1/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": currentTenant
        },
        body: JSON.stringify({ payload })
      });
      const data = await res.json();
      displayInferenceResult(data);
    } catch (err) {
      showToast(`Inference error: ${err.message}`);
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '<span class="btn-icon-play">▶</span> Execute Hybrid Inference';
    }
  });
}

function loadPreset(type) {
  const tenantPresets = PRESETS[currentTenant] || PRESETS.fintech_corp;
  const p = tenantPresets[type] || {};
  document.getElementById("queryPayloadInput").value = JSON.stringify(p, null, 2);
}

function displayInferenceResult(res) {
  const rawPre = document.getElementById("rawResponseOutput");
  rawPre.textContent = JSON.stringify(res, null, 2);

  const payload = res.data || {};
  const route = payload.route || "unknown";
  const latency = payload.latency_ms || 0;

  // Update Route Badge
  const routeBadge = document.getElementById("responseRouteBadge");
  routeBadge.textContent = `Route: ${route}`;
  if (route === "classical_ml") routeBadge.className = "badge badge-emerald font-mono";
  else if (route === "cold_start_rag") routeBadge.className = "badge badge-amber font-mono";
  else routeBadge.className = "badge badge-indigo font-mono";

  document.getElementById("responseLatencyText").textContent = `Latency: ${latency} ms`;
  document.getElementById("nodeLatency").textContent = `${latency} ms`;

  // Animate Topology
  animateRoutingTopology(route);

  // Update Summary Card
  const title = document.getElementById("summaryTitle");
  const conf = document.getElementById("summaryConfidence");
  const body = document.getElementById("summaryBody");

  if (route === "classical_ml") {
    title.textContent = `Classical Tabular ML | Prediction: ${payload.prediction.toFixed(4)}`;
    conf.textContent = `${(payload.confidence * 100).toFixed(1)}% Confidence`;
    body.textContent = (
      `Label: ${payload.label === 1 ? 'Positive / Flagged' : 'Normal / Approved'}\n` +
      `Model: ${payload.model_type} Booster\n` +
      `Features Evaluated: ${payload.features_evaluated ? payload.features_evaluated.join(', ') : 'All schema fields'}\n` +
      (payload.drift_detected ? `⚠️ Feature Drift Detected: ${JSON.stringify(payload.drift_details)}` : `✓ No statistical feature drift detected.`)
    );
  } else {
    // RAG Route
    title.textContent = route === "cold_start_rag" ? "Cold-Start Fallback (Pinecone + LLM)" : "Unstructured Query Retrieval (Pinecone + LLM)";
    conf.textContent = `${(payload.confidence * 100).toFixed(1)}% Match Score`;
    body.textContent = payload.answer || "No synthesis text available.";
  }
}

function animateRoutingTopology(activeRoute) {
  // Reset nodes
  const allNodes = document.querySelectorAll(".pipeline-node");
  allNodes.forEach((n) => n.className = "pipeline-node");

  document.getElementById("nodeInput").classList.add("active-path");
  document.getElementById("nodeRouter").classList.add("active-path");
  document.getElementById("nodeOutput").classList.add("active-path");

  const statusText = document.getElementById("routeActiveStatus");

  if (activeRoute === "classical_ml") {
    document.getElementById("nodeML").classList.add("active-ml");
    statusText.textContent = "Route: Classical Tabular ML (LightGBM)";
    statusText.style.color = "#10b981";
  } else if (activeRoute === "cold_start_rag") {
    document.getElementById("nodeCold").classList.add("active-cold");
    statusText.textContent = "Route: Cold-Start Fallback (Pinecone Namespace)";
    statusText.style.color = "#f59e0b";
  } else {
    document.getElementById("nodeNL").classList.add("active-path");
    statusText.textContent = "Route: Unstructured Vector RAG (Pinecone Namespace)";
    statusText.style.color = "#818cf8";
  }
}

// 5. Tabular Dataset & ML Studio
async function loadDatasetPreview() {
  const tableWrapper = document.getElementById("datasetTableWrapper");
  const countEl = document.getElementById("dataSampleCount");
  const badgeEl = document.getElementById("dataThresholdBadge");
  const progressFill = document.getElementById("sampleProgressFill");
  const totalText = document.getElementById("datasetRowsTotal");

  try {
    const res = await fetch("/api/v1/data/preview?limit=8", {
      headers: { "X-Tenant-ID": currentTenant }
    });
    const result = await res.json();
    const data = result.data || {};
    const total = data.total_records || 0;

    countEl.textContent = `${total} samples`;
    totalText.textContent = `${total} total rows in dataset`;

    const pct = Math.min(100, Math.round((total / 100) * 100));
    progressFill.style.width = `${pct}%`;

    if (total >= 100) {
      badgeEl.className = "badge badge-emerald";
      badgeEl.textContent = "Ready (>= 100)";
    } else {
      badgeEl.className = "badge badge-amber";
      badgeEl.textContent = `Cold-Start (< 100)`;
    }

    // Render table
    const columns = data.columns || [];
    const preview = data.preview || [];

    if (preview.length === 0) {
      tableWrapper.innerHTML = '<div class="empty-state"><div class="empty-desc">No tabular dataset ingested for this tenant.</div></div>';
      return;
    }

    let html = '<table class="data-table"><thead><tr>';
    columns.forEach((c) => html += `<th>${c}</th>`);
    html += '</tr></thead><tbody>';

    preview.forEach((row) => {
      html += '<tr>';
      columns.forEach((c) => html += `<td>${row[c] !== undefined ? row[c] : ''}</td>`);
      html += '</tr>';
    });
    html += '</tbody></table>';
    tableWrapper.innerHTML = html;

  } catch (err) {
    console.error("Failed to load dataset preview:", err);
  }
}

async function loadModelMetrics() {
  const container = document.getElementById("mlMetricsContainer");
  const placeholder = document.getElementById("noModelPlaceholder");
  const statusBadge = document.getElementById("mlMetricsStatusBadge");

  try {
    const res = await fetch("/api/v1/models/metrics", {
      headers: { "X-Tenant-ID": currentTenant }
    });
    const result = await res.json();
    if (!result.success || !result.data) {
      container.style.display = "none";
      placeholder.style.display = "block";
      statusBadge.textContent = "Not Trained";
      statusBadge.className = "badge badge-amber";
      return;
    }

    const meta = result.data;
    container.style.display = "block";
    placeholder.style.display = "none";
    statusBadge.textContent = "LightGBM Active";
    statusBadge.className = "badge badge-emerald";

    const metrics = meta.metrics || {};
    document.getElementById("metricAUC").textContent = metrics.auc !== undefined ? metrics.auc : (metrics.rmse !== undefined ? metrics.rmse : "--");
    document.getElementById("metricAccuracy").textContent = metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}%` : (metrics.r2 !== undefined ? metrics.r2 : "--");
    document.getElementById("metricBestIter").textContent = meta.best_iteration || "--";
    document.getElementById("metricSamples").textContent = meta.num_samples || "--";

    // Feature Importances
    const barsContainer = document.getElementById("featureImportanceBars");
    barsContainer.innerHTML = "";
    const importances = meta.feature_importances || {};
    for (const [feat, val] of Object.entries(importances)) {
      const pct = Math.round(val * 100);
      barsContainer.innerHTML += `
        <div class="bar-row">
          <span class="bar-label" title="${feat}">${feat}</span>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct}%"></div></div>
          <span class="bar-val font-mono">${pct}%</span>
        </div>
      `;
    }
  } catch (err) {
    container.style.display = "none";
    placeholder.style.display = "block";
  }
}

function initMLStudio() {
  const btnTrain = document.getElementById("btnTrainModel");
  btnTrain.addEventListener("click", async () => {
    const targetCol = document.getElementById("targetColInput").value.trim();
    const taskType = document.getElementById("taskTypeSelect").value;
    const tuneOptuna = document.getElementById("chkOptuna").checked;

    btnTrain.disabled = true;
    btnTrain.innerHTML = "<span>⏳</span> Training LightGBM + Optuna Tuning...";

    try {
      const res = await fetch("/api/v1/models/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": currentTenant
        },
        body: JSON.stringify({
          target_column: targetCol,
          task_type: taskType,
          tune_hyperparameters: tuneOptuna
        })
      });
      const data = await res.json();
      if (data.success) {
        showToast("Model trained and persisted successfully!");
        loadTenantData();
      } else {
        showToast(`Training error: ${data.detail || data.message}`);
      }
    } catch (err) {
      showToast(`Training failed: ${err.message}`);
    } finally {
      btnTrain.disabled = false;
      btnTrain.innerHTML = '<span>⚡</span> Run Automated Tabular ML Training';
    }
  });
}

// 6. Pinecone RAG Manager
function initRAGManager() {
  const btnIndex = document.getElementById("btnIndexDoc");
  btnIndex.addEventListener("click", async () => {
    const docId = document.getElementById("docIdInput").value.trim();
    const category = document.getElementById("docCategoryInput").value.trim();
    const text = document.getElementById("docTextInput").value.trim();

    if (!docId || !text) {
      showToast("Please provide both Document ID and content.");
      return;
    }

    btnIndex.disabled = true;
    btnIndex.innerHTML = "<span>⏳</span> Indexing into Pinecone...";

    try {
      const res = await fetch("/api/v1/rag/upsert", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": currentTenant
        },
        body: JSON.stringify({
          doc_id: docId,
          text: text,
          metadata: { category: category || "general" }
        })
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Indexed '${docId}' into namespace [${currentTenant}]!`);
        document.getElementById("docIdInput").value = "";
        document.getElementById("docTextInput").value = "";
        loadTenantData();
      } else {
        showToast(`Pinecone indexing error: ${data.detail || data.message}`);
      }
    } catch (err) {
      showToast(`Indexing failed: ${err.message}`);
    } finally {
      btnIndex.disabled = false;
      btnIndex.innerHTML = '<span>🌲</span> Index into Pinecone Namespace';
    }
  });

  const btnSearch = document.getElementById("btnVectorSearch");
  const searchInput = document.getElementById("vectorSearchInput");

  btnSearch.addEventListener("click", async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    btnSearch.disabled = true;
    try {
      const res = await fetch(`/api/v1/rag/search?query=${encodeURIComponent(query)}&top_k=3`, {
        method: "POST",
        headers: { "X-Tenant-ID": currentTenant }
      });
      const data = await res.json();
      renderVectorResults(data.data ? data.data.matches : []);
    } catch (err) {
      showToast(`Vector query failed: ${err.message}`);
    } finally {
      btnSearch.disabled = false;
    }
  });
}

function renderVectorResults(matches) {
  const container = document.getElementById("vectorResultsList");
  if (!matches || matches.length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="empty-desc">No matching vectors found in this tenant namespace.</div></div>';
    return;
  }

  let html = "";
  matches.forEach((m) => {
    const scorePct = Math.round(m.score * 100);
    const text = m.metadata ? (m.metadata.text || JSON.stringify(m.metadata)) : "Doc Vector";
    const category = m.metadata ? (m.metadata.category || "policy") : "";
    html += `
      <div class="vector-result-card">
        <div class="vector-result-top">
          <span class="font-mono text-muted">ID: ${m.id} | Category: ${category}</span>
          <span class="badge badge-indigo font-mono">${scorePct}% Cosine Sim</span>
        </div>
        <div class="vector-result-text">${text}</div>
      </div>
    `;
  });
  container.innerHTML = html;
}

// 7. Dynamic Schema Editor
async function loadSchemaDefinition() {
  const editor = document.getElementById("schemaDefinitionEditor");
  try {
    const res = await fetch("/api/v1/schemas", {
      headers: { "X-Tenant-ID": currentTenant }
    });
    const data = await res.json();
    if (data.success && data.data) {
      editor.value = JSON.stringify(data.data.schema, null, 2);
    }
  } catch (err) {
    editor.value = "{}";
  }
}

function initSchemaEditor() {
  const btnSave = document.getElementById("btnSaveSchema");
  const editor = document.getElementById("schemaDefinitionEditor");

  btnSave.addEventListener("click", async () => {
    let schemaObj;
    try {
      schemaObj = JSON.parse(editor.value);
    } catch (err) {
      showToast("Invalid JSON syntax in schema editor.");
      return;
    }

    try {
      const res = await fetch("/api/v1/schemas", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": currentTenant
        },
        body: JSON.stringify(schemaObj)
      });
      const data = await res.json();
      if (data.success) {
        showToast(`Schema updated successfully for ${currentTenant}!`);
        loadTenantData();
      } else {
        showToast(`Schema update error: ${data.detail || data.message}`);
      }
    } catch (err) {
      showToast(`Failed to update schema: ${err.message}`);
    }
  });
}
