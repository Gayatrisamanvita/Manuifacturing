/**
 * PrecisionPulse AI - Frontend Application Controller
 * Manages Chart.js visualizations, Factory Digital Twin interactivity,
 * operational presets, real-time SHAP waterfall, and 8D SCAR dossier rendering.
 */

// Global State
let currentVendorSCAR = "VENDOR_B";
let activeStationId = "ST-20";
let chartStationRisk = null;
let chartSupplierTrends = null;
let chartToleranceLeakage = null;
let supplierCatalog = null;
let selectedMaterialRecords = [];
let finishedProductSearchTimer = null;

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  console.log("PrecisionPulse AI Suite initialized.");
  initCharts();
  openSCAR("VENDOR_B");
  triggerSimulation();
  loadSupplierAnalysis();
  loadMaterialAnalysis();
  loadProductionAnalysis();
  document.getElementById("finished-product-search")?.addEventListener("keydown", event => {
    if (event.key === "Enter") searchFinishedProducts();
  });
  document.getElementById("supplier-products-table")?.addEventListener("click", event => {
    const row = event.target.closest("[data-sku]");
    if (row) selectSupplierProduct(row.dataset.sku);
  });
  document.getElementById("supplier-products-table")?.addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("[data-sku]");
    if (row) {
      event.preventDefault();
      selectSupplierProduct(row.dataset.sku);
    }
  });
  document.getElementById("supplier-risk-products")?.addEventListener("click", event => {
    const item = event.target.closest("[data-sku]");
    if (item) selectSupplierProduct(item.dataset.sku);
  });
  document.getElementById("all-materials-table")?.addEventListener("click", event => {
    const row = event.target.closest("[data-lot]");
    if (row) selectMaterialLot(row.dataset.lot);
  });
  document.getElementById("all-materials-table")?.addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("[data-lot]");
    if (row) {
      event.preventDefault();
      selectMaterialLot(row.dataset.lot);
    }
  });
  document.getElementById("high-risk-materials")?.addEventListener("click", event => {
    const item = event.target.closest("[data-lot]");
    if (item) selectMaterialLot(item.dataset.lot);
  });
});

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[character]);
}

function showLoadError(elementId, error) {
  const element = document.getElementById(elementId);
  if (element) element.innerHTML = `<p class="error-state">${escapeHTML(error.message || "Unable to load analysis.")}</p>`;
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (error) {
      console.error(`Could not read error response for ${url}`, error);
    }
    throw new Error(detail);
  }
  return response.json();
}

function riskBadge(level) {
  const normalized = String(level || "LOW").toUpperCase();
  return `<span class="risk-badge risk-${escapeHTML(normalized.toLowerCase())}">${escapeHTML(normalized)} RISK</span>`;
}

function renderSupplierSummaryTable(suppliers) {
  const body = document.getElementById("overall-supplier-table");
  body.innerHTML = suppliers.map(supplier => `
    <tr>
      <td><strong>${escapeHTML(supplier.supplier_id)}</strong></td>
      <td>${Number(supplier.current_defect_rate_pct).toFixed(2)}%</td>
      <td>${escapeHTML(supplier.top_recurring_failure_mode)}</td>
      <td>${supplier.capa_recurrence_count}</td>
      <td>${supplier.high_risk_products} / ${supplier.product_count}</td>
      <td>${supplier.average_delay_days.toFixed(1)} days</td>
      <td>${supplier.warranty_claim_count.toLocaleString()}</td>
      <td>$${supplier.total_financial_impact_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td>${escapeHTML(supplier.worst_product_name)} <span class="text-rose">(+${supplier.worst_product_delay_days}d)</span></td>
    </tr>
  `).join("");
}

async function loadSupplierAnalysis() {
  try {
    supplierCatalog = await fetchJSON("/api/tier3/catalog");
    const supplierSelect = document.getElementById("supplier-select");
    supplierSelect.innerHTML = supplierCatalog.suppliers.map(supplier =>
      `<option value="${escapeHTML(supplier.supplier_id)}">${escapeHTML(supplier.supplier_id.replace(/_/g, " "))} · ${supplier.product_count} products</option>`
    ).join("");
    renderSupplierSummaryTable(supplierCatalog.suppliers);
    if (supplierCatalog.suppliers.length) selectSupplier(supplierSelect.value);
  } catch (error) {
    console.error("Supplier catalog load failed", error);
    showLoadError("supplier-risk-products", error);
    showLoadError("overall-supplier-table", error);
  }
}

function selectSupplier(supplierId) {
  if (!supplierCatalog) return;
  const supplier = supplierCatalog.suppliers.find(item => item.supplier_id === supplierId);
  const products = supplierCatalog.products_by_supplier[supplierId] || [];
  const highRiskProducts = products.filter(product => product.risk_level !== "LOW").slice(0, 5);
  const riskList = document.getElementById("supplier-risk-products");

  if (!supplier) {
    riskList.innerHTML = '<p class="empty-state">No product data is available for this supplier.</p>';
    return;
  }

  riskList.innerHTML = `
    <div class="summary-metric-grid">
      <div class="summary-metric"><span>Products supplied</span><strong>${supplier.product_count}</strong></div>
      <div class="summary-metric"><span>At-risk products</span><strong class="text-rose">${supplier.high_risk_products}</strong></div>
      <div class="summary-metric"><span>Average delivery delay</span><strong>${supplier.average_delay_days.toFixed(1)} days</strong></div>
      <div class="summary-metric"><span>Average product quality</span><strong>${supplier.average_quality_score_pct.toFixed(1)}%</strong></div>
      <div class="summary-metric"><span>Current defect rate</span><strong>${Number(supplier.current_defect_rate_pct).toFixed(2)}%</strong></div>
      <div class="summary-metric"><span>Warranty claims</span><strong>${supplier.warranty_claim_count.toLocaleString()}</strong></div>
      <div class="summary-metric"><span>Financial impact</span><strong>$${supplier.total_financial_impact_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong></div>
      <div class="summary-metric"><span>Top recurring defect</span><strong>${escapeHTML(supplier.top_recurring_failure_mode)}</strong></div>
    </div>
    <h3 class="analysis-subheading">Most delayed and highest-risk products</h3>
    ${highRiskProducts.length ? `<div class="risk-product-list">${highRiskProducts.map(product => `
      <button type="button" class="risk-product-item" data-sku="${escapeHTML(product.sku)}">
        <span>${riskBadge(product.risk_level)}<strong>${escapeHTML(product.product_name)}</strong><small>${escapeHTML(product.sku)} · ${escapeHTML(product.status)}</small></span>
        <span class="delay-value">+${product.delay_days}d</span>
      </button>
    `).join("")}</div>` : '<p class="empty-state">No delayed products are currently listed for this supplier.</p>'}
    <button class="btn-primary supplier-scar-button" type="button" onclick="openSCAR('${escapeHTML(supplierId)}')">Generate ${escapeHTML(supplierId)} SCAR 8D</button>
  `;

  const productSelect = document.getElementById("supplier-product-select");
  productSelect.innerHTML = products.map(product =>
    `<option value="${escapeHTML(product.sku)}">${escapeHTML(product.sku)} · ${escapeHTML(product.product_name)}</option>`
  ).join("");

  document.getElementById("supplier-products-table").innerHTML = products.map(product => `
    <tr data-sku="${escapeHTML(product.sku)}" tabindex="0" role="button" aria-label="Analyze ${escapeHTML(product.product_name)}">
      <td>${riskBadge(product.risk_level)}</td>
      <td><strong>${escapeHTML(product.product_name)}</strong><br><span class="text-muted">${escapeHTML(product.sku)}</span></td>
      <td>${escapeHTML(product.category)}</td>
      <td>${product.target_lead_days} days</td>
      <td>${product.actual_lead_days} days</td>
      <td class="${product.delay_days >= 7 ? "text-rose" : ""}">+${product.delay_days} days</td>
      <td>${product.quality_score_pct.toFixed(1)}%</td>
    </tr>
  `).join("");

  if (products.length) selectSupplierProduct(products[0].sku);
}

function selectSupplierProduct(sku) {
  if (!supplierCatalog || !sku) return;
  const supplierId = document.getElementById("supplier-select").value;
  const product = (supplierCatalog.products_by_supplier[supplierId] || []).find(item => item.sku === sku);
  if (!product) return;
  document.getElementById("supplier-product-select").value = sku;
  const riskExplanation = product.risk_level === "HIGH"
    ? "High delivery risk: the observed delay is at least 7 days and may affect production readiness."
    : product.risk_level === "MODERATE"
      ? "Moderate delivery risk: build the observed delay into purchasing and production planning."
      : "Low delivery risk: this product is currently on schedule or has only a short delay.";

  document.getElementById("supplier-product-analysis").innerHTML = `
    <div class="product-analysis-heading">
      <div>${riskBadge(product.risk_level)}<h3>${escapeHTML(product.product_name)}</h3><p>${escapeHTML(product.sku)} · ${escapeHTML(product.category)}</p></div>
      <div class="product-delay-callout"><span>Delivery delay</span><strong>+${product.delay_days} days</strong><small>${escapeHTML(product.status)}</small></div>
    </div>
    <div class="summary-metric-grid">
      <div class="summary-metric"><span>Target lead time</span><strong>${product.target_lead_days} days</strong></div>
      <div class="summary-metric"><span>Actual lead time</span><strong>${product.actual_lead_days} days</strong></div>
      <div class="summary-metric"><span>Quality score</span><strong>${product.quality_score_pct.toFixed(1)}%</strong></div>
      <div class="summary-metric"><span>Monthly volume</span><strong>${product.monthly_volume_units.toLocaleString()}</strong></div>
    </div>
    <p class="analysis-note">${riskExplanation}</p>
    <div class="analysis-copy-grid">
      <div><h4>Quality / defect concern</h4><p>${escapeHTML(product.defect_mode)}</p></div>
      <div><h4>Likely root cause</h4><p>${escapeHTML(product.root_cause)}</p></div>
      <div class="analysis-recommendation"><h4>Recommended action</h4><p>${escapeHTML(product.recommendations)}</p></div>
    </div>
  `;
}

async function loadMaterialAnalysis() {
  try {
    const result = await fetchJSON("/api/tier2/materials");
    selectedMaterialRecords = result.materials;
    document.getElementById("material-batch-count").innerText =
      `${result.batch_count} lots · ${result.high_risk_count} high risk`;

    document.getElementById("high-risk-materials").innerHTML = result.materials
      .filter(material => material.risk_level === "HIGH")
      .slice(0, 5)
      .map(material => `
        <button type="button" class="risk-product-item" data-lot="${escapeHTML(material.material_batch_id)}">
          <span>${riskBadge(material.risk_level)}<strong>${escapeHTML(material.material_batch_id)}</strong><small>${escapeHTML(material.supplier_id)} · ${escapeHTML(material.material_risk_status)}</small></span>
          <span class="delay-value">${Number(material.material_hardness_hv).toFixed(1)} HV</span>
        </button>
      `).join("") || '<p class="empty-state">No high-risk lots found in the material batch records.</p>';

    const lotSelect = document.getElementById("material-lot-select");
    lotSelect.innerHTML = result.materials.map(material =>
      `<option value="${escapeHTML(material.material_batch_id)}">${escapeHTML(material.material_batch_id)} · ${escapeHTML(material.supplier_id)} · ${escapeHTML(material.risk_level)} RISK</option>`
    ).join("");
    document.getElementById("all-materials-table").innerHTML = result.materials.map(material => `
      <tr data-lot="${escapeHTML(material.material_batch_id)}" tabindex="0" role="button" aria-label="Analyze raw material lot ${escapeHTML(material.material_batch_id)}">
        <td>${riskBadge(material.risk_level)}</td>
        <td><strong>${escapeHTML(material.material_batch_id)}</strong></td>
        <td>${escapeHTML(material.supplier_id)}</td>
        <td>${Number(material.material_hardness_hv).toFixed(1)} HV</td>
        <td>${escapeHTML(material.material_risk_status)}</td>
        <td>${Number(material.scrap_count).toLocaleString()}</td>
        <td>$${Number(material.chargeback_amount_usd).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
      </tr>
    `).join("");
    if (result.materials.length) selectMaterialLot(lotSelect.value);
  } catch (error) {
    console.error("Material analysis load failed", error);
    showLoadError("high-risk-materials", error);
    showLoadError("all-materials-table", error);
  }
}

function selectMaterialLot(lotId) {
  if (!lotId) return;
  const material = selectedMaterialRecords.find(item => item.material_batch_id === lotId);
  if (!material) return;
  document.getElementById("material-lot-select").value = lotId;
  document.getElementById("material-lot-details").innerHTML = `
    <div class="product-analysis-heading">
      <div>${riskBadge(material.risk_level)}<h3>${escapeHTML(material.material_batch_id)}</h3><p>${escapeHTML(material.supplier_id)} · Snapshot ${escapeHTML(material.snapshot_date)}</p></div>
      <div class="product-delay-callout"><span>Risk score</span><strong>${Number(material.risk_score).toFixed(0)} / 100</strong><small>${escapeHTML(material.material_risk_status)}</small></div>
    </div>
    <div class="summary-metric-grid">
      <div class="summary-metric"><span>Hardness</span><strong>${Number(material.material_hardness_hv).toFixed(1)} HV</strong></div>
      <div class="summary-metric"><span>Tensile strength</span><strong>${Number(material.tensile_strength_mpa).toFixed(1)} MPa</strong></div>
      <div class="summary-metric"><span>Specification</span><strong>${Number(material.lower_spec_limit).toFixed(0)}–${Number(material.upper_spec_limit).toFixed(0)} HV</strong></div>
      <div class="summary-metric"><span>Recorded scrap</span><strong>${Number(material.scrap_count).toLocaleString()} units</strong></div>
      <div class="summary-metric"><span>Chargeback</span><strong>$${Number(material.chargeback_amount_usd).toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></div>
    </div>
    <p class="analysis-note">${material.risk_level === "HIGH" ? "This lot is flagged as high risk. Review its receiving measurements and run the downstream impact evaluation below before release." : "Use the receiving measurements and downstream impact evaluation below to confirm this lot is ready for production."}</p>
  `;
  const hardnessInput = document.getElementById("slider-mat-hardness");
  const tensileInput = document.getElementById("slider-mat-tensile");
  hardnessInput.value = Math.max(Number(hardnessInput.min), Math.min(Number(hardnessInput.max), Number(material.material_hardness_hv)));
  tensileInput.value = Math.max(Number(tensileInput.min), Math.min(Number(tensileInput.max), Number(material.tensile_strength_mpa)));
  document.getElementById("sel-mat-supplier").value = material.supplier_id;
  updateMaterialEvaluator();
}

async function loadProductionAnalysis(search = "") {
  try {
    const [summary, productList] = await Promise.all([
      fetchJSON("/api/tier1/production/summary"),
      fetchJSON(`/api/tier1/finished-products${search ? `?search=${encodeURIComponent(search)}` : ""}`)
    ]);
    document.getElementById("production-metrics").innerHTML = `
      <div class="summary-metric"><span>Recorded units produced</span><strong>${summary.total_units_produced.toLocaleString()}</strong></div>
      <div class="summary-metric"><span>Recorded defects</span><strong>${summary.total_defects.toLocaleString()}</strong></div>
      <div class="summary-metric"><span>Recorded defect rate</span><strong>${summary.defect_rate_pct.toFixed(2)}%</strong></div>
      <div class="summary-metric"><span>Average line speed</span><strong>${summary.average_line_speed_upm.toFixed(1)} units/min</strong></div>
    `;
    document.getElementById("production-stations").innerHTML = summary.stations.map(station => `
      <div class="production-station-card">
        <strong>${escapeHTML(station.station_id)}</strong>
        <span>${Number(station.units_produced).toLocaleString()} units · ${Number(station.defect_count).toLocaleString()} defects</span>
        <small>${Number(station.average_line_speed_upm).toFixed(1)} units/min average</small>
      </div>
    `).join("");
    const select = document.getElementById("finished-product-select");
    select.innerHTML = productList.products.length
      ? productList.products.map(product => `<option value="${escapeHTML(product.part_serial_no)}">${escapeHTML(product.part_serial_no)} · ${product.defect_flag ? "Inspection defect" : "Inspection pass"} · ${escapeHTML(product.material_batch_id)}</option>`).join("")
      : '<option value="">No matching finished products found</option>';
    if (productList.products.length) await loadFinishedProductDetails(select.value);
    else document.getElementById("finished-product-details").innerHTML = '<p class="empty-state">No finished product serials match that search. Try another part number.</p>';
  } catch (error) {
    console.error("Production analysis load failed", error);
    showLoadError("production-metrics", error);
    showLoadError("finished-product-details", error);
  }
}

function searchFinishedProducts() {
  clearTimeout(finishedProductSearchTimer);
  finishedProductSearchTimer = setTimeout(() => {
    const query = document.getElementById("finished-product-search").value.trim();
    loadProductionAnalysis(query);
  }, 150);
}

async function loadFinishedProductDetails(serial) {
  if (!serial) return;
  try {
    const product = await fetchJSON(`/api/tier1/finished-products/${encodeURIComponent(serial)}`);
    const isDefect = Number(product.defect_flag) === 1;
    document.getElementById("finished-product-details").innerHTML = `
      <div class="product-analysis-heading">
        <div>${riskBadge(isDefect ? "HIGH" : product.material_risk_status === "MARGINAL_HIGH" ? "MODERATE" : "LOW")}<h3>${escapeHTML(product.part_serial_no)}</h3><p>Finished part · ${escapeHTML(product.fastener_cycle_id)}</p></div>
        <div class="product-delay-callout"><span>Inspection result</span><strong>${isDefect ? "Defect detected" : "Pass"}</strong><small>${escapeHTML(product.failure_mode)}</small></div>
      </div>
      <div class="summary-metric-grid">
        <div class="summary-metric"><span>Defect severity</span><strong>${Number(product.defect_severity)} / 5</strong></div>
        <div class="summary-metric"><span>Manufacturing station</span><strong>${escapeHTML(product.station_id)}</strong></div>
        <div class="summary-metric"><span>Material lot</span><strong>${escapeHTML(product.material_batch_id)}</strong></div>
        <div class="summary-metric"><span>Supplier</span><strong>${escapeHTML(product.supplier_id)}</strong></div>
        <div class="summary-metric"><span>Material hardness</span><strong>${Number(product.material_hardness_hv).toFixed(1)} HV</strong></div>
        <div class="summary-metric"><span>Production shift</span><strong>${escapeHTML(product.shift_id)}</strong></div>
        <div class="summary-metric"><span>Line speed</span><strong>${Number(product.line_speed_upm).toFixed(1)} units/min</strong></div>
        <div class="summary-metric"><span>Torque status</span><strong>${escapeHTML(product.torque_status)}</strong></div>
      </div>
      <div class="analysis-copy-grid">
        <div><h4>Inspection action</h4><p>${escapeHTML(product.action_code)}</p></div>
        <div><h4>Process telemetry</h4><p>${Number(product.peak_torque_nm).toFixed(2)} Nm torque · ${Number(product.shutoff_angle_deg).toFixed(1)}° shutoff angle</p></div>
      </div>
    `;
  } catch (error) {
    console.error("Finished product detail load failed", error);
    showLoadError("finished-product-details", error);
  }
}

// Tab Switcher
function switchTab(tabId) {
  const buttons = document.querySelectorAll(".nav-tab-btn");
  buttons.forEach(btn => {
    btn.classList.remove("active");
    btn.removeAttribute("aria-current");
  });
  
  const targetBtn = {
    "tab-overview": "tab-btn-overview",
    "tab-tier1": "tab-btn-tier1",
    "tab-tier2": "tab-btn-tier2",
    "tab-tier3": "tab-btn-tier3",
  }[tabId];

  if (targetBtn && document.getElementById(targetBtn)) {
    const activeButton = document.getElementById(targetBtn);
    activeButton.classList.add("active");
    activeButton.setAttribute("aria-current", "page");
  }

  const contents = document.querySelectorAll(".tab-content");
  contents.forEach(content => content.classList.remove("active"));

  const targetContent = document.getElementById(tabId);
  if (targetContent) {
    targetContent.classList.add("active");
  }

  // Trigger chart resize if newly visible
  setTimeout(() => {
    if (chartStationRisk) chartStationRisk.resize();
    if (chartSupplierTrends) chartSupplierTrends.resize();
    if (chartToleranceLeakage) chartToleranceLeakage.resize();
  }, 100);
}

// -------------------------------------------------------------
// Factory Digital Twin Station Selection
// -------------------------------------------------------------
function selectStation(stId) {
  activeStationId = stId;
  const nodes = document.querySelectorAll(".station-node");
  nodes.forEach(n => n.classList.remove("active-station"));
  
  const selectedNode = document.getElementById(`node-${stId}`);
  if (selectedNode) selectedNode.classList.add("active-station");

  const selElem = document.getElementById("sel-station");
  if (selElem) selElem.value = stId;

  // Set telemetry presets matching station characteristics
  if (stId === "ST-20" || stId === "ST-30") {
    document.getElementById("slider-peak-torque").value = 13.4;
    document.getElementById("slider-shutoff-angle").value = 268.0;
  } else {
    document.getElementById("slider-peak-torque").value = 12.5;
    document.getElementById("slider-shutoff-angle").value = 358.0;
  }
  updateSimulator();
}

// -------------------------------------------------------------
// Operational Presets
// -------------------------------------------------------------
function applyPreset(scenario) {
  switchTab("tab-tier1");

  if (scenario === "normal") {
    setSliderValues(55.0, 12.5, 360.0, 1.2, 30.0, 115.0);
  } else if (scenario === "fastening") {
    setSliderValues(64.5, 13.4, 268.0, 2.1, 52.0, 115.0);
  } else if (scenario === "surface") {
    setSliderValues(68.0, 12.5, 358.0, 2.8, 72.0, 115.0);
  } else if (scenario === "wear") {
    setSliderValues(58.0, 11.2, 360.0, 3.4, 88.0, 115.0);
  } else if (scenario === "hydraulic") {
    setSliderValues(78.5, 12.1, 293.0, 2.0, 50.0, 78.0);
  }
}

function setSliderValues(temp, torque, angle, vib, wear, press) {
  document.getElementById("slider-spindle-temp").value = temp;
  document.getElementById("slider-peak-torque").value = torque;
  document.getElementById("slider-shutoff-angle").value = angle;
  document.getElementById("slider-vibration").value = vib;
  document.getElementById("slider-tool-wear").value = wear;
  document.getElementById("slider-pressure").value = press;
  updateSimulator();
}

// -------------------------------------------------------------
// Chart.js Visualizations Setup
// -------------------------------------------------------------
function initCharts() {
  // 1. Station Defect Risk Bar Chart
  const ctxStation = document.getElementById("chart-station-risk");
  if (ctxStation && !chartStationRisk) {
    chartStationRisk = new Chart(ctxStation, {
      type: "bar",
      data: {
        labels: ["ST-10 (Pre-Fasten)", "ST-20 (High Torque)", "ST-30 (Torque Insp)", "ST-40 (Final QC)"],
        datasets: [{
          label: "Predicted Defective Parts / Shift",
          data: [63.4, 64.4, 64.2, 63.4],
          backgroundColor: [
            "rgba(245, 158, 11, 0.65)",
            "rgba(244, 63, 94, 0.85)",
            "rgba(244, 63, 94, 0.8)",
            "rgba(16, 185, 129, 0.65)"
          ],
          borderColor: [
            "#f59e0b",
            "#f43f5e",
            "#f43f5e",
            "#10b981"
          ],
          borderWidth: 2,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            titleColor: "#ffffff",
            bodyColor: "#cbd5e1",
            borderColor: "rgba(255, 255, 255, 0.1)",
            borderWidth: 1
          }
        },
        scales: {
          y: {
            min: 50,
            max: 70,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8", font: { family: "Inter" } }
          },
          x: {
            grid: { display: false },
            ticks: { color: "#cbd5e1", font: { family: "Inter", weight: "500" } }
          }
        }
      }
    });
  }

  // 2. 30-Day Supplier Defect Rate Trends
  const ctxTrends = document.getElementById("chart-supplier-trends");
  if (ctxTrends && !chartSupplierTrends) {
    chartSupplierTrends = new Chart(ctxTrends, {
      type: "line",
      data: {
        labels: ["Mar 25", "May 25", "Jul 25", "Sep 25", "Nov 25", "Jan 26"],
        datasets: [
          {
            label: "VENDOR A (Surface Flaw / Fastening)",
            data: [2.95, 3.40, 2.80, 4.10, 3.80, 5.68],
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.1)",
            tension: 0.35,
            borderWidth: 3,
            fill: true
          },
          {
            label: "VENDOR B (Incomplete Fastening / Hardness)",
            data: [4.20, 6.10, 5.50, 8.20, 9.40, 12.85],
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244, 63, 94, 0.1)",
            tension: 0.35,
            borderWidth: 3,
            fill: true
          },
          {
            label: "VENDOR C (Thread Wear / Alloy)",
            data: [1.80, 2.10, 2.40, 2.90, 2.30, 2.59],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.05)",
            tension: 0.35,
            borderWidth: 2.5,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#cbd5e1", font: { family: "Inter", size: 11 } }
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            borderWidth: 1,
            borderColor: "rgba(255, 255, 255, 0.1)"
          }
        },
        scales: {
          y: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: {
              color: "#94a3b8",
              callback: val => `${val}%`
            }
          },
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          }
        }
      }
    });
  }

  // 3. Tolerance Leakage Bell Curve & Loss Area
  const ctxLeakage = document.getElementById("chart-tolerance-leakage");
  if (ctxLeakage && !chartToleranceLeakage) {
    chartToleranceLeakage = new Chart(ctxLeakage, {
      type: "line",
      data: {
        labels: ["180", "190", "195", "205 (Nom)", "210", "216 (Cutoff)", "220", "225", "230 (USL)"],
        datasets: [
          {
            label: "Safe Operating Core Zone (195-216 HV)",
            data: [5, 15, 45, 95, 60, 25, 0, 0, 0],
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.2)",
            tension: 0.4,
            fill: true
          },
          {
            label: "Tolerance Leakage Zone ($438.7k Field Claims)",
            data: [0, 0, 0, 0, 0, 25, 70, 110, 85],
            borderColor: "#f43f5e",
            backgroundColor: "rgba(244, 63, 94, 0.35)",
            tension: 0.4,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#cbd5e1", font: { family: "Inter", size: 11 } }
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.9)",
            borderWidth: 1,
            borderColor: "rgba(255, 255, 255, 0.1)",
            callbacks: {
              afterLabel: (ctx) => ctx.datasetIndex === 1 ? "⚠️ 25.5% of all warranty claims originate here" : "Nominal compliant process"
            }
          }
        },
        scales: {
          y: {
            display: false
          },
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          }
        }
      }
    });
  }
}

// -------------------------------------------------------------
// TIER 1: Real-time Telemetry Simulator & SHAP Waterfall
// -------------------------------------------------------------
function updateSimulator() {
  const temp = parseFloat(document.getElementById("slider-spindle-temp").value);
  const torque = parseFloat(document.getElementById("slider-peak-torque").value);
  const angle = parseFloat(document.getElementById("slider-shutoff-angle").value);
  const vibration = parseFloat(document.getElementById("slider-vibration").value);
  const toolWear = parseFloat(document.getElementById("slider-tool-wear").value);
  const pressure = parseFloat(document.getElementById("slider-pressure").value);

  document.getElementById("val-spindle-temp").innerText = `${temp.toFixed(1)} °C`;
  document.getElementById("val-peak-torque").innerText = `${torque.toFixed(1)} Nm`;
  document.getElementById("val-shutoff-angle").innerText = `${angle.toFixed(1)} °`;
  document.getElementById("val-vibration").innerText = `${vibration.toFixed(2)} g`;
  document.getElementById("val-tool-wear").innerText = `${toolWear.toFixed(1)} %`;
  document.getElementById("val-pressure").innerText = `${pressure.toFixed(1)} psi`;

  triggerSimulation();
}

async function triggerSimulation() {
  const stationId = document.getElementById("sel-station").value;
  const temp = parseFloat(document.getElementById("slider-spindle-temp").value);
  const torque = parseFloat(document.getElementById("slider-peak-torque").value);
  const angle = parseFloat(document.getElementById("slider-shutoff-angle").value);
  const vibration = parseFloat(document.getElementById("slider-vibration").value);
  const toolWear = parseFloat(document.getElementById("slider-tool-wear").value);
  const pressure = parseFloat(document.getElementById("slider-pressure").value);

  const payload = {
    station_id: stationId,
    spindle_temp_c: temp,
    system_pressure_psi: pressure,
    vibration_rms_g: vibration,
    tool_wear_index: toolWear,
    hours_since_service: toolWear * 0.35,
    peak_torque_nm: torque,
    shutoff_angle_deg: angle,
    torque_status: (torque > 13.0 || angle < 300) ? "NOK_HIGH" : "OK",
    material_hardness_hv: 220.0,
    tensile_strength_mpa: 645.0
  };

  try {
    const res = await fetch("/api/tier1/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      renderSimulationResults(data);
    } else {
      renderLocalSimulation(payload);
    }
  } catch (err) {
    renderLocalSimulation(payload);
  }
}

function renderSimulationResults(data) {
  const mode = data.predicted_failure_mode;
  const confidence = (data.confidence * 100).toFixed(1);

  document.getElementById("res-failure-mode").innerText = mode;
  document.getElementById("res-confidence").innerText = `(Confidence: ${confidence}%)`;

  const badge = document.getElementById("badge-defect-severity");
  if (mode === "NONE") {
    badge.innerText = "NORMAL OPERATION";
    badge.style.background = "rgba(16, 185, 129, 0.15)";
    badge.style.color = "#34d399";
    badge.style.borderColor = "rgba(16, 185, 129, 0.4)";
  } else {
    badge.innerText = "DEFECT DETECTED";
    badge.style.background = "rgba(239, 68, 68, 0.15)";
    badge.style.color = "#f87171";
    badge.style.borderColor = "rgba(239, 68, 68, 0.4)";
  }

  // Render SHAP Contributors
  const container = document.getElementById("shap-contributor-list");
  container.innerHTML = "";

  const maxShap = Math.max(...data.top_physical_contributors.map(c => Math.abs(c.shap_impact)), 0.1);

  data.top_physical_contributors.forEach(item => {
    const isRisk = item.direction === "RISK_INFLATING";
    const barWidth = Math.min(100, Math.round((Math.abs(item.shap_impact) / maxShap) * 90) + 10);
    const sign = item.shap_impact >= 0 ? "+" : "";

    const div = document.createElement("div");
    div.className = "shap-item";
    div.innerHTML = `
      <div class="shap-label-row">
        <span class="shap-feat-name">${item.feature} (${item.value})</span>
        <span class="shap-impact-val ${isRisk ? 'text-rose' : 'text-emerald'}">
          ${sign}${item.shap_impact.toFixed(3)} (${isRisk ? 'Risk Inflating' : 'Protective'})
        </span>
      </div>
      <div class="shap-bar-bg">
        <div class="shap-bar-fill ${isRisk ? 'shap-fill-risk' : 'shap-fill-protective'}" style="width: ${barWidth}%;"></div>
      </div>
    `;
    container.appendChild(div);
  });

  // Render Prescriptive Action
  const act = data.prescriptive_action;
  document.getElementById("prescriptive-urgency").innerText = act.urgency;
  document.getElementById("prescriptive-code").innerText = act.action_code;
  document.getElementById("prescriptive-task").innerText = act.recommended_task;

  const offsetsList = document.getElementById("prescriptive-offsets-list");
  offsetsList.innerHTML = "";

  for (const [k, v] of Object.entries(act.prescriptive_recipe_offsets)) {
    const pill = document.createElement("div");
    pill.className = "offset-pill";
    const formattedKey = k.replace(/_/g, " ");
    const formattedVal = typeof v === "number" ? (v > 0 ? `+${v}` : v) : v;
    pill.innerHTML = `<span class="text-muted">${formattedKey}:</span> <strong class="text-cyan font-mono">${formattedVal}</strong>`;
    offsetsList.appendChild(pill);
  }
}

function renderLocalSimulation(p) {
  let mode = "NONE";
  let conf = 0.99;
  let urgency = "NORMAL - OPERATIONAL";
  let code = "CONTINUE_MONITORING";
  let task = "Process is operating well within 6-sigma tolerance boundaries. Continue standard monitoring.";
  let offsets = { "no_offset_needed": true };

  if (p.tool_wear_index >= 80 || p.vibration_rms_g > 3.2) {
    mode = "DIMENSIONAL_DRIFT";
    urgency = "CRITICAL - IMMEDIATE HALT";
    code = "STOP_LINE_REPLACE_TOOL";
    task = `Replace cutting/fastening bit immediately. Tool wear index is ${p.tool_wear_index.toFixed(1)}% and vibration is ${p.vibration_rms_g.toFixed(2)}g.`;
    offsets = { "tool_replacement": "REQUIRED", "reset_wear_counter": true };
  } else if (p.peak_torque_nm > 13.0 || p.shutoff_angle_deg < 300) {
    mode = "INCOMPLETE_FASTENING";
    urgency = "HIGH URGENCY - BEFORE NEXT BATCH";
    code = "RECALIBRATE_TOOL";
    task = "Recalibrate torque shutoff sensor and inspect fastener thread engagement depth.";
    offsets = {
      "torque_offset_nm": (12.5 - p.peak_torque_nm).toFixed(2),
      "angle_offset_deg": (360.0 - p.shutoff_angle_deg).toFixed(1),
      "rundown_speed_reduction_pct": -10.0
    };
  } else if (p.spindle_temp_c > 65) {
    mode = "SURFACE_FLAW";
    urgency = "MEDIUM - ACTIVE INTERVENTION";
    code = "ADJUST_SPEED_PRESSURE";
    task = "Increase coolant manifold flow rate and throttle spindle speed to avoid surface thermal scoring.";
    offsets = { "spindle_rpm_offset_pct": -8.0, "coolant_pressure_boost_psi": +10.0 };
  } else if (p.system_pressure_psi < 95) {
    mode = "HYDRAULIC_PRESSURE_DROP";
    urgency = "CRITICAL - PRESSURE LOSS";
    code = "ADJUST_SPEED_PRESSURE";
    task = `Inspect hydraulic line filters and pump seal. Current pressure ${p.system_pressure_psi.toFixed(1)} psi is below 95 psi minimum.`;
    offsets = { "hydraulic_pump_boost_psi": (115.0 - p.system_pressure_psi).toFixed(1) };
  }

  const contributors = [
    { feature: "shutoff_angle_deg", value: p.shutoff_angle_deg, shap_impact: p.shutoff_angle_deg < 300 ? 0.45 : -0.12, direction: p.shutoff_angle_deg < 300 ? "RISK_INFLATING" : "PROTECTIVE" },
    { feature: "peak_torque_nm", value: p.peak_torque_nm, shap_impact: p.peak_torque_nm > 13 ? 0.35 : -0.08, direction: p.peak_torque_nm > 13 ? "RISK_INFLATING" : "PROTECTIVE" },
    { feature: "tool_wear_index", value: p.tool_wear_index, shap_impact: p.tool_wear_index > 50 ? 0.22 : -0.15, direction: p.tool_wear_index > 50 ? "RISK_INFLATING" : "PROTECTIVE" },
    { feature: "vibration_rms_g", value: p.vibration_rms_g, shap_impact: p.vibration_rms_g > 2.0 ? 0.18 : -0.10, direction: p.vibration_rms_g > 2.0 ? "RISK_INFLATING" : "PROTECTIVE" },
    { feature: "system_pressure_psi", value: p.system_pressure_psi, shap_impact: -0.05, direction: "PROTECTIVE" }
  ];

  renderSimulationResults({
    predicted_failure_mode: mode,
    confidence: conf,
    top_physical_contributors: contributors,
    prescriptive_action: {
      action_code: code,
      urgency: urgency,
      recommended_task: task,
      prescriptive_recipe_offsets: offsets
    }
  });
}

// -------------------------------------------------------------
// TIER 2: Material QC & Adaptive Parameter Compensation
// -------------------------------------------------------------
function updateMaterialEvaluator() {
  const hardness = parseFloat(document.getElementById("slider-mat-hardness").value);
  const tensile = parseFloat(document.getElementById("slider-mat-tensile").value);
  const supplier = document.getElementById("sel-mat-supplier").value;

  document.getElementById("val-mat-hardness").innerText = `${hardness.toFixed(1)} HV`;
  document.getElementById("val-mat-tensile").innerText = `${tensile.toFixed(1)} MPa`;

  const payload = {
    material_hardness_hv: hardness,
    tensile_strength_mpa: tensile,
    lower_spec_limit: 180.0,
    upper_spec_limit: 230.0,
    supplier_id: supplier
  };

  fetch("/api/tier2/evaluate_material", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  })
  .then(res => res.json())
  .then(data => renderMaterialResults(data))
  .catch(() => renderLocalMaterial(payload));
}

function renderMaterialResults(data) {
  const risk = data.risk_evaluation;
  const forecast = data.downstream_forecast;
  const comp = data.adaptive_compensation;

  const riskStatus = document.getElementById("val-batch-risk-status");
  const riskDesc = document.getElementById("val-batch-risk-desc");
  const riskBox = document.getElementById("box-batch-risk");

  riskStatus.innerText = risk.batch_risk_tier.replace(/_/g, " ");
  riskDesc.innerHTML = `Downstream Risk Probability is <strong>${(risk.downstream_risk_probability * 100).toFixed(1)}%</strong>. ${risk.recommendation}`;

  if (risk.batch_risk_tier.includes("HIGH")) {
    riskBox.style.background = "linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.12) 100%)";
    riskBox.style.borderColor = "rgba(239, 68, 68, 0.35)";
  } else {
    riskBox.style.background = "linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.1) 100%)";
    riskBox.style.borderColor = "rgba(16, 185, 129, 0.35)";
  }

  document.getElementById("forecast-scrap-units").innerText = `${forecast.expected_scrap_count} units`;
  document.getElementById("forecast-scrap-cost").innerText = `Est. Scrap Cost: $${forecast.projected_cost_usd.toLocaleString()}`;
  document.getElementById("forecast-severity").innerText = `${forecast.expected_defect_severity.toFixed(1)} / 5.0`;
  document.getElementById("forecast-mode-title").innerText = `Likely Failure Mode: ${forecast.likely_downstream_failure_mode}`;
  document.getElementById("forecast-root-cause").innerText = forecast.physical_root_cause;

  document.getElementById("comp-target-torque").innerText = `${comp.prescribed_torque_target_nm.toFixed(2)} Nm`;
  document.getElementById("comp-offset-torque").innerText = `${comp.torque_offset_nm >= 0 ? '+' : ''}${comp.torque_offset_nm.toFixed(2)} Nm Offset`;
  document.getElementById("comp-target-angle").innerText = `${comp.prescribed_shutoff_angle_deg.toFixed(1)}°`;
  document.getElementById("comp-offset-angle").innerText = `${comp.angle_offset_deg >= 0 ? '+' : ''}${comp.angle_offset_deg.toFixed(1)}° Offset`;
  document.getElementById("comp-speed-throttle").innerText = `${comp.spindle_speed_offset_pct.toFixed(1)}%`;
  document.getElementById("comp-coolant-mode").innerText = comp.coolant_lubrication_mode.replace(/_/g, " ");
  document.getElementById("comp-guidance").innerText = comp.engineering_guidance;
}

function renderLocalMaterial(p) {
  const delta = p.material_hardness_hv - 205.0;
  const isHigh = p.material_hardness_hv >= 216.0;
  const scrap = isHigh ? Math.round(250 + delta * 8) : 110;

  renderMaterialResults({
    risk_evaluation: {
      batch_risk_tier: isHigh ? "HIGH_RISK_MARGINAL" : "STANDARD_PASS",
      downstream_risk_probability: isHigh ? 0.78 : 0.15,
      recommendation: isHigh ? "Apply adaptive recipe compensation before feeding to line" : "Standard line feed"
    },
    downstream_forecast: {
      expected_scrap_count: Math.max(20, scrap),
      projected_cost_usd: Math.max(20, scrap) * 55.0,
      expected_defect_severity: isHigh ? 2.3 : 1.1,
      likely_downstream_failure_mode: isHigh ? "INCOMPLETE_FASTENING" : "SURFACE_FLAW",
      physical_root_cause: isHigh ? "Excessive material hardness creates elevated torque resistance before shutoff angle is attained." : "Nominal material properties."
    },
    adaptive_compensation: {
      prescribed_torque_target_nm: isHigh ? 12.88 : 12.50,
      torque_offset_nm: isHigh ? 0.38 : 0.0,
      prescribed_shutoff_angle_deg: isHigh ? 371.4 : 360.0,
      angle_offset_deg: isHigh ? 11.4 : 0.0,
      spindle_speed_offset_pct: isHigh ? -7.6 : 0.0,
      coolant_lubrication_mode: isHigh ? "HIGH_PRESSURE_MIST" : "STANDARD_COOLANT",
      engineering_guidance: isHigh ? `Batch hardness is +${delta.toFixed(1)} HV above nominal. Applying torque offset of +0.38 Nm and speed reduction of -7.6% to eliminate incomplete fastening risk.` : "Material is within nominal process capability."
    }
  });
}

// -------------------------------------------------------------
// TIER 3: Warranty Genealogy Trace & SCAR Dossier Viewer
// -------------------------------------------------------------
async function runClaimTrace() {
  const claimId = document.getElementById("inp-trace-id").value.trim();
  if (!claimId) return;

  try {
    const res = await fetch(`/api/tier3/trace/${claimId}`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById("chain-claim-id").innerText = data.claim_id;
      document.getElementById("chain-claim-cost").innerText = `$${data.claim_cost_usd.toFixed(2)}`;
      document.getElementById("chain-part-serial").innerText = data.part_serial_no;
      document.getElementById("chain-cycle-id").innerText = data.fastener_cycle_id;
      document.getElementById("chain-torque-val").innerText = `${data.torque_telemetry.peak_torque_nm} Nm | ${data.torque_telemetry.shutoff_angle_deg}°`;
      document.getElementById("chain-station-id").innerText = data.station_id;
      document.getElementById("chain-batch-id").innerText = data.material_genealogy.material_batch_id;
      document.getElementById("chain-hardness-val").innerText = `${data.material_genealogy.material_hardness_hv} HV (In Leakage Zone!)`;
      document.getElementById("chain-supplier-id").innerText = data.material_genealogy.supplier_id;
    }
  } catch (e) {
    console.error("Genealogy trace fetch error", e);
  }
}

async function openSCAR(vendorId) {
  currentVendorSCAR = vendorId;
  try {
    const res = await fetch(`/api/tier3/scar/${vendorId}`);
    if (res.ok) {
      const data = await res.json();
      renderSCARDossier(data);
    }
  } catch (e) {
    console.error("SCAR report load error", e);
  }
}

function renderSCARDossier(scar) {
  document.getElementById("scar-doc-header").innerText = `${scar.scar_id} • Total Demand: $${scar.financial_recovery_breakdown.total_demand_amount_usd.toLocaleString()}`;

  const body = document.getElementById("scar-dossier-body");
  body.innerHTML = `
    <div style="margin-bottom: 16px;">
      <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 6px;">Executive Non-Conformance Notice</div>
      <p style="color: #cbd5e1; line-height: 1.6;">${scar.executive_summary}</p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px;">
      <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: var(--radius-sm);">
        <span class="text-muted">In-Plant Scrap Recovery:</span>
        <div style="font-family: var(--font-mono); font-size: 16px; font-weight: 700; color: #f87171;">$${scar.financial_recovery_breakdown.in_plant_scrap_recovery_usd.toLocaleString()}</div>
      </div>
      <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: var(--radius-sm);">
        <span class="text-muted">Contractual Chargebacks:</span>
        <div style="font-family: var(--font-mono); font-size: 16px; font-weight: 700; color: #fbbf24;">$${scar.financial_recovery_breakdown.contractual_chargebacks_usd.toLocaleString()}</div>
      </div>
      <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: var(--radius-sm);">
        <span class="text-muted">Field Warranty Recovery:</span>
        <div style="font-family: var(--font-mono); font-size: 16px; font-weight: 700; color: #a5b4fc;">$${scar.financial_recovery_breakdown.field_warranty_recovery_usd.toLocaleString()}</div>
      </div>
    </div>

    <div style="font-size: 14px; font-weight: 700; color: #ffffff; margin-bottom: 10px;">Mandatory 8D Corrective Action Milestones</div>
    <div class="scar-step-list">
      ${scar.mandatory_8d_actions.map(a => `
        <div class="scar-step-item">
          <div class="scar-step-id">${a.step}</div>
          <div class="scar-step-req">${a.requirement}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function copySCAR() {
  const text = document.getElementById("scar-dossier-body").innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert(`8D SCAR Dossier for ${currentVendorSCAR} copied to clipboard!`);
  });
}
