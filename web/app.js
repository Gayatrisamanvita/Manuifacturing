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

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  console.log("PrecisionPulse AI Suite initialized.");
  initCharts();
  openSCAR("VENDOR_B");
  triggerSimulation();
});

// Tab Switcher
function switchTab(tabId) {
  const buttons = document.querySelectorAll(".nav-tab-btn");
  buttons.forEach(btn => btn.classList.remove("active"));
  
  const targetBtn = {
    "tab-overview": "tab-btn-overview",
    "tab-tier1": "tab-btn-tier1",
    "tab-tier2": "tab-btn-tier2",
    "tab-tier3": "tab-btn-tier3",
  }[tabId];

  if (targetBtn && document.getElementById(targetBtn)) {
    document.getElementById(targetBtn).classList.add("active");
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
