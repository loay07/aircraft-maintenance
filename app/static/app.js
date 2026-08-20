const chartColors = {
  accent: "#35d0c0",
  accent2: "#f5a524",
  danger: "#f0576a",
  grid: "rgba(159, 171, 199, 0.15)",
  text: "#9fabc7",
};

Chart.defaults.color = chartColors.text;
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.borderColor = chartColors.grid;

let scatterChart = null;
let trendChart = null;
let sensorChart = null;

function setMetricText(selector, value) {
  document.querySelectorAll(`[data-metric="${selector}"]`).forEach((el) => {
    el.textContent = value;
  });
}

async function loadMetrics() {
  const res = await fetch("/api/metrics");
  const data = await res.json();

  setMetricText("mae", data.mae.toFixed(2));
  setMetricText("rmse", data.rmse.toFixed(2));
  setMetricText("r2", data.r2.toFixed(3));

  const ctx = document.getElementById("scatter-chart");
  const points = data.scatter.map((d) => ({ x: d.actual, y: d.predicted }));
  const maxVal = Math.max(...points.map((p) => Math.max(p.x, p.y)), data.rul_cap) + 5;

  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Test engines",
          data: points,
          backgroundColor: "rgba(53, 208, 192, 0.75)",
          pointRadius: 4,
        },
        {
          label: "Perfect prediction",
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          type: "line",
          borderColor: "rgba(159, 171, 199, 0.6)",
          borderDash: [6, 6],
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 14 } } },
      scales: {
        x: { title: { display: true, text: "Actual RUL (capped, cycles)" }, grid: { color: chartColors.grid } },
        y: { title: { display: true, text: "Predicted RUL (cycles)" }, grid: { color: chartColors.grid } },
      },
    },
  });
}

async function loadEngineList() {
  const res = await fetch("/api/engines");
  const engines = await res.json();

  const select = document.getElementById("engine-select");
  select.innerHTML = engines
    .map((e) => `<option value="${e.engine_id}">Engine ${e.engine_id} &mdash; ${e.max_cycle} cycles observed</option>`)
    .join("");

  select.addEventListener("change", () => loadEngine(select.value));
  loadEngine(engines[0].engine_id);
}

function statusFor(rul) {
  if (rul > 60) return { label: "Healthy", cls: "badge-good" };
  if (rul >= 30) return { label: "Monitor", cls: "badge-warn" };
  return { label: "Maintenance Needed", cls: "badge-danger" };
}

async function loadEngine(engineId) {
  const statusEl = document.getElementById("demo-status");
  statusEl.textContent = "Running model...";

  const [summaryRes, trendRes] = await Promise.all([
    fetch("/api/engines").then((r) => r.json()),
    fetch(`/api/engines/${engineId}/trend`).then((r) => r.json()),
  ]);

  const summary = summaryRes.find((e) => e.engine_id === Number(engineId));

  document.getElementById("stat-predicted").textContent = summary.predicted_rul_last_cycle.toFixed(1);
  document.getElementById("stat-actual").textContent = summary.true_rul_capped.toFixed(1);
  document.getElementById("stat-cycle-info").textContent = `at cycle ${summary.max_cycle}`;

  const status = statusFor(summary.predicted_rul_last_cycle);
  const badge = document.getElementById("stat-badge");
  badge.textContent = status.label;
  badge.className = `stat-badge ${status.cls}`;

  renderTrendChart(trendRes);
  renderSensorChart(trendRes);

  statusEl.textContent = "";
}

function renderTrendChart(trend) {
  const ctx = document.getElementById("trend-chart");
  if (trendChart) trendChart.destroy();

  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.cycles,
      datasets: [
        {
          label: "Predicted RUL",
          data: trend.predicted_rul,
          borderColor: chartColors.accent,
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
          tension: 0.15,
        },
        {
          label: "True remaining life (reference)",
          data: trend.true_rul_reference,
          borderColor: "rgba(159, 171, 199, 0.7)",
          backgroundColor: "transparent",
          borderDash: [6, 6],
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 14 } } },
      scales: {
        x: { title: { display: true, text: "Cycle" }, grid: { display: false } },
        y: { title: { display: true, text: "RUL (cycles)" }, grid: { color: chartColors.grid } },
      },
    },
  });
}

function renderSensorChart(trend) {
  const ctx = document.getElementById("sensor-chart");
  if (sensorChart) sensorChart.destroy();

  const colors = [chartColors.accent, chartColors.accent2, chartColors.danger];
  const sensorNames = Object.keys(trend.sensors);

  sensorChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: trend.cycles,
      datasets: sensorNames.map((name, i) => ({
        label: name.replace("sensor_", "Sensor "),
        data: trend.sensors[name],
        borderColor: colors[i % colors.length],
        backgroundColor: "transparent",
        pointRadius: 0,
        borderWidth: 2,
        yAxisID: `y${i}`,
        tension: 0.15,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 14 } } },
      scales: {
        x: { title: { display: true, text: "Cycle" }, grid: { display: false } },
        y0: { type: "linear", position: "left", grid: { color: chartColors.grid }, title: { display: true, text: "Sensor 11" } },
        y1: { type: "linear", position: "right", grid: { display: false }, title: { display: true, text: "Sensor 4" } },
        y2: { type: "linear", position: "right", grid: { display: false }, display: false },
      },
    },
  });
}

loadMetrics();
loadEngineList();
