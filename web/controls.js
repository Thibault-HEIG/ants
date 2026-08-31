const CONSTANT_DESCRIPTIONS = {
  // World
  'WORLD_WIDTH': 'Arena width in pixels',
  'WORLD_HEIGHT': 'Arena height in pixels',
  'ZONE_BOUNDARY_X': 'X position dividing ant/spider zones',
  'RANDOM_SEED': 'Seed for reproducible random generation',
  'GENERATION_DURATION': 'Seconds per generational episode',
  'BROADCAST_INTERVAL': 'Seconds between client snapshot broadcasts',
  // Food
  'SUGAR_NUTRITION': 'HP restored when eating sugar',
  'SEED_NUTRITION': 'HP restored when eating a seed',
  'SUGAR_WEIGHT': 'Probability weight for spawning sugar vs seed',
  'SEED_WEIGHT': 'Probability weight for spawning seed vs sugar',
  'FOOD_RADIUS': 'Collision/render radius of food items',
  'MAX_FOOD': 'Global cap on food items in the world',
  'FOOD_SOURCE_RADIUS': 'Visual/collision radius of food sources',
  'FOOD_SOURCE_LIFETIME': 'Seconds before a food source expires',
  'FOOD_SOURCE_SPAWN_RATE': 'Food items generated per second per source',
  'FOOD_SOURCE_SPAWN_RADIUS': 'Scatter radius around a food source',
  'MAX_FOOD_SOURCES': 'Max simultaneous active food sources',
  'FOOD_SOURCE_COOLDOWN': 'Seconds between new source spawns',
  'FOOD_SOURCE_LEFT_ZONE_PROB': 'Chance food spawns in Ant zone (left)',
  'EAT_PICKUP_RADIUS': 'Distance to food required to eat',
  'CARRY_SPEED_MULTIPLIER': 'Speed reduction while carrying an object',
  // Combat
  'ATTACK_DURATION': 'Fixed duration of attack action in seconds',
  // Lifecycle
  'HEALTH_DECAY_RATE': 'HP lost per second for all creatures',
  'MAX_AGE_NORMALIZATION': 'Normalization cap for age neural input',
  // Evolution
  'CONTINUOUS_MUTATION_RATE': 'Mutation rate in continuous mode',
  'CONTINUOUS_MUTATION_STRENGTH': 'Mutation magnitude in continuous mode',
  'CONTINUOUS_SELECTION_FRACTION': 'Bottom fraction culled in continuous mode',
  'GENERATIONAL_MUTATION_RATE': 'Mutation rate in generational mode',
  'GENERATIONAL_MUTATION_STRENGTH': 'Mutation magnitude in generational mode',
  'GENERATIONAL_SELECTION_FRACTION': 'Top fraction selected for breeding',
  'EXTINCTION_MUTATION_RATE': 'Mutation rate after species extinction',
  'EXTINCTION_MUTATION_STRENGTH': 'Mutation magnitude after extinction',
  // Sensors
  'SENSOR_ANGLE': 'Half FOV angle in radians (±80° = 160° total)',
  'MAX_DENSITY_COUNT': 'Normalization cap for nearby entity count',
  'NN_NUM_SENSORS': 'Directional sensors spanning the FOV',
  // Neural Network
  'STATE_INPUTS': 'Non-sensor neural network inputs',
  'NN_INPUTS': 'Total neural network input neurons',
  'NN_HIDDEN_1': 'First hidden layer size',
  'NN_HIDDEN_2': 'Second hidden layer size',
  'NN_OUTPUTS': 'Output neurons (turn, speed, attack, eat, take, release)',
  'GENOME_SIZE': 'Total genome weight count',
  // Ants
  'ANT_COUNT': 'Starting ant population',
  'MAX_ANTS': 'Maximum ant population cap',
  'ANT_INITIAL_HEALTH': 'Starting HP for ants',
  'ANT_REPRODUCTION_THRESHOLD': 'HP threshold for continuous reproduction',
  'ANT_MAX_SPEED': 'Maximum movement speed (px/s)',
  'ANT_RADIUS': 'Collision/render radius',
  'ANT_STRIKE_RANGE': 'Attack reach distance',
  'ANT_TURN_RATE': 'Turning speed (rad/s)',
  'ANT_DAMAGE': 'Damage per attack hit',
  'ANT_ATTACK_COST': 'HP cost per attack action',
  'ANT_EATING_TIME': 'Seconds to consume food',
  'ANT_SENSOR_RANGE': 'Sensor ray detection range',
  'ANT_SENSOR_ANGLE': 'Sensor angle (inherits SENSOR_ANGLE)',
  'DENSITY_RADIUS_ANT': 'Density counting radius for ants',
  'PHEROMONE_STRENGTH': 'Pheromone intensity deposited per tile',
  'PHEROMONE_DURATION': 'Pheromone decay time in seconds',
  'ANT_SPAWN_NB_AT_DELIVERY': 'Ants spawned per food delivery at home',
  // Ant Fitness
  'FITNESS_SURVIVAL_WEIGHT': 'Weight for survival time',
  'FITNESS_TILES_COVERED_WEIGHT': 'Weight for exploration coverage',
  'FITNESS_FOLLOW_PHEROMONES_WEIGHT': 'Weight for following pheromone trails',
  'FITNESS_BRAIN_ORIGINALITY_WEIGHT': 'Weight for genome uniqueness bonus',
  'FITNESS_FOOD_WEIGHT': 'Weight for food consumption',
  'FITNESS_TIMES_EATING_FOR_NOTHING_WEIGHT': 'Penalty for failed eating attempts',
  'FITNESS_ENEMIES_TOUCHED_WEIGHT': 'Weight for combat engagement',
  'FITNESS_TIMES_ATTACKING_FOR_NOTHING_WEIGHT': 'Penalty for missed attacks',
  'FITNESS_TAKEN_OBJECT_WEIGHT': 'Weight for picking up objects',
  'FITNESS_WALK_HOME_DIRECTION_WEIGHT': 'Weight for carrying objects homeward',
  'FITNESS_WALK_OPPOSITE_HOME_WEIGHT': 'Penalty for carrying objects away',
  'FITNESS_RELEASE_ANYWHERE_WEIGHT': 'Penalty for dropping objects outside home',
  'FITNESS_RELEASE_AT_HOME_WEIGHT': 'Weight for delivering objects to anthill',
  // Ant Metric Bounds
  'ANT_METRIC_BOUNDS': 'Normalization bounds for ant metrics',
  // Spiders
  'SPIDER_COUNT': 'Starting spider population',
  'MAX_SPIDERS': 'Maximum spider population cap',
  'SPIDER_INITIAL_HEALTH': 'Starting HP for spiders',
  'SPIDER_REPRODUCTION_THRESHOLD': 'HP threshold for continuous reproduction',
  'SPIDER_MAX_SPEED': 'Maximum movement speed (px/s)',
  'SPIDER_RADIUS': 'Collision/render radius',
  'SPIDER_STRIKE_RANGE': 'Attack reach distance',
  'SPIDER_TURN_RATE': 'Turning speed (rad/s)',
  'SPIDER_DAMAGE': 'Damage per attack hit',
  'SPIDER_ATTACK_COST': 'HP cost per attack action',
  'SPIDER_EATING_TIME': 'Seconds to consume food',
  'SPIDER_SENSOR_RANGE': 'Sensor ray detection range',
  'SPIDER_SENSOR_ANGLE': 'Sensor angle (inherits SENSOR_ANGLE)',
  'DENSITY_RADIUS_SPIDER': 'Density counting radius for spiders',
  'SPIDER_METRIC_BOUNDS': 'Normalization bounds for spider metrics',
};

let ws = null;
let latestSnapshot = null;
let liveConstants = {};
let previousConstants = null;
let showSensors = false;
let fitnessChart = null;
let activeTabId = 'tab-live-analytics';

let lastSnapTime = performance.now();
let fpsFrames = 0;
let fpsVal = 0;


let populationChart = null;
let trainingCharts = {};
let trainingNeedsRedraw = { 'tab-ants-training': true, 'tab-spiders-training': true };
let currentRunId = null;
let maxChartTime = 300.0;
let maxFitnessVal = 100.0;
let maxPopVal = 100.0;

function showBanner(text, type, duration = 3000) {
  const container = document.getElementById('bannerContainer');
  const banner = document.createElement('div');
  banner.className = `banner ${type} show`;
  banner.innerText = text;
  container.appendChild(banner);
  if (duration > 0) {
    setTimeout(() => {
      banner.classList.remove('show');
      setTimeout(() => banner.remove(), 300);
    }, duration);
  }
  return banner;
}

function updateStatus(connected) {
  const badge = document.getElementById("connectionStatus");
  const text = document.getElementById("statusText");
  if (connected) {
    badge.className = "status-badge";
    text.innerText = "Connected";
    // Clear stale banners from before reconnect (e.g. "restarting" banners)
    const container = document.getElementById('bannerContainer');
    container.querySelectorAll('.banner.restarting').forEach(b => b.remove());
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'get_constants' }));
    }
  } else {
    badge.className = "status-badge disconnected";
    text.innerText = "Disconnected";
  }
}

function connectWS() {
  const host = window.location.hostname || 'localhost';
  ws = new WebSocket(`ws://${host}:8765`);
  ws.onopen = () => updateStatus(true);
  ws.onclose = () => {
    updateStatus(false);
    setTimeout(connectWS, 2000);
  };
  ws.onmessage = (evt) => handleMessage(JSON.parse(evt.data));
}

function handleMessage(msg) {
  if (msg.type === "full" || msg.type === "aggregate") {
    handleSnapshot(msg);
  } else if (msg.type === "save_result") {
    showBanner('State saved successfully', 'success');
  } else if (msg.type === "load_result") {
    if (msg.ok) showBanner(msg.message || 'State loaded successfully', 'success');
    else showBanner(msg.message || 'Load failed', 'error');
  } else if (msg.type === "reload_starting") {
    showBanner('⏳ Restarting server with code changes...', 'restarting', 0);
  } else if (msg.type === "reload_result") {
    if (msg.ok) showBanner(msg.message, 'success', 6000);
    else showBanner(msg.message, 'error', 6000);
  }
}

function handleSnapshot(snap) {
  latestSnapshot = snap;

  const now = performance.now();
  fpsFrames++;
  if (now - lastSnapTime >= 1000) {
    fpsVal = fpsFrames;
    fpsFrames = 0;
    lastSnapTime = now;
    document.getElementById("fpsDisplay").innerText = `${fpsVal} FPS`;
  }

  document.getElementById("timeDisplay").innerText = snap.time.toFixed(1) + "s";
  document.getElementById("genDisplay").innerText = snap.generation;

  // Speed display: just the target multiplier
  const target = snap.targetMultiplier !== undefined ? snap.targetMultiplier : snap.speed;
  const actual = snap.actualMultiplier !== undefined ? snap.actualMultiplier : snap.speed;

  document.getElementById("speedDisplay").innerText = target + "x";
  document.getElementById("actualDtDisplay").innerText = "Actual dt: " + Math.round(actual) + "x";

  document.getElementById("btnPause").innerHTML = snap.paused ? "▶ Resume" : "⏸ Pause";
  document.getElementById("ultraBanner").style.display = snap.ultra ? "flex" : "none";
  if (snap.type === "full" && !snap.ultra) {
    Renderer.render(snap, Renderer.getCamera(), showSensors);
  }
}

  function renderBoundsTable(tbodyId, bounds) {
    const tbody = document.getElementById(tbodyId);
    tbody.innerHTML = "";
    for (const [metric, data] of Object.entries(bounds)) {
      const tr = document.createElement("tr");
      const isWarn = data.max > data.bound;
      const isUnder = data.max < (0.5 * data.bound);
      let statusHTML = '<span style="color:var(--success);">OK</span>';
      if (isWarn) {
        statusHTML = '<span style="color:var(--warning); font-weight:bold;">⚠️ Over</span>';
      } else if (isUnder) {
        statusHTML = '<span style="color:var(--warning); font-weight:bold;">‼️ Under</span>';
      }
      tr.innerHTML = `
      <td>${metric}</td>
      <td class="mono">${data.max.toFixed(1)}</td>
      <td class="mono">${data.bound.toFixed(1)}</td>
      <td>${statusHTML}</td>
    `;
      tbody.appendChild(tr);
    }
  }

  // Square-root scale: spreads low values more than linear, less than log.
  class SqrtScale extends Chart.Scale {
    constructor(cfg) {
      super(cfg);
      this._tickValues = [1, 5, 10, 20, 50, 100, 200, 300];
    }

    buildTicks() {
      const maxVal = this.max || 300;
      this.ticks = this._tickValues
        .filter(v => v <= maxVal)
        .map(v => ({ value: v }));
      return this.ticks;
    }

    getPixelForValue(value) {
      const min = this.min || 0;
      const max = this.max || 300;
      const sqrtMin = Math.sqrt(Math.max(0, min));
      const sqrtMax = Math.sqrt(Math.max(0, max));
      const sqrtVal = Math.sqrt(Math.max(0, value));
      const ratio = (sqrtVal - sqrtMin) / (sqrtMax - sqrtMin || 1);
      return this.getPixelForDecimal(ratio);
    }

    getValueForPixel(pixel) {
      const min = this.min || 0;
      const max = this.max || 300;
      const sqrtMin = Math.sqrt(Math.max(0, min));
      const sqrtMax = Math.sqrt(Math.max(0, max));
      const decimal = this.getDecimalForPixel(pixel);
      const sqrtVal = sqrtMin + decimal * (sqrtMax - sqrtMin);
      return sqrtVal * sqrtVal;
    }
  }
  SqrtScale.id = 'sqrt';
  SqrtScale.defaults = {
    grid: { color: "#3d3228" },
    ticks: { color: "#9b8b7a" }
  };
  Chart.register(SqrtScale);

  function createChart(ctx, datasets, yScaleConfig, title) {
    return new Chart(ctx, {
      type: "scatter",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear',
            display: true,
            min: 0,
            max: maxChartTime,
            ticks: {
              color: "#9b8b7a",
              maxTicksLimit: 10,
              callback: function (value) { return value.toFixed(0) + 's'; }
            },
            grid: { color: "rgba(255,255,255,0.05)" }
          },
          y: yScaleConfig
        },
        plugins: {
          title: {
            display: !!title,
            text: title || '',
            color: '#e8e0d4',
            font: { size: 14, weight: '600' },
            padding: { bottom: 10 }
          },
          legend: {
            labels: { color: "#e8e0d4", font: { size: 12 }, usePointStyle: true, boxWidth: 20 }
          },
          tooltip: { enabled: false }
        }
      }
    });
  }

  function initChart() {
    const ctxFit = document.getElementById("fitnessChart").getContext("2d");
    const ctxPop = document.getElementById("populationChart").getContext("2d");

    const dsFit = [
      { label: "Ant Best", borderColor: "#6fb87a", borderDash: [4, 4], data: [], borderWidth: 1.5, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
      { label: "Ant Avg", borderColor: "#4a7c59", data: [], borderWidth: 2, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
      { label: "Spider Best", borderColor: "#c94a4a", borderDash: [4, 4], data: [], borderWidth: 1.5, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
      { label: "Spider Avg", borderColor: "#8c3a3a", data: [], borderWidth: 2, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
    ];

    const dsPop = [
      { label: "Ants", borderColor: "#6fb87a", data: [], borderWidth: 2, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line', fill: true, backgroundColor: "rgba(111, 184, 122, 0.1)" },
      { label: "Spiders", borderColor: "#c94a4a", data: [], borderWidth: 2, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line', fill: true, backgroundColor: "rgba(201, 74, 74, 0.1)" }
    ];

    const fitScaleConfig = {
      type: 'logarithmic',
      grid: { color: "#3d3228" },
      ticks: {
        color: "#9b8b7a",
        callback: function (value) {
          if (![1, 10, 100, 1000, 10000].includes(value)) return null;
          return value;
        }
      },
      min: 1.0,
      max: maxFitnessVal
    };

    const popScaleConfig = {
      type: 'sqrt',
      min: 0,
      max: maxPopVal,
      ticks: {
        color: "#9b8b7a",
        callback: function (value) {
          return Number.isInteger(value) ? value : null;
        }
      },
      grid: { color: "#3d3228" }
    };

    fitnessChart = createChart(ctxFit, dsFit, fitScaleConfig, 'Fitness Tracking');
    populationChart = createChart(ctxPop, dsPop, popScaleConfig, 'Population History');
  }

  // Training chart zone definitions: maps canvas ID → { species, bestKey, avgKey, title }
  const TRAINING_CHART_DEFS = [
    // Ant Eat
    { id: 'antEatLossChart', species: 'Ant', bestKey: 'times_eating_for_nothing_best', avgKey: 'times_eating_for_nothing_avg', title: 'Eat Loss (eating for nothing)', color: '#c94a4a', colorAvg: '#8c3a3a' },
    { id: 'antEatSuccessChart', species: 'Ant', bestKey: 'computed_food_eaten_best', avgKey: 'computed_food_eaten_avg', title: 'Eat Success (food eaten)', color: '#6fb87a', colorAvg: '#4a7c59' },
    // Ant Attack
    { id: 'antAttackLossChart', species: 'Ant', bestKey: 'computed_times_attacking_for_nothing_best', avgKey: 'computed_times_attacking_for_nothing_avg', title: 'Attack Loss (attacking for nothing)', color: '#c94a4a', colorAvg: '#8c3a3a' },
    { id: 'antAttackSuccessChart', species: 'Ant', bestKey: 'computed_enemies_touched_best', avgKey: 'computed_enemies_touched_avg', title: 'Attack Success (enemies touched)', color: '#6fb87a', colorAvg: '#4a7c59' },
    // Ant Pheromone (success only)
    { id: 'antPheromonePlacementChart', species: 'Ant', bestKey: 'released_pheromone_around_food_source_best', avgKey: 'released_pheromone_around_food_source_avg', title: 'Pheromone Placement (near food)', color: '#6fb87a', colorAvg: '#4a7c59' },
    { id: 'antPheromoneSuccessChart', species: 'Ant', bestKey: 'follow_pheromones_best', avgKey: 'follow_pheromones_avg', title: 'Pheromone Success (follow pheromones)', color: '#6fb87a', colorAvg: '#4a7c59' },
    // Ant Carry
    { id: 'antCarryLossChart', species: 'Ant', bestKey: 'walk_with_object_in_opposite_home_direction_best', avgKey: 'walk_with_object_in_opposite_home_direction_avg', title: 'Carry Loss (walk opposite home)', color: '#c94a4a', colorAvg: '#8c3a3a' },
    { id: 'antCarrySuccessChart', species: 'Ant', bestKey: 'release_at_home_count_best', avgKey: 'release_at_home_count_avg', title: 'Carry Success (release at home)', color: '#6fb87a', colorAvg: '#4a7c59' },
    { id: 'antCarryHomeChart', species: 'Ant', bestKey: 'walk_with_object_in_home_direction_best', avgKey: 'walk_with_object_in_home_direction_avg', title: 'Carry (walk home direction)', color: '#5a9e8f', colorAvg: '#3d7a6d' },
    // Spider Eat
    { id: 'spiderEatLossChart', species: 'Spider', bestKey: 'times_eating_for_nothing_best', avgKey: 'times_eating_for_nothing_avg', title: 'Eat Loss (eating for nothing)', color: '#c94a4a', colorAvg: '#8c3a3a' },
    { id: 'spiderEatSuccessChart', species: 'Spider', bestKey: 'computed_food_eaten_best', avgKey: 'computed_food_eaten_avg', title: 'Eat Success (food eaten)', color: '#6fb87a', colorAvg: '#4a7c59' },
    // Spider Attack
    { id: 'spiderAttackLossChart', species: 'Spider', bestKey: 'computed_times_attacking_for_nothing_best', avgKey: 'computed_times_attacking_for_nothing_avg', title: 'Attack Loss (attacking for nothing)', color: '#c94a4a', colorAvg: '#8c3a3a' },
    { id: 'spiderAttackSuccessChart', species: 'Spider', bestKey: 'computed_enemies_touched_best', avgKey: 'computed_enemies_touched_avg', title: 'Attack Success (enemies touched)', color: '#6fb87a', colorAvg: '#4a7c59' },
  ];

  function createTrainingChart(canvasId, title, bestColor, avgColor) {
    const el = document.getElementById(canvasId);
    if (!el) return null;
    const ctx = el.getContext('2d');
    return new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Best', borderColor: bestColor, borderDash: [4, 4], data: [], borderWidth: 1.5, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
          { label: 'Avg', borderColor: avgColor, data: [], borderWidth: 2, pointRadius: 0, tension: 0.1, showLine: true, pointStyle: 'line' },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            type: 'linear',
            display: true,
            min: 0,
            max: 300,
            ticks: {
              color: '#9b8b7a',
              maxTicksLimit: 6,
              callback: function (value) { return value.toFixed(0) + 's'; }
            },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            type: 'linear',
            min: 0,
            ticks: { color: '#9b8b7a' },
            grid: { color: '#3d3228' }
          }
        },
        plugins: {
          title: {
            display: true,
            text: title,
            color: '#e8e0d4',
            font: { size: 12, weight: '600' },
            padding: { bottom: 6 }
          },
          legend: {
            labels: { color: '#e8e0d4', font: { size: 10 }, usePointStyle: true, boxWidth: 15 }
          },
          tooltip: { enabled: false }
        }
      }
    });
  }
  let chartRenderingMode = 'Continuous';

  function initTrainingCharts() {
    for (const def of TRAINING_CHART_DEFS) {
      const chart = createTrainingChart(def.id, def.title, def.color, def.colorAvg);
      if (chart) {
        trainingCharts[def.id] = chart;
      }
    }

    const btnContinuous = document.getElementById('btnContinuous');
    const btnGenerational = document.getElementById('btnGenerational');
    if (btnContinuous && btnGenerational) {
      btnContinuous.addEventListener('click', () => {
        chartRenderingMode = 'Continuous';
        btnContinuous.style.background = '#c4a35a';
        btnContinuous.style.color = '#1a1a1a';
        btnGenerational.style.background = 'transparent';
        btnGenerational.style.color = '#9b8b7a';
        if (lastChartData.training.length > 0) updateTrainingCharts(lastChartData.training);
      });
      btnGenerational.addEventListener('click', () => {
        chartRenderingMode = 'Generational';
        btnGenerational.style.background = '#c4a35a';
        btnGenerational.style.color = '#1a1a1a';
        btnContinuous.style.background = 'transparent';
        btnContinuous.style.color = '#9b8b7a';
        if (lastChartData.training.length > 0) updateTrainingCharts(lastChartData.training);
      });
    }
  }

  function updateTrainingCharts(rows) {
    if (!rows || rows.length === 0) return;

    const GENERATION_DURATION = window.getConstant ? (window.getConstant('GENERATION_DURATION') || 300) : 300;

    // Group rows by species and metric for easy plotting
    const grouped = { Ant: {}, Spider: {} };
    for (const r of rows) {
      if (!grouped[r.species_name][r.metric]) {
        grouped[r.species_name][r.metric] = [];
      }
      grouped[r.species_name][r.metric].push(r);
    }

    let maxTime = 300;

    for (const def of TRAINING_CHART_DEFS) {
      const chart = trainingCharts[def.id];
      if (!chart) continue;

      // e.g. def.bestKey is 'computed_food_eaten_best' => we want metric 'computed_food_eaten'
      const metricName = def.bestKey.replace('_best', '');
      const metricRows = grouped[def.species][metricName] || [];

      let datasetBest = [];
      let datasetAvg = [];

      if (chartRenderingMode === 'Generational') {
        const halfGen = GENERATION_DURATION / 2;
        const buckets = {};

        for (const r of metricRows) {
          const bucketIdx = Math.floor(r.time / halfGen);
          if (!buckets[bucketIdx]) buckets[bucketIdx] = { sumBest: 0, sumAvg: 0, timeSum: 0, count: 0 };
          const b = buckets[bucketIdx];

          b.sumBest += r.best / Math.max(1.0, r.best_lifetime);
          b.sumAvg += r.avg / Math.max(1.0, r.avg_lifetime);
          b.timeSum += r.time;
          b.count++;
        }

        const sortedBuckets = Object.keys(buckets).map(Number).sort((a, b) => a - b);
        for (const bIdx of sortedBuckets) {
          const b = buckets[bIdx];
          if (b.count > 0) {
            datasetBest.push({ x: b.timeSum / b.count, y: b.sumBest / b.count });
            datasetAvg.push({ x: b.timeSum / b.count, y: b.sumAvg / b.count });
          }
        }
      } else {
        for (const r of metricRows) {
          datasetBest.push({ x: r.time, y: r.best / Math.max(1.0, r.best_lifetime) });
          datasetAvg.push({ x: r.time, y: r.avg / Math.max(1.0, r.avg_lifetime) });
        }
      }

      if (metricRows.length > 0) {
        const lastT = metricRows[metricRows.length - 1].time;
        if (lastT > maxTime) maxTime = lastT;
      }

      chart.data.datasets[0].data = datasetBest;
      chart.data.datasets[1].data = datasetAvg;
      chart.options.scales.x.max = Math.max(300, maxTime * 1.05);
      chart.update('none');
    }
  }

  function resetTrainingCharts() {
    for (const chart of Object.values(trainingCharts)) {
      chart.data.datasets.forEach(ds => ds.data = []);
      chart.options.scales.x.max = 300;
      chart.update('none');
    }
  }

  // -------------------------------------------------------------------------
  // SQLite DB Polling
  // -------------------------------------------------------------------------

  let lastChartData = { live: [], training: [] };

  async function pollData() {
    if (ws && ws.readyState !== WebSocket.OPEN) return;

    if (!currentRunId) {
      try {
        const res = await fetch('/api/runs');
        const runs = await res.json();
        if (runs.length > 0) {
          currentRunId = runs[0].id;
        }
      } catch (e) { }
      if (!currentRunId) return;
    }

    // Fetch live stats
    try {
      const res = await fetch(`/api/runs/${currentRunId}/stats`);
      const stats = await res.json();
      lastChartData.live = stats;
      updateLiveStatsAndCharts(stats);
    } catch (e) { }

    // Fetch training stats
    try {
      const res = await fetch(`/api/runs/${currentRunId}/training`);
      const training = await res.json();
      lastChartData.training = training;
      if (activeTabId === 'tab-ants-training' || activeTabId === 'tab-spiders-training') {
        updateTrainingCharts(training);
        trainingNeedsRedraw[activeTabId] = false;
      } else {
        trainingNeedsRedraw['tab-ants-training'] = true;
        trainingNeedsRedraw['tab-spiders-training'] = true;
      }
    } catch (e) { }

    // Fetch metric bounds
    try {
      const res = await fetch(`/api/runs/${currentRunId}/bounds`);
      const bounds = await res.json();
      const formattedBounds = { Ant: {}, Spider: {} };
      for (const b of bounds) {
        formattedBounds[b.species_name][b.metric] = { max: b.max_observed, bound: b.bound };
      }
      renderBoundsTable("antBoundsBody", formattedBounds.Ant);
      renderBoundsTable("spiderBoundsBody", formattedBounds.Spider);
    } catch (e) { }
  }

  function updateLiveStatsAndCharts(rows) {
    if (!fitnessChart || !populationChart) return;

    const dsFit = fitnessChart.data.datasets;
    const dsPop = populationChart.data.datasets;

    dsFit.forEach(ds => ds.data = []);
    dsPop.forEach(ds => ds.data = []);

    let maxTime = 300.0;
    let maxFit = 1.0;
    let maxPop = 100.0;

    let latestAnt = null;
    let latestSpider = null;

    for (const r of rows) {
      const t = r.time;
      if (t > maxTime) maxTime = t;
      if (r.max_pop > maxPop) maxPop = r.max_pop;

      if (r.species_name === 'Ant') {
        latestAnt = r;
        if (r.fitness_best > maxFit) maxFit = r.fitness_best;
        if (r.fitness_avg > maxFit) maxFit = r.fitness_avg;
        dsFit[0].data.push({ x: t, y: Math.max(1.0, r.fitness_best) });
        dsFit[1].data.push({ x: t, y: Math.max(1.0, r.fitness_avg) });
        dsPop[0].data.push({ x: t, y: r.alive });
      } else if (r.species_name === 'Spider') {
        latestSpider = r;
        if (r.fitness_best > maxFit) maxFit = r.fitness_best;
        if (r.fitness_avg > maxFit) maxFit = r.fitness_avg;
        dsFit[2].data.push({ x: t, y: Math.max(1.0, r.fitness_best) });
        dsFit[3].data.push({ x: t, y: Math.max(1.0, r.fitness_avg) });
        dsPop[1].data.push({ x: t, y: r.alive });
      }
    }

    if (latestAnt) {
      document.getElementById('antAlive').innerText = `${latestAnt.alive}/${latestAnt.max_pop}`;
      document.getElementById('antBestFit').innerText = latestAnt.fitness_best.toFixed(2);
      document.getElementById('antAvgFit').innerText = latestAnt.fitness_avg.toFixed(2);
      document.getElementById('antBestLife').innerText = latestAnt.lifetime_best.toFixed(1) + 's';
      document.getElementById('antAvgLife').innerText = latestAnt.lifetime_avg.toFixed(1) + 's';
      document.getElementById('antBestFood').innerText = latestAnt.food_best.toFixed(0);
      document.getElementById('antAvgFood').innerText = latestAnt.food_avg.toFixed(1);
      document.getElementById('antBestEnemies').innerText = latestAnt.enemies_best.toFixed(0);
      document.getElementById('antAvgEnemies').innerText = latestAnt.enemies_avg.toFixed(1);
      document.getElementById('antBestTiles').innerText = latestAnt.tiles_best.toFixed(0);
      document.getElementById('antAvgTiles').innerText = latestAnt.tiles_avg.toFixed(1);
      if (document.getElementById('antBestHomeFood')) document.getElementById('antBestHomeFood').innerText = latestAnt.release_home_best.toFixed(0);
      if (document.getElementById('antAvgHomeFood')) document.getElementById('antAvgHomeFood').innerText = latestAnt.release_home_avg.toFixed(1);
    }

    if (latestSpider) {
      document.getElementById('spiderAlive').innerText = `${latestSpider.alive}/${latestSpider.max_pop}`;
      document.getElementById('spiderBestFit').innerText = latestSpider.fitness_best.toFixed(2);
      document.getElementById('spiderAvgFit').innerText = latestSpider.fitness_avg.toFixed(2);
      document.getElementById('spiderBestLife').innerText = latestSpider.lifetime_best.toFixed(1) + 's';
      document.getElementById('spiderAvgLife').innerText = latestSpider.lifetime_avg.toFixed(1) + 's';
      document.getElementById('spiderBestFood').innerText = latestSpider.food_best.toFixed(0);
      document.getElementById('spiderAvgFood').innerText = latestSpider.food_avg.toFixed(1);
      document.getElementById('spiderBestEnemies').innerText = latestSpider.enemies_best.toFixed(0);
      document.getElementById('spiderAvgEnemies').innerText = latestSpider.enemies_avg.toFixed(1);
      document.getElementById('spiderBestTiles').innerText = latestSpider.tiles_best.toFixed(0);
      document.getElementById('spiderAvgTiles').innerText = latestSpider.tiles_avg.toFixed(1);
    }

    fitnessChart.options.scales.x.max = maxTime * 1.05;
    populationChart.options.scales.x.max = maxTime * 1.05;

    fitnessChart.options.scales.y.max = maxFit * 1.2;
    populationChart.options.scales.y.max = maxPop;

    fitnessChart.update('none');
    populationChart.update('none');
  }

  setInterval(pollData, 2000);

  document.addEventListener("DOMContentLoaded", () => {
    Renderer.init(document.getElementById('simCanvas'));
    initChart();
    initTrainingCharts();
    connectWS();

    // Tab switching
    document.querySelectorAll('.analytics-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.analytics-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        const targetId = tab.dataset.tab;
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) targetPanel.classList.add('active');
        activeTabId = targetId;
        if (trainingNeedsRedraw[targetId] && lastChartData.training.length > 0) {
          updateTrainingCharts(lastChartData.training);
          trainingNeedsRedraw[targetId] = false;
        }
      });
    });



    // Controls
    document.getElementById("btnPause").onclick = () => {
      ws.send(JSON.stringify({ type: "pause_toggle" }));
    };
    document.getElementById("btnSpeedUp").onclick = () => ws.send(JSON.stringify({ type: "set_speed", direction: "up" }));
    document.getElementById("btnSpeedDown").onclick = () => ws.send(JSON.stringify({ type: "set_speed", direction: "down" }));
    document.getElementById("btnUltra").onclick = () => ws.send(JSON.stringify({ type: "toggle_ultra" }));
    document.getElementById("btnSensors").onclick = () => {
      showSensors = !showSensors;
      document.getElementById("btnSensors").classList.toggle("btn-active", showSensors);
    };

    const originalHandleSnapshot = handleSnapshot;
    handleSnapshot = function (snap) {
      originalHandleSnapshot(snap);
      document.getElementById("btnPause").classList.toggle("btn-active", snap.paused);
      document.getElementById("btnUltra").classList.toggle("btn-active", snap.ultra);
    };

    // Zoom
    document.getElementById("btnZoomIn").onclick = () => {
      Renderer.zoomIn();
      document.getElementById("zoomDisplay").innerText = Renderer.getZoomPercent();
    };
    document.getElementById("btnZoomOut").onclick = () => {
      Renderer.zoomOut();
      document.getElementById("zoomDisplay").innerText = Renderer.getZoomPercent();
    };

    // Modals
    const saveModal = document.getElementById("saveModal");
    const controlsModal = document.getElementById("controlsModal");

    document.getElementById("btnSave").onclick = () => {
      document.getElementById("saveFilename").value = new Date().toISOString().replace(/T/, '-').replace(/:/g, '-').slice(2, 16);
      saveModal.classList.add("open");
    };
    document.getElementById("btnCancelSave").onclick = () => saveModal.classList.remove("open");
    document.getElementById("btnConfirmSave").onclick = () => {
      ws.send(JSON.stringify({
        type: 'save_full_state',
        filename: document.getElementById("saveFilename").value,
        notes: document.getElementById("saveNotes").value
      }));
      saveModal.classList.remove("open");
    };

    document.getElementById("btnControls").onclick = () => controlsModal.classList.add("open");
    document.getElementById("btnCloseControls").onclick = () => controlsModal.classList.remove("open");



    // Load DB Modal
    const loadDbModal = document.getElementById("loadDbModal");
    document.getElementById("btnLoadDB").onclick = async () => {
      try {
        const res = await fetch('/api/runs');
        const runs = await res.json();
        const select = document.getElementById("loadDbRunId");
        select.innerHTML = "";
        runs.forEach(r => {
          const option = document.createElement("option");
          option.value = r.id;
          option.innerText = `Run ${r.id} - ${r.started_at} ${r.notes ? '(' + r.notes + ')' : ''}`;
          select.appendChild(option);
        });
        loadDbModal.classList.add("open");
      } catch (e) {
        showBanner("Failed to load runs from DB", "error");
      }
    };

    document.getElementById("btnCancelLoadDb").onclick = () => loadDbModal.classList.remove("open");

    document.getElementById("btnConfirmLoadDb").onclick = () => {
      const runId = document.getElementById("loadDbRunId").value;
      const resetStats = document.getElementById("loadDbResetStats").checked;
      if (runId && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'load_from_db',
          run_id: parseInt(runId, 10),
          reset_stats: resetStats
        }));
        currentRunId = null; // force repoll of current run ID
      }
      loadDbModal.classList.remove("open");
    };

    // Reload with code changes
    document.getElementById("btnReload").onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "reload_with_changes" }));
      }
    };

    const canvas = document.getElementById('simCanvas');
    canvas.addEventListener("mousedown", (e) => Renderer.startDrag(e));
    window.addEventListener("mousemove", (e) => Renderer.moveDrag(e));
    window.addEventListener("mouseup", () => Renderer.endDrag());

    // Scroll to zoom
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        Renderer.zoomIn();
      } else {
        Renderer.zoomOut();
      }
      document.getElementById("zoomDisplay").innerText = Renderer.getZoomPercent();
    }, { passive: false });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (evt) => {
      if (evt.target.tagName === "INPUT" || evt.target.tagName === "TEXTAREA") return;
      if (evt.code === "Space") { evt.preventDefault(); ws.send(JSON.stringify({ type: "pause_toggle" })); }
      else if (evt.code === "ArrowRight" || evt.code === "ArrowUp") { ws.send(JSON.stringify({ type: "set_speed", direction: "up" })); }
      else if (evt.code === "ArrowLeft" || evt.code === "ArrowDown") { ws.send(JSON.stringify({ type: "set_speed", direction: "down" })); }
      else if (evt.code === "KeyS") { showSensors = !showSensors; document.getElementById("btnSensors").classList.toggle("btn-active", showSensors); }
      else if (evt.code === "KeyU") { ws.send(JSON.stringify({ type: "toggle_ultra" })); }
      else if (evt.code === "KeyP") { ws.send(JSON.stringify({ type: "print_population" })); }
    });
  });
