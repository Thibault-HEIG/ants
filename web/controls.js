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

let lastSnapTime = performance.now();
let fpsFrames = 0;
let fpsVal = 0;

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
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({type: 'get_constants'}));
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
  } else if (msg.type === "constants_data") {
    liveConstants = msg.constants;
    if (previousConstants) {
      const changed = [];
      for (const key in liveConstants) {
        if (previousConstants[key] && previousConstants[key].value !== liveConstants[key].value) {
          changed.push(key);
        }
      }
      if (changed.length > 0) {
        showBanner(`Successfully updated: ${changed.join(', ')}`, 'success', 5000);
      }
      previousConstants = null;
      document.querySelector('.banner.restarting')?.remove();
    }
    renderConstants();
  } else if (msg.type === "save_result") {
    showBanner('State saved successfully', 'success');
  } else if (msg.type === "load_result") {
    showBanner('State loaded successfully', 'success');
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
  document.getElementById("speedDisplay").innerText = snap.speed + "x";
  document.getElementById("btnPause").innerHTML = snap.paused ? "▶ Resume" : "⏸ Pause";
  document.getElementById("ultraBanner").style.display = snap.ultra ? "flex" : "none";

  if (snap.stats) {
    if (snap.stats.Ant) {
      const a = snap.stats.Ant;
      document.getElementById("antAlive").innerText = `${a.alive}/${a.maxPop}`;
      document.getElementById("antBestFit").innerText = (a.bestFitness || 0).toFixed(2);
      document.getElementById("antAvgFit").innerText = (a.avgFitness || 0).toFixed(2);
      document.getElementById("antBestLife").innerText = (a.bestLifetime || 0).toFixed(1) + "s";
      document.getElementById("antAvgLife").innerText = (a.avgLifetime || 0).toFixed(1) + "s";
      document.getElementById("antBestFood").innerText = a.bestComputedFood || a.bestFood || 0;
      document.getElementById("antAvgFood").innerText = (a.avgComputedFood || a.avgFood || 0).toFixed(1);
      document.getElementById("antBestEnemies").innerText = a.bestComputedEnemies || a.bestEnemies || 0;
      document.getElementById("antAvgEnemies").innerText = (a.avgComputedEnemies || a.avgEnemies || 0).toFixed(1);
      document.getElementById("antBestTiles").innerText = a.bestTilesCovered || 0;
      document.getElementById("antAvgTiles").innerText = (a.avgTilesCovered || 0).toFixed(1);
      document.getElementById("antBestHomeFood").innerText = a.bestReleaseAtHome || 0;
      document.getElementById("antAvgHomeFood").innerText = (a.avgReleaseAtHome || 0).toFixed(1);
    }
    if (snap.stats.Spider) {
      const s = snap.stats.Spider;
      document.getElementById("spiderAlive").innerText = `${s.alive}/${s.maxPop}`;
      document.getElementById("spiderBestFit").innerText = (s.bestFitness || 0).toFixed(2);
      document.getElementById("spiderAvgFit").innerText = (s.avgFitness || 0).toFixed(2);
      document.getElementById("spiderBestLife").innerText = (s.bestLifetime || 0).toFixed(1) + "s";
      document.getElementById("spiderAvgLife").innerText = (s.avgLifetime || 0).toFixed(1) + "s";
      document.getElementById("spiderBestFood").innerText = s.bestComputedFood || s.bestFood || 0;
      document.getElementById("spiderAvgFood").innerText = (s.avgComputedFood || s.avgFood || 0).toFixed(1);
      document.getElementById("spiderBestEnemies").innerText = s.bestComputedEnemies || s.bestEnemies || 0;
      document.getElementById("spiderAvgEnemies").innerText = (s.avgComputedEnemies || s.avgEnemies || 0).toFixed(1);
      document.getElementById("spiderBestTiles").innerText = s.bestTilesCovered || 0;
      document.getElementById("spiderAvgTiles").innerText = (s.avgTilesCovered || 0).toFixed(1);
    }

    if (fitnessChart && fitnessChart.data.labels.length > 0) {
       fitnessChart.data.labels.push(snap.time.toFixed(1));
       fitnessChart.data.datasets[0].data.push(snap.stats.Ant ? snap.stats.Ant.bestFitness : 0);
       fitnessChart.data.datasets[1].data.push(snap.stats.Ant ? snap.stats.Ant.avgFitness : 0);
       fitnessChart.data.datasets[2].data.push(snap.stats.Spider ? snap.stats.Spider.bestFitness : 0);
       fitnessChart.data.datasets[3].data.push(snap.stats.Spider ? snap.stats.Spider.avgFitness : 0);
       if (fitnessChart.data.labels.length > 200) {
         fitnessChart.data.labels.shift();
         fitnessChart.data.datasets.forEach(ds => ds.data.shift());
       }
       fitnessChart.update('none');
    }
  }

  if (snap.metricBounds) {
    renderBoundsTable("antBoundsBody", snap.metricBounds.Ant || {});
    renderBoundsTable("spiderBoundsBody", snap.metricBounds.Spider || {});
  }

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
    tr.innerHTML = `
      <td>${metric}</td>
      <td class="mono">${data.max.toFixed(1)}</td>
      <td class="mono">${data.bound.toFixed(1)}</td>
      <td>${isWarn ? '<span style="color:var(--warning); font-weight:bold;">⚠️ Over</span>' : '<span style="color:var(--success);">OK</span>'}</td>
    `;
    tbody.appendChild(tr);
  }
}

function initChart() {
  const ctx = document.getElementById("fitnessChart").getContext("2d");
  fitnessChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ['0'],
      datasets: [
        { label: "Ant Best", borderColor: "#6fb87a", data: [0], borderWidth: 2, pointRadius: 0 },
        { label: "Ant Avg", borderColor: "#4a7c59", borderDash: [4,4], data: [0], borderWidth: 1.5, pointRadius: 0 },
        { label: "Spider Best", borderColor: "#c94a4a", data: [0], borderWidth: 2, pointRadius: 0 },
        { label: "Spider Avg", borderColor: "#8c3a3a", borderDash: [4,4], data: [0], borderWidth: 1.5, pointRadius: 0 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { display: false },
        y: { type: 'logarithmic', grid: { color: "#3d3228" }, ticks: { color: "#9b8b7a" } }
      },
      plugins: {
        legend: { labels: { color: "#e8e0d4", font: { size: 12 } } }
      }
    }
  });
}

function renderConstants() {
  const container = document.getElementById("constantsAccordion");
  container.innerHTML = "";

  const groups = {
    "World": [],
    "Food": [],
    "Combat": [],
    "Evolution": [],
    "Sensors": [],
    "Neural Network": [],
    "Ants": [],
    "Spiders": []
  };

  for (const [key, data] of Object.entries(liveConstants)) {
    let g = "World";
    if (key.includes("FOOD") || key.includes("SUGAR") || key.includes("SEED") || key === 'EAT_PICKUP_RADIUS' || key === 'CARRY_SPEED_MULTIPLIER') g = "Food";
    else if (key.includes("ATTACK") && !key.includes("ANT") && !key.includes("SPIDER") && !key.includes("FITNESS")) g = "Combat";
    else if (key.includes("MUTATION") || key.includes("SELECTION") || key.includes("EXTINCTION") || key === 'GENERATION_DURATION') g = "Evolution";
    else if (key.includes("SENSOR") && !key.includes("ANT") && !key.includes("SPIDER")) g = "Sensors";
    else if (key.includes("NN_") || key === 'GENOME_SIZE' || key === 'STATE_INPUTS') g = "Neural Network";
    else if (key.startsWith("ANT") || key.startsWith("MAX_ANT") || key.includes("PHEROMONE") || key.startsWith("DENSITY_RADIUS_ANT") || (key.startsWith("FITNESS") && !key.startsWith("SPIDER"))) g = "Ants";
    else if (key.startsWith("SPIDER") || key.startsWith("MAX_SPIDER") || key.startsWith("DENSITY_RADIUS_SPIDER")) g = "Spiders";
    
    if (groups[g]) groups[g].push({key, data});
  }

  for (const [gName, items] of Object.entries(groups)) {
    if (items.length === 0) continue;
    
    const grp = document.createElement("div");
    grp.className = "accordion-group";
    const isHighlighted = gName === "Evolution" || 
      (items.some(i => i.key.includes('FITNESS')) && (gName === 'Ants' || gName === 'Spiders'));
    if (isHighlighted) grp.classList.add("highlight-evolution");
    
    const header = document.createElement("div");
    header.className = "accordion-header";
    header.innerHTML = `<span>${isHighlighted ? '⭐ ' : ''}${gName}</span><span>▼</span>`;
    
    const body = document.createElement("div");
    body.className = "accordion-body";
    
    header.onclick = () => {
      body.classList.toggle("open");
      header.innerHTML = `<span>${gName}</span><span>${body.classList.contains("open") ? "▲" : "▼"}</span>`;
    };

    items.forEach(item => {
      const desc = CONSTANT_DESCRIPTIONS[item.key] || "";
      const row = document.createElement("div");
      row.className = "constant-row";
      row.innerHTML = `
        <div class="constant-info">
          <span class="constant-name">${item.key}</span>
          ${desc ? `<span class="constant-desc">${desc}</span>` : ""}
        </div>
        <div class="constant-val">${item.data.value}</div>
      `;
      body.appendChild(row);
    });

    grp.appendChild(header);
    grp.appendChild(body);
    container.appendChild(grp);
  }
}

window.getConstant = function(key) {
  return liveConstants[key] ? liveConstants[key].value : null;
};

document.addEventListener("DOMContentLoaded", () => {
  Renderer.init(document.getElementById('simCanvas'));
  initChart();
  connectWS();

  // Scroll Sync
  const container = document.getElementById("scrollContainer");
  const indicators = document.querySelectorAll(".indicator");
  container.addEventListener('scroll', () => {
    const idx = Math.round(container.scrollLeft / window.innerWidth);
    indicators.forEach((ind, i) => ind.classList.toggle('active', i === idx));
  });
  indicators.forEach((ind, i) => {
    ind.onclick = () => {
      container.scrollTo({ left: i * window.innerWidth, behavior: 'smooth' });
    };
  });

  // Controls
  document.getElementById("btnPause").onclick = () => ws.send(JSON.stringify({ type: "pause_toggle" }));
  document.getElementById("btnSpeedUp").onclick = () => ws.send(JSON.stringify({ type: "set_speed", direction: "up" }));
  document.getElementById("btnSpeedDown").onclick = () => ws.send(JSON.stringify({ type: "set_speed", direction: "down" }));
  document.getElementById("btnUltra").onclick = () => ws.send(JSON.stringify({ type: "toggle_ultra" }));
  document.getElementById("btnSensors").onclick = () => {
    showSensors = !showSensors;
    document.getElementById("btnSensors").classList.toggle("btn-active", showSensors);
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
    document.getElementById("saveFilename").value = new Date().toISOString().replace(/T/, '-').replace(/:/g, '-').slice(2,16);
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

  // Load
  const fileInput = document.getElementById("fileLoad");
  document.getElementById("btnLoad").onclick = () => fileInput.click();
  fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
           const parsedJSON = JSON.parse(ev.target.result);
           ws.send(JSON.stringify({type: 'load_save_data', content: parsedJSON}));
        } catch (err) {
           showBanner("Error parsing JSON", "error");
        }
      };
      reader.readAsText(file);
    }
    fileInput.value = "";
  };

  // Constants Refresh
  document.getElementById("btnRefreshConstants").onclick = () => {
    previousConstants = JSON.parse(JSON.stringify(liveConstants));
    ws.send(JSON.stringify({type: 'refresh_constants'}));
    showBanner("Restarting simulation to apply constants...", "restarting", 0);
  };

  // Dragging
  const canvas = document.getElementById('simCanvas');
  canvas.addEventListener("mousedown", (e) => Renderer.startDrag(e));
  window.addEventListener("mousemove", (e) => Renderer.moveDrag(e));
  window.addEventListener("mouseup", () => Renderer.endDrag());

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
