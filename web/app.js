// Canvas drawing can't read CSS custom properties, so the palette from
// styles.css (--blue, --red, --objective, --text, --surface-2) is mirrored
// here. Keep both in sync if the theme changes.
const THEME = {
  blueRgb: [95, 143, 191],
  redRgb: [200, 102, 85],
  goldRgb: [214, 168, 79],
  lightRgb: [226, 227, 231],
  darkRgb: [19, 19, 22],
  blue: "#5f8fbf",
  red: "#c86655",
  gold: "#d6a84f",
  goldLight: "#f0d58c",
  dark: "#131316",
  light: "#e2e3e7",
  sceneBg: "#0d1110",
  // Flat/conceptual-mode ground floor - buildings sit on a cleared dirt lot,
  // not the near-black void the letterbox background uses.
  dirtFloor: "#33291c",
  transparent: "rgba(0,0,0,0)",
};

// Matches the packed-facility generator's cell_size (src/core/instances.py
// create_packed_facility) - every building edge lands on a multiple of this,
// so the grid overlay and ground tiling both use it too, instead of an
// unrelated round number that never lines up with real building edges.
const GROUND_CELL_WORLD_SIZE = 4;

function rgba([r, g, b], alpha) {
  return `rgba(${r},${g},${b},${alpha})`;
}

// --- DOM references & top-level state -------------------------------------

const instanceSelect = document.getElementById("instanceSelect");
const policyButton = document.getElementById("policyButton");
const policyModalOverlay = document.getElementById("policyModalOverlay");
const policyModalClose = document.getElementById("policyModalClose");
const policyBlue = document.getElementById("policyBlue");
const policyRed = document.getElementById("policyRed");
const playPause = document.getElementById("playPause");
const playIcon = playPause.querySelector(".icon-play");
const pauseIcon = playPause.querySelector(".icon-pause");
const resumeIcon = playPause.querySelector(".icon-resume");
const stepButton = document.getElementById("step");
const resetButton = document.getElementById("reset");
const terrainRealistic = document.getElementById("terrainRealistic");
const terrainConceptual = document.getElementById("terrainConceptual");
const metrics = document.getElementById("metrics");
const eyeButton = document.getElementById("eyeButton");
const configModalOverlay = document.getElementById("configModalOverlay");
const configModalClose = document.getElementById("configModalClose");
const instanceConfigTitle = document.getElementById("instanceConfigTitle");
const instanceConfigBody = document.getElementById("instanceConfigBody");

let state = null;
let socket = null;
let trails = new Map();
let controlsReady = false;
let terrainImage = new Image();
let terrainImageUrl = null;
let playbackTouched = false;
let lastAnimationRender = 0;
let instancePaths = {};
let loadedConfigForInstance = null;
// Recently-destroyed scouts: agentId -> { start, position, heading }. Kept
// around for DEATH_FADE_MS after death so the vehicle fades out instead of
// just vanishing the instant the server marks it dead.
let deathFades = new Map();

// Per-team fog-of-war: each team starts knowing only its own side of the
// front line. Territory beyond the line is revealed permanently once a
// vehicle from that team gets within its own vision radius of it, mirroring
// "collected map knowledge" rather than momentary visibility.
let fogCanvases = {};
const FOG_PX_PER_UNIT = 4;
const DEATH_FADE_MS = 700;

// --- Global / per-team view toggle -------------------------------------------
// Three views always exist (global map, blue team perspective, red team
// perspective). The first entry in `viewOrder` is rendered large; the other
// two sit as small thumbnails. Clicking a thumbnail swaps it to the front.

const bigSlot = document.getElementById("bigSlot");
const thumbSlot1 = document.getElementById("thumbSlot1");
const thumbSlot2 = document.getElementById("thumbSlot2");

const viewDefs = {
  global: { label: "Global Map", drawOptions: {} },
  blue: { label: "Blue Team View", drawOptions: { teamPerspective: "blue" } },
  red: { label: "Red Team View", drawOptions: { teamPerspective: "red" } },
};

const viewPanels = {};
let viewOrder = ["global", "blue", "red"];

Object.entries(viewDefs).forEach(([key, def]) => {
  const panel = document.createElement("div");
  panel.className = "view-panel";
  panel.dataset.view = key;
  const header = document.createElement("header");
  header.textContent = def.label;
  const canvas = document.createElement("canvas");
  canvas.width = 720;
  canvas.height = 720;
  panel.append(header, canvas);
  panel.addEventListener("click", () => {
    if (viewOrder[0] === key) return;
    focusView(key);
  });
  viewPanels[key] = { panel, canvas };
});

function focusView(key) {
  viewOrder = [key, ...viewOrder.filter((entry) => entry !== key)];
  layoutViews();
  if (state) requestAnimationFrame((animationTime) => renderAllScenes(animationTime));
}

function layoutViews() {
  bigSlot.replaceChildren(viewPanels[viewOrder[0]].panel);
  thumbSlot1.replaceChildren(viewPanels[viewOrder[1]].panel);
  thumbSlot2.replaceChildren(viewPanels[viewOrder[2]].panel);
}

layoutViews();

connect();
requestAnimationFrame(animate);

// --- Navigation & control wiring --------------------------------------------

document.querySelectorAll("[data-page-link]").forEach((element) => {
  element.addEventListener("click", (event) => {
    event.preventDefault();
    showPage(element.dataset.pageLink);
  });
});

playPause.addEventListener("click", () => {
  if (!state) return;
  const nextPlaying = !state.controls.playing;
  if (nextPlaying) playbackTouched = true;
  sendControl({ playing: nextPlaying });
});

stepButton.addEventListener("click", () => {
  if (state?.controls.playing) {
    stepButton.classList.remove("is-denied");
    void stepButton.offsetWidth;
    stepButton.classList.add("is-denied");
    return;
  }
  sendCommand("step");
});
stepButton.addEventListener("animationend", () => stepButton.classList.remove("is-denied"));
resetButton.addEventListener("click", () => {
  trails = new Map();
  fogCanvases = {};
  scoutedFractionCache = { value: 0, ts: 0 };
  playbackTouched = false;
  sendCommand("reset");
});

policyBlue.addEventListener("change", () => sendControl({ policy_blue: policyBlue.value }));
policyRed.addEventListener("change", () => sendControl({ policy_red: policyRed.value }));
policyButton.addEventListener("click", () => policyModalOverlay.classList.add("is-open"));
policyModalClose.addEventListener("click", () => policyModalOverlay.classList.remove("is-open"));
policyModalOverlay.addEventListener("click", (event) => {
  if (event.target === policyModalOverlay) policyModalOverlay.classList.remove("is-open");
});
terrainRealistic.addEventListener("click", () => {
  setTerrainButtons(true);
  sendControl({ terrain_enabled: true });
});
terrainConceptual.addEventListener("click", () => {
  setTerrainButtons(false);
  sendControl({ terrain_enabled: false });
});

function setTerrainButtons(realisticEnabled) {
  terrainRealistic.classList.toggle("active", realisticEnabled);
  terrainRealistic.setAttribute("aria-pressed", String(realisticEnabled));
  terrainConceptual.classList.toggle("active", !realisticEnabled);
  terrainConceptual.setAttribute("aria-pressed", String(!realisticEnabled));
}
instanceSelect.addEventListener("change", () => {
  trails = new Map();
  fogCanvases = {};
  scoutedFractionCache = { value: 0, ts: 0 };
  playbackTouched = false;
  loadedConfigForInstance = null;
  sendCommand("load_instance", { instance_id: instanceSelect.value });
});

eyeButton.addEventListener("click", () => {
  openConfigModal();
  loadInstanceConfig();
});

configModalClose.addEventListener("click", closeConfigModal);
configModalOverlay.addEventListener("click", (event) => {
  if (event.target === configModalOverlay) closeConfigModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeConfigModal();
    policyModalOverlay.classList.remove("is-open");
  }
});

function openConfigModal() {
  configModalOverlay.classList.add("is-open");
}

function closeConfigModal() {
  configModalOverlay.classList.remove("is-open");
}

function loadInstanceConfig() {
  const instanceId = instanceSelect.value;
  const path = instancePaths[instanceId];
  if (!instanceId || !path || loadedConfigForInstance === instanceId) return;
  instanceConfigTitle.textContent = path;
  instanceConfigBody.textContent = "Loading…";
  fetch(`/instance-files/${path}`)
    .then((response) => response.text())
    .then((text) => {
      try {
        instanceConfigBody.textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch {
        instanceConfigBody.textContent = text;
      }
      loadedConfigForInstance = instanceId;
    })
    .catch(() => {
      instanceConfigBody.textContent = "Failed to load config.";
    });
}

// --- WebSocket connection ----------------------------------------------------

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "state") {
      update(message.state);
    }
  });
  socket.addEventListener("close", () => setTimeout(connect, 700));
}

function sendCommand(command, extra = {}) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ command, ...extra }));
  }
}

function sendControl(data) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ command: "control", data }));
  }
}

// --- State sync & the render loop -------------------------------------------

function update(nextState) {
  if (state) {
    for (const [agentId, agent] of Object.entries(state.agents)) {
      const next = nextState.agents[agentId];
      if (agent.alive && next && !next.alive) {
        deathFades.set(agentId, {
          start: performance.now(),
          position: agent.position,
          heading: agent.heading,
        });
      }
    }
  }
  state = nextState;
  // Blue holds this ground and knows the terrain - only Red's side of the
  // line (Blue's territory, from Red's perspective) is fogged.
  updateFog("red", state);
  syncControls();
  updateTrails();
  renderAllScenes();
  renderMetrics();
}

function animate(timestamp) {
  if (state && timestamp - lastAnimationRender > 33) {
    renderAllScenes(timestamp);
    lastAnimationRender = timestamp;
  }
  requestAnimationFrame(animate);
}

function renderAllScenes(animationTime = performance.now()) {
  Object.entries(viewDefs).forEach(([key, def]) => {
    drawScene(viewPanels[key].canvas, state, { ...def.drawOptions, animationTime });
  });
}

function showPage(pageName) {
  const pageIds = {
    simulation: "simulationPage",
    about: "aboutPage",
  };
  const nextPage = document.getElementById(pageIds[pageName]);
  const currentPage = document.querySelector(".page.active");

  document.querySelectorAll(".nav-link").forEach((link) => {
    link.classList.toggle("active", link.dataset.pageLink === pageName);
  });
  document.body.classList.toggle("page-about", pageName === "about");

  if (currentPage && currentPage !== nextPage) {
    currentPage.classList.remove("active");
    currentPage.addEventListener("transitionend", () => currentPage.classList.remove("visible"), { once: true });
  }

  // Add "visible" (display: block) a frame before "active" so the opacity/
  // transform change is a transition rather than an instant jump.
  nextPage.classList.add("visible");
  void nextPage.offsetWidth;
  requestAnimationFrame(() => {
    nextPage.classList.add("active");
    if (state) {
      requestAnimationFrame((animationTime) => renderAllScenes(animationTime));
    }
  });
}

// Reconcile toolbar widgets with the latest snapshot from the server. Dropdown
// contents are only rebuilt when their option set actually changed, since
// this runs on every WebSocket update.
function syncControls() {
  const control = state.controls;
  playIcon.classList.remove("is-active");
  pauseIcon.classList.remove("is-active");
  resumeIcon.classList.remove("is-active");
  if (control.playing) {
    pauseIcon.classList.add("is-active");
    playPause.setAttribute("aria-label", "Pause");
    playPause.title = "Pause";
  } else if (playbackTouched) {
    resumeIcon.classList.add("is-active");
    playPause.setAttribute("aria-label", "Resume");
    playPause.title = "Resume";
  } else {
    playIcon.classList.add("is-active");
    playPause.setAttribute("aria-label", "Play");
    playPause.title = "Play";
  }

  if (!controlsReady) {
    policyBlue.innerHTML = "";
    policyRed.innerHTML = "";
    control.policy_options.forEach((option) => {
      policyBlue.append(new Option(option, option));
      policyRed.append(new Option(option, option));
    });
    controlsReady = true;
  }

  policyBlue.value = control.policy_blue;
  policyRed.value = control.policy_red;
  setTerrainButtons(control.terrain_enabled);

  const currentInstanceOptions = Array.from(instanceSelect.options).map((option) => option.value).join("|");
  const nextInstanceOptions = control.instances.map((instance) => instance.id).join("|");
  if (currentInstanceOptions !== nextInstanceOptions) {
    instanceSelect.innerHTML = "";
    instancePaths = {};
    control.instances.forEach((instance) => {
      instanceSelect.append(new Option(instance.name, instance.id));
      instancePaths[instance.id] = instance.path;
    });
  }
  instanceSelect.value = control.instance_id;
}

// --- Scene drawing (Canvas 2D) -----------------------------------------------

function updateTrails() {
  for (const [agentId, agent] of Object.entries(state.agents)) {
    if (!agent.alive) continue;
    const trail = trails.get(agentId) ?? [];
    trail.push(agent.position);
    while (trail.length > 44) trail.shift();
    trails.set(agentId, trail);
  }
}

function drawScene(canvas, sceneState, options = {}) {
  const ctx = canvas.getContext("2d");
  resizeCanvas(canvas);
  const width = canvas.width;
  const height = canvas.height;
  const [worldW, worldH] = sceneState.world_size;
  const pixelRatio = window.devicePixelRatio || 1;
  const visualPadding = Math.max(24 * pixelRatio, Math.min(width, height) * 0.075);
  const scale = Math.min((width - visualPadding * 2) / worldW, (height - visualPadding * 2) / worldH);
  const offsetX = (width - worldW * scale) / 2;
  const offsetY = (height - worldH * scale) / 2;
  const view = {
    ctx,
    scale,
    offsetX,
    offsetY,
    worldW,
    worldH,
    buildings: sceneState.buildings,
    animationTime: options.animationTime ?? performance.now(),
  };

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.sceneBg;
  ctx.fillRect(0, 0, width, height);
  const realistic = !!sceneState.terrain?.enabled;
  // Dirt floor first (under any custom terrain image), same in both modes -
  // buildings render identically regardless of the realistic/flat toggle now,
  // so the ground shouldn't visually diverge between the two either.
  drawFlatGround(view, worldW, worldH);
  drawTerrain(view, sceneState, worldW, worldH);
  if (realistic) {
    drawGroundTexture(view, worldW, worldH);
  } else {
    drawGrid(view, worldW, worldH);
  }

  if (options.teamPerspective === "red") {
    // Buildings/zones must be fully invisible in undiscovered territory, not
    // just dimmed - draw them on a scratch layer, erase the undiscovered
    // part entirely, then composite what's left onto the bare (still fully
    // visible) ground floor drawn above.
    const scratch = getScratchCanvas(width, height);
    const scratchView = { ...view, ctx: scratch.ctx };
    drawZoneBand(scratchView, sceneState.blue_spawn_zone, THEME.blueRgb);
    sceneState.red_spawn_zones.forEach((zone) => drawZoneBand(scratchView, zone, THEME.redRgb));
    sceneState.buildings.forEach((building) => drawBuilding(scratchView, building, realistic));
    maskToRevealed(scratch.ctx, view, "red", sceneState);
    ctx.drawImage(scratch.canvas, 0, 0);

    drawFog(view, "red", sceneState);
  } else {
    drawZoneBand(view, sceneState.blue_spawn_zone, THEME.blueRgb);
    sceneState.red_spawn_zones.forEach((zone) => drawZoneBand(view, zone, THEME.redRgb));
    sceneState.buildings.forEach((building) => drawBuilding(view, building, realistic));
  }
  drawFrontLine(view, sceneState);

  sceneState.detections.forEach((line) => {
    if (lineVisibleForOptions(sceneState, line, options)) {
      drawLine(view, line.start, line.end, rgba(THEME.lightRgb, 0.45), 1.3);
    }
  });

  for (const [agentId, trail] of trails.entries()) {
    const agent = sceneState.agents[agentId];
    if (!agent || !agentVisibleForOptions(sceneState, agentId, options)) continue;
    drawTrail(view, trail, agent.team === "blue" ? THEME.blueRgb : THEME.redRgb);
  }

  for (const [agentId, agent] of Object.entries(sceneState.agents)) {
    if (!agentVisibleForOptions(sceneState, agentId, options)) continue;
    drawAgent(view, agentId, agent, options.viewer === agentId, options.teamPerspective);
  }
}

function agentVisibleForOptions(sceneState, agentId, options) {
  const agent = sceneState.agents[agentId];
  if (!agent) return false;
  if (options.selectedOnly && !agent.visible) return false;
  if (options.teamPerspective === "blue" && agent.team === "red") {
    return sceneState.team_visibility.blue_visible_reds.includes(agentId);
  }
  if (options.teamPerspective === "red" && agent.team === "blue") {
    return sceneState.team_visibility.red_visible_blues.includes(agentId);
  }
  return true;
}

function lineVisibleForOptions(sceneState, line, options) {
  return agentVisibleForOptions(sceneState, line.blue, options) && agentVisibleForOptions(sceneState, line.red, options);
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const nextWidth = Math.max(1, Math.floor(rect.width * ratio));
  const nextHeight = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
}

function worldToCanvas(view, point) {
  return [
    view.offsetX + point[0] * view.scale,
    view.offsetY + (view.worldH - point[1]) * view.scale,
  ];
}

function drawFlatGround(view, worldW, worldH) {
  const { ctx, scale, offsetX, offsetY } = view;
  ctx.save();
  ctx.fillStyle = THEME.dirtFloor;
  ctx.fillRect(offsetX, offsetY, worldW * scale, worldH * scale);
  ctx.restore();
}

function drawGrid(view, worldW, worldH) {
  const { ctx, scale, offsetX, offsetY } = view;
  ctx.save();
  ctx.strokeStyle = "rgba(183,176,161,0.10)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= worldW; x += GROUND_CELL_WORLD_SIZE) {
    const px = offsetX + x * scale;
    ctx.beginPath();
    ctx.moveTo(px, offsetY);
    ctx.lineTo(px, offsetY + worldH * scale);
    ctx.stroke();
  }
  for (let y = 0; y <= worldH; y += GROUND_CELL_WORLD_SIZE) {
    const py = offsetY + y * scale;
    ctx.beginPath();
    ctx.moveTo(offsetX, py);
    ctx.lineTo(offsetX + worldW * scale, py);
    ctx.stroke();
  }
  ctx.restore();
}

// Deterministic pseudo-random in [0,1) for a world-space cell — used so
// scattered ground detail stays put across the ~30 re-renders/sec instead of
// flickering, without needing to store per-cell state.
function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

// In realistic mode, texture the open ground with faint pavement seams —
// distinct roads read as a toy grid at this scale, so this stays a plain
// grey street surface instead.
function drawGroundTexture(view, worldW, worldH) {
  const { ctx, scale, offsetX, offsetY } = view;
  ctx.save();
  ctx.strokeStyle = "rgba(200,200,204,0.06)";
  ctx.lineWidth = 1;
  const seam = 4;
  for (let x = 0; x <= worldW; x += seam) {
    const px = offsetX + x * scale;
    ctx.beginPath();
    ctx.moveTo(px, offsetY);
    ctx.lineTo(px, offsetY + worldH * scale);
    ctx.stroke();
  }
  for (let y = 0; y <= worldH; y += seam) {
    const py = offsetY + y * scale;
    ctx.beginPath();
    ctx.moveTo(offsetX, py);
    ctx.lineTo(offsetX + worldW * scale, py);
    ctx.stroke();
  }
  ctx.restore();
}

function drawZoneBand(view, zone, colorRgb) {
  const { ctx, worldH } = view;
  const nearTop = zone.y + zone.h / 2 < worldH / 2;
  const edgeY = nearTop ? 0 : worldH;
  const farY = nearTop ? zone.y + zone.h : zone.y;
  const [xLeft] = worldToCanvas(view, [zone.x, 0]);
  const [xRight] = worldToCanvas(view, [zone.x + zone.w, 0]);
  const [, yEdge] = worldToCanvas(view, [0, edgeY]);
  const [, yFar] = worldToCanvas(view, [0, farY]);
  const gradient = ctx.createLinearGradient(0, yEdge, 0, yFar);
  gradient.addColorStop(0, rgba(colorRgb, 0.16));
  gradient.addColorStop(1, rgba(colorRgb, 0));
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.fillRect(Math.min(xLeft, xRight), Math.min(yEdge, yFar), Math.abs(xRight - xLeft), Math.abs(yFar - yEdge));
  ctx.restore();
}

// --- Front line & per-team fog-of-war ---------------------------------------

// True if `team`'s spawn sits on the smaller-world-y side of the front line.
function teamHomeIsBelowLine(team, sceneState) {
  const spawnCenterY =
    team === "blue" ? sceneState.blue_spawn_zone.center[1] : sceneState.red_spawn_zones[0].center[1];
  return spawnCenterY < sceneState.front_line_y;
}

function drawFrontLine(view, sceneState) {
  const { ctx, worldW } = view;
  const y = sceneState.front_line_y;
  const [x0, y0] = worldToCanvas(view, [0, y]);
  const [x1] = worldToCanvas(view, [worldW, y]);
  ctx.save();
  ctx.strokeStyle = THEME.goldLight;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([9, 7]);
  ctx.globalAlpha = 0.75;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y0);
  ctx.stroke();
  ctx.restore();
}

// The team's own home half (always known, never fogged) as a {y, h} row
// range in fog-canvas pixel space; the complement is the fogged/foreign half.
function homeRowRange(team, sceneState, canvasHeight) {
  const worldH = sceneState.world_size[1];
  const homeBelow = teamHomeIsBelowLine(team, sceneState);
  const lineFogY = (worldH - sceneState.front_line_y) * FOG_PX_PER_UNIT;
  return homeBelow ? { y: lineFogY, h: canvasHeight - lineFogY } : { y: 0, h: lineFogY };
}

function foreignRowRange(team, sceneState, canvasHeight) {
  const home = homeRowRange(team, sceneState, canvasHeight);
  return home.y === 0 ? { y: home.h, h: canvasHeight - home.h } : { y: 0, h: home.y };
}

function ensureFogCanvas(team, sceneState) {
  const [worldW, worldH] = sceneState.world_size;
  const existing = fogCanvases[team];
  if (existing && existing.worldW === worldW && existing.worldH === worldH) {
    return existing;
  }

  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(worldW * FOG_PX_PER_UNIT);
  canvas.height = Math.ceil(worldH * FOG_PX_PER_UNIT);
  const ctx = canvas.getContext("2d");
  // Fully opaque: this canvas doubles as a binary "undiscovered" mask (used
  // to completely erase buildings/zones from unexplored ground, not just
  // dim them) as well as the dark visual tint painted over that ground -
  // drawFog() applies its own reduced alpha for the tint.
  ctx.fillStyle = "rgb(6,7,8)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Cut the fog away entirely over the team's own home half - that
  // territory is always known, never fogged.
  const home = homeRowRange(team, sceneState, canvas.height);
  ctx.clearRect(0, home.y, canvas.width, home.h);

  const entry = { canvas, ctx, worldW, worldH };
  fogCanvases[team] = entry;
  return entry;
}

function fogToCanvas(worldH, point) {
  return [point[0] * FOG_PX_PER_UNIT, (worldH - point[1]) * FOG_PX_PER_UNIT];
}

function updateFog(team, sceneState) {
  const fog = ensureFogCanvas(team, sceneState);
  const visionRadiusKey = team === "blue" ? "detection_radius" : "scouting_radius";
  for (const agent of Object.values(sceneState.agents)) {
    if (agent.team !== team || !agent.alive) continue;
    const radius = agent[visionRadiusKey];
    if (!radius) continue;
    const [x, y] = fogToCanvas(fog.worldH, agent.position);
    fog.ctx.save();
    fog.ctx.globalCompositeOperation = "destination-out";
    fog.ctx.beginPath();
    fog.ctx.arc(x, y, radius * FOG_PX_PER_UNIT, 0, Math.PI * 2);
    fog.ctx.fill();
    fog.ctx.restore();
  }
}

function drawFog(view, team, sceneState) {
  const fog = ensureFogCanvas(team, sceneState);
  const { ctx, offsetX, offsetY, worldW, worldH, scale } = view;
  ctx.save();
  // Fully opaque - unseen-so-far territory is solid black, not a dim tint
  // over visible terrain. maskToRevealed already fully erases buildings/zones
  // there; this is the ground layer's half of the same "never seen" state.
  ctx.globalAlpha = 1;
  ctx.drawImage(fog.canvas, 0, 0, fog.canvas.width, fog.canvas.height, offsetX, offsetY, worldW * scale, worldH * scale);
  ctx.restore();
}

// Reusable offscreen layer for masking map features out of undiscovered
// territory before compositing them onto the scene.
let scratchCanvas = null;
function getScratchCanvas(width, height) {
  if (!scratchCanvas) scratchCanvas = document.createElement("canvas");
  if (scratchCanvas.width !== width || scratchCanvas.height !== height) {
    scratchCanvas.width = width;
    scratchCanvas.height = height;
  }
  const ctx = scratchCanvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  return { canvas: scratchCanvas, ctx };
}

// Erases anything drawn on `ctx` wherever `team` hasn't discovered the
// ground yet - full erasure, not a partial dim, so nothing bleeds through.
function maskToRevealed(ctx, view, team, sceneState) {
  const fog = ensureFogCanvas(team, sceneState);
  const { offsetX, offsetY, worldW, worldH, scale } = view;
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(fog.canvas, 0, 0, fog.canvas.width, fog.canvas.height, offsetX, offsetY, worldW * scale, worldH * scale);
  ctx.restore();
}

// Fraction of the opponent's territory a team has scouted (permanently
// revealed) so far - reads the fog canvas alpha rather than tracking area
// analytically, since revealed vision circles overlap unpredictably.
function computeScoutedFraction(team, sceneState) {
  const fog = fogCanvases[team];
  if (!fog) return 0;
  const region = foreignRowRange(team, sceneState, fog.canvas.height);
  if (region.h <= 0) return 0;
  const data = fog.ctx.getImageData(0, region.y, fog.canvas.width, region.h).data;
  let revealed = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 128) revealed += 1;
  }
  return revealed / (fog.canvas.width * region.h);
}

// getImageData is too costly to run every render; cache and refresh a few
// times a second instead.
let scoutedFractionCache = { value: 0, ts: 0 };
function getScoutedFraction(sceneState) {
  const now = performance.now();
  if (now - scoutedFractionCache.ts > 200) {
    scoutedFractionCache = { value: computeScoutedFraction("red", sceneState), ts: now };
  }
  return scoutedFractionCache.value;
}

function drawRect(view, rect, fill, stroke, width, touching = EMPTY_TOUCHING) {
  const { ctx, scale } = view;
  const [x, y] = worldToCanvas(view, [rect.x, rect.y + rect.h]);
  const w = rect.w * scale;
  const h = rect.h * scale;
  ctx.save();
  ctx.fillStyle = fill;
  ctx.fillRect(x, y, w, h);

  // Only stroke the parts of each edge NOT covered by a flush-adjacent
  // neighbor (see touchingIntervals below), segment by segment - a whole-side
  // skip is wrong whenever a neighbor only partially covers an edge (mixed
  // footprint sizes make that the common case), since the rest of that edge
  // still faces open ground and needs its border.
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (const [lo, hi] of subtractIntervals(rect.x, rect.x + rect.w, touching.north)) {
    const sx = x + (lo - rect.x) * scale;
    const ex = x + (hi - rect.x) * scale;
    ctx.moveTo(sx, y);
    ctx.lineTo(ex, y);
  }
  for (const [lo, hi] of subtractIntervals(rect.x, rect.x + rect.w, touching.south)) {
    const sx = x + (lo - rect.x) * scale;
    const ex = x + (hi - rect.x) * scale;
    ctx.moveTo(sx, y + h);
    ctx.lineTo(ex, y + h);
  }
  for (const [lo, hi] of subtractIntervals(rect.y, rect.y + rect.h, touching.west)) {
    const yLo = y + h - (lo - rect.y) * scale;
    const yHi = y + h - (hi - rect.y) * scale;
    ctx.moveTo(x, yHi);
    ctx.lineTo(x, yLo);
  }
  for (const [lo, hi] of subtractIntervals(rect.y, rect.y + rect.h, touching.east)) {
    const yLo = y + h - (lo - rect.y) * scale;
    const yHi = y + h - (hi - rect.y) * scale;
    ctx.moveTo(x + w, yHi);
    ctx.lineTo(x + w, yLo);
  }
  ctx.stroke();
  ctx.restore();
}

const EMPTY_TOUCHING = { north: [], south: [], west: [], east: [] };

// For each of rect's four world-space edges, the [lo, hi] sub-intervals
// (along that edge) that sit flush against another rect's edge with zero
// gap - used to stroke only the leftover, ground-facing portions.
function touchingIntervals(rect, buildings) {
  const result = { north: [], south: [], west: [], east: [] };
  const EPS = 1e-6;
  for (const other of buildings) {
    if (other === rect || other.shape === "circle") continue;
    if (Math.abs(rect.x - (other.x + other.w)) < EPS) {
      pushOverlap(result.west, rect.y, rect.y + rect.h, other.y, other.y + other.h);
    }
    if (Math.abs(rect.x + rect.w - other.x) < EPS) {
      pushOverlap(result.east, rect.y, rect.y + rect.h, other.y, other.y + other.h);
    }
    if (Math.abs(rect.y - (other.y + other.h)) < EPS) {
      pushOverlap(result.south, rect.x, rect.x + rect.w, other.x, other.x + other.w);
    }
    if (Math.abs(rect.y + rect.h - other.y) < EPS) {
      pushOverlap(result.north, rect.x, rect.x + rect.w, other.x, other.x + other.w);
    }
  }
  return result;
}

function pushOverlap(list, aLo, aHi, bLo, bHi) {
  const lo = Math.max(aLo, bLo);
  const hi = Math.min(aHi, bHi);
  if (hi > lo + 1e-6) list.push([lo, hi]);
}

// Gaps in [fullLo, fullHi] not covered by any interval in `covered`.
function subtractIntervals(fullLo, fullHi, covered) {
  if (covered.length === 0) return [[fullLo, fullHi]];
  const sorted = covered.slice().sort((a, b) => a[0] - b[0]);
  const gaps = [];
  let cursor = fullLo;
  for (const [lo, hi] of sorted) {
    if (lo > cursor) gaps.push([cursor, Math.min(lo, fullHi)]);
    cursor = Math.max(cursor, hi);
  }
  if (cursor < fullHi) gaps.push([cursor, fullHi]);
  return gaps;
}

// In realistic mode, buildings render as rooftops (a shingled gradient with
// a ridge line down the long axis) instead of flat gray blocks, since the
// view is top-down and a roof — not a wall — is what's actually visible.
// In realistic mode, a building is a pitched roof seen from above: two
// slopes meeting at a ridge (shaded differently since one catches more
// light), shingle courses running down each slope, and a chimney whose
// position is hashed from the building's own coordinates so it stays put
// across re-renders instead of jittering every frame.
function drawBuilding(view, building, realistic) {
  if (building.shape === "circle") {
    drawCircularBuilding(view, building, realistic);
    return;
  }
  drawRectBuilding(view, building, realistic, view.buildings ?? []);
}

// Round obstacles (tanks, silos, planters, rotundas) get their own draw path
// instead of being squeezed through the rectangle roof renderer below - a
// dome reads very differently from a pitched roof even in flat/top-down mode.
function drawCircularBuilding(view, circle, realistic) {
  const { ctx, scale } = view;
  const [cx, cy] = worldToCanvas(view, circle.center);
  const r = circle.radius * scale;

  if (!realistic) {
    drawCircle(view, circle.center, circle.radius, "#59564e", "#b7b0a1", 1.2);
    return;
  }

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx + r * 0.06, cy + r * 0.1, r, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fill();

  const gradient = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.35, r * 0.1, cx, cy, r);
  gradient.addColorStop(0, "#9a9186");
  gradient.addColorStop(1, "#5c5648");
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = "rgba(0,0,0,0.22)";
  ctx.lineWidth = Math.max(0.6, r * 0.035);
  for (let ring = 0.35; ring < 1; ring += 0.28) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * ring, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.strokeStyle = "#3f3122";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  const hatchRadius = Math.max(1.2, r * 0.14);
  const seed = hash2(circle.center[0] * 1.7, circle.center[1] * 2.3);
  const angle = seed * Math.PI * 2;
  const hx = cx + Math.cos(angle) * r * 0.3;
  const hy = cy + Math.sin(angle) * r * 0.3;
  ctx.fillStyle = "#2c241a";
  ctx.beginPath();
  ctx.arc(hx, hy, hatchRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Buildings are plain Rect obstacles with no per-type art to hang a
// "realistic roof" on, so rectangular buildings always render as a bordered
// grey plane (matching the industrial_rooftop reference look) regardless of
// the realistic/flat terrain toggle - that toggle only affects the ground
// texture now. touchingIntervals keeps two flush-adjacent buildings reading
// as one continuous surface instead of a doubled seam, without dropping the
// border on the rest of an edge when a neighbor only partially covers it.
function drawRectBuilding(view, rect, realistic, buildings) {
  drawRect(view, rect, "#59564e", "#b7b0a1", 1.2, touchingIntervals(rect, buildings));
}

function drawCircle(view, center, radius, fill, stroke, width) {
  const { ctx, scale } = view;
  const [x, y] = worldToCanvas(view, center);
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius * scale, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawLine(view, start, end, color, width) {
  const { ctx } = view;
  const [sx, sy] = worldToCanvas(view, start);
  const [ex, ey] = worldToCanvas(view, end);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.restore();
}

function drawTrail(view, trail, rgb) {
  for (let i = 1; i < trail.length; i += 1) {
    const alpha = 0.08 + 0.28 * (i / trail.length);
    drawLine(view, trail[i - 1], trail[i], rgba(rgb, alpha), 1.2);
  }
}

// Colors a tethered red scout shifts toward as exposure builds, at full
// exposure (about to be destroyed).
const ALARM_RGB = [255, 45, 35];

function drawAgent(view, agentId, agent, highlighted = false, teamPerspective = null) {
  if (!agent.alive) {
    drawDeathFade(view, agentId, agent);
    return;
  }
  const teamRgb = agent.team === "blue" ? THEME.blueRgb : THEME.redRgb;
  let color = agent.team === "blue" ? THEME.blue : THEME.red;
  let drawPosition = agent.position;

  const exposureFrac = agent.exposure_frac ?? 0;
  if (agent.team === "red" && exposureFrac > 0) {
    const mixed = teamRgb.map((component, index) =>
      Math.round(component + (ALARM_RGB[index] - component) * exposureFrac),
    );
    color = rgba(mixed, 1);
    const seed = hashStringSeed(agentId);
    const t = view.animationTime * 0.045;
    const shakeMag = exposureFrac * exposureFrac * 1.4;
    drawPosition = [
      agent.position[0] + Math.sin(t * 9 + seed) * shakeMag,
      agent.position[1] + Math.cos(t * 11 + seed * 1.7) * shakeMag,
    ];
  }

  // In a team-perspective view you only know your own vehicles' sensor reach -
  // an enemy vehicle's vision radius is drawn only in the global (no-perspective) view.
  const showRing = !teamPerspective || teamPerspective === agent.team;
  const soft = rgba(teamRgb, 0.1);
  if (showRing && agent.team === "blue") {
    drawRing(view, agent.position, agent.detection_radius, soft, rgba(THEME.blueRgb, 0.22), 1);
  } else if (showRing) {
    drawRing(view, agent.position, agent.scouting_radius, soft, rgba(THEME.redRgb, 0.22), 1);
  }
  drawScoutIcon(view, drawPosition, agent.heading, agent.radius, color);
  if (highlighted) drawRing(view, agent.position, agent.radius * 3.8, THEME.transparent, rgba(THEME.goldRgb, 0.95), 2.4);
  drawLabel(view, agentId, drawPosition, color);
}

function hashStringSeed(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) % 1000;
  }
  return hash;
}

function drawDeathFade(view, agentId, agent) {
  const fade = deathFades.get(agentId);
  if (!fade) return;
  const elapsed = view.animationTime - fade.start;
  if (elapsed > DEATH_FADE_MS) {
    deathFades.delete(agentId);
    return;
  }
  const color = agent.team === "blue" ? THEME.blue : THEME.red;
  const { ctx } = view;
  ctx.save();
  ctx.globalAlpha = 1 - elapsed / DEATH_FADE_MS;
  drawScoutIcon(view, fade.position, fade.heading, agent.radius, color);
  ctx.restore();
}

function drawRing(view, center, radius, fill, stroke, width) {
  const { ctx } = view;
  const polygon = visibleRangePolygon(view, center, radius);
  if (polygon.length < 3) return;

  ctx.save();
  clipToWorld(view);
  ctx.beginPath();
  polygon.forEach((point, index) => {
    const [x, y] = worldToCanvas(view, point);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function visibleRangePolygon(view, center, radius) {
  const angles = new Set();
  const epsilon = 0.0008;
  const sampleCount = 128;

  for (let i = 0; i < sampleCount; i += 1) {
    angles.add(normalizeAngle((Math.PI * 2 * i) / sampleCount));
  }

  for (const point of visibilityCornerPoints(view)) {
    const angle = Math.atan2(point[1] - center[1], point[0] - center[0]);
    angles.add(normalizeAngle(angle - epsilon));
    angles.add(normalizeAngle(angle));
    angles.add(normalizeAngle(angle + epsilon));
  }

  return Array.from(angles)
    .sort((a, b) => a - b)
    .map((angle) => castVisibilityRay(view, center, radius, angle));
}

function visibilityCornerPoints(view) {
  const points = [
    [0, 0],
    [view.worldW, 0],
    [view.worldW, view.worldH],
    [0, view.worldH],
  ];
  for (const building of view.buildings ?? []) {
    if (building.shape === "circle") {
      points.push(...circlePoints(building, 12));
      continue;
    }
    points.push(
      [building.x, building.y],
      [building.x + building.w, building.y],
      [building.x + building.w, building.y + building.h],
      [building.x, building.y + building.h],
    );
  }
  return points;
}

function castVisibilityRay(view, center, radius, angle) {
  const direction = [Math.cos(angle), Math.sin(angle)];
  let nearestDistance = radius;

  for (const edge of worldEdges(view)) {
    const distance = raySegmentDistance(center, direction, edge[0], edge[1]);
    if (distance !== null && distance < nearestDistance) nearestDistance = distance;
  }

  for (const building of view.buildings ?? []) {
    const edges = building.shape === "circle" ? circleEdges(building, 24) : rectEdges(building);
    for (const edge of edges) {
      const distance = raySegmentDistance(center, direction, edge[0], edge[1]);
      if (distance !== null && distance < nearestDistance) nearestDistance = distance;
    }
  }

  return [
    center[0] + direction[0] * nearestDistance,
    center[1] + direction[1] * nearestDistance,
  ];
}

function worldEdges(view) {
  return [
    [[0, 0], [view.worldW, 0]],
    [[view.worldW, 0], [view.worldW, view.worldH]],
    [[view.worldW, view.worldH], [0, view.worldH]],
    [[0, view.worldH], [0, 0]],
  ];
}

function rectEdges(rect) {
  const minX = rect.x;
  const maxX = rect.x + rect.w;
  const minY = rect.y;
  const maxY = rect.y + rect.h;
  return [
    [[minX, minY], [maxX, minY]],
    [[maxX, minY], [maxX, maxY]],
    [[maxX, maxY], [minX, maxY]],
    [[minX, maxY], [minX, minY]],
  ];
}

function circlePoints(circle, segments = 12) {
  const [cx, cy] = circle.center;
  const r = circle.radius;
  const points = [];
  for (let i = 0; i < segments; i += 1) {
    const angle = (Math.PI * 2 * i) / segments;
    points.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
  }
  return points;
}

// Ray-vs-circle intersection is exact, but approximating the boundary as a
// polygon lets circular obstacles reuse the same ray/segment machinery the
// rest of visibleRangePolygon already relies on for rectangles.
function circleEdges(circle, segments = 24) {
  const points = circlePoints(circle, segments);
  const edges = [];
  for (let i = 0; i < points.length; i += 1) {
    edges.push([points[i], points[(i + 1) % points.length]]);
  }
  return edges;
}

function raySegmentDistance(origin, direction, start, end) {
  const segment = [end[0] - start[0], end[1] - start[1]];
  const denom = cross(direction, segment);
  if (Math.abs(denom) < 1e-9) return null;

  const delta = [start[0] - origin[0], start[1] - origin[1]];
  const rayDistance = cross(delta, segment) / denom;
  const segmentFraction = cross(delta, direction) / denom;
  if (rayDistance < 0 || segmentFraction < -1e-7 || segmentFraction > 1 + 1e-7) return null;
  return rayDistance;
}

function cross(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function normalizeAngle(angle) {
  const twoPi = Math.PI * 2;
  return ((angle % twoPi) + twoPi) % twoPi;
}

function clipToWorld(view) {
  const { ctx, offsetX, offsetY, scale, worldW, worldH } = view;
  ctx.beginPath();
  ctx.rect(offsetX, offsetY, worldW * scale, worldH * scale);
  ctx.clip();
}

// Hand-drawn top-down ground vehicle icon: rounded body, four wheel ticks,
// and a windshield wedge marking the front.
function drawScoutIcon(view, position, heading, radius, color) {
  const { ctx, scale } = view;
  const [x, y] = worldToCanvas(view, position);
  const size = Math.max(14, radius * scale * 4.3);
  const angle = -heading;
  const dark = THEME.dark;
  const light = THEME.light;

  const bodyLength = size * 0.95;
  const bodyWidth = size * 0.56;
  const bodyRadius = Math.min(bodyWidth, bodyLength) * 0.28;
  const wheelLength = bodyLength * 0.32;
  const wheelWidth = Math.max(1.4, size * 0.14);
  const wheelInset = bodyWidth * 0.5;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Four wheel ticks, top-down, at the body corners.
  ctx.fillStyle = rgba(THEME.darkRgb, 0.85);
  for (const side of [-1, 1]) {
    for (const front of [-1, 1]) {
      ctx.save();
      ctx.translate(front * bodyLength * 0.32, side * wheelInset);
      ctx.fillRect(-wheelLength / 2, -wheelWidth / 2, wheelLength, wheelWidth);
      ctx.restore();
    }
  }

  ctx.beginPath();
  ctx.roundRect(-bodyLength / 2, -bodyWidth / 2, bodyLength, bodyWidth, bodyRadius);
  ctx.fillStyle = color;
  ctx.strokeStyle = light;
  ctx.lineWidth = Math.max(0.8, size * 0.09);
  ctx.fill();
  ctx.stroke();

  // Windshield wedge marks the front of the vehicle.
  ctx.beginPath();
  ctx.moveTo(bodyLength * 0.46, 0);
  ctx.lineTo(bodyLength * 0.12, bodyWidth * 0.28);
  ctx.lineTo(bodyLength * 0.12, -bodyWidth * 0.28);
  ctx.closePath();
  ctx.fillStyle = light;
  ctx.globalAlpha = 0.72;
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.beginPath();
  ctx.arc(0, 0, size * 0.1, 0, Math.PI * 2);
  ctx.fillStyle = dark;
  ctx.strokeStyle = light;
  ctx.lineWidth = 0.55;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawTerrain(view, sceneState, worldW, worldH) {
  if (!sceneState.terrain?.enabled) return;
  const { ctx, scale, offsetX, offsetY } = view;
  const width = worldW * scale;
  const height = worldH * scale;
  const url = sceneState.terrain.url;
  if (url && terrainImageUrl !== url) {
    terrainImageUrl = url;
    terrainImage = new Image();
    terrainImage.src = url;
  }
  ctx.save();
  if (url && terrainImage.complete && terrainImage.naturalWidth > 0) {
    ctx.globalAlpha = 0.72;
    ctx.drawImage(terrainImage, offsetX, offsetY, width, height);
  } else {
    // No custom terrain image uploaded for this instance - render an actual
    // dirt texture instead of a flat placeholder gradient. Flat/conceptual
    // mode doesn't go through this path at all (drawTerrain no-ops when
    // terrain isn't enabled), so it's untouched by this.
    drawDirtTexture(view, worldW, worldH);
  }
  ctx.restore();
}

// The actual uploaded ground sprites, tiled across the world: grass
// everywhere, except a one-cell dirt halo around every building (a cleared
// lot, not a building sitting in the middle of a lawn). Loaded once at
// module scope, same pattern as the per-instance terrain image above.
function loadGroundSprite(path) {
  const state = { image: new Image(), loaded: false, tileCache: { px: 0, canvas: null } };
  state.image.onload = () => {
    state.loaded = true;
  };
  state.image.src = path;
  return state;
}

const dirtGround = loadGroundSprite("/static/assets/ground/dirt_dark.png");
const grassGround = loadGroundSprite("/static/assets/ground/grass.png");

// createPattern repeats a source image at its own native pixel size, which
// doesn't track the view's world-to-canvas scale - so a tile canvas is
// pre-rendered at the current scale's pixel size and re-cached only when
// that pixel size actually changes (e.g. on window resize), not every frame.
function getGroundTile(sprite, tilePx) {
  const rounded = Math.max(4, Math.round(tilePx));
  if (sprite.tileCache.px === rounded && sprite.tileCache.canvas) {
    return sprite.tileCache.canvas;
  }
  const canvas = document.createElement("canvas");
  canvas.width = rounded;
  canvas.height = rounded;
  canvas.getContext("2d").drawImage(sprite.image, 0, 0, rounded, rounded);
  sprite.tileCache = { px: rounded, canvas };
  return canvas;
}

function fillGroundPattern(view, sprite, fallbackColor, region) {
  const { ctx, scale } = view;
  if (!sprite.loaded) {
    ctx.fillStyle = fallbackColor;
  } else {
    const tile = getGroundTile(sprite, GROUND_CELL_WORLD_SIZE * scale);
    ctx.fillStyle = ctx.createPattern(tile, "repeat");
  }
  ctx.fillRect(region.x, region.y, region.w, region.h);
}

function drawDirtTexture(view, worldW, worldH) {
  const { ctx, scale, offsetX, offsetY, buildings } = view;
  ctx.save();
  ctx.translate(offsetX, offsetY);
  fillGroundPattern(view, grassGround, THEME.dirtFloor, { x: 0, y: 0, w: worldW * scale, h: worldH * scale });

  // Dirt halo: each building's footprint expanded by one ground cell on
  // every side (the building sprite drawn afterward covers the footprint
  // interior, leaving just the one-cell ring of dirt visible around it).
  const margin = GROUND_CELL_WORLD_SIZE;
  for (const building of buildings ?? []) {
    if (building.shape === "circle") {
      continue; // circular obstacles aren't produced by the current generator; skip for now
    }
    const hx = (building.x - margin) * scale;
    const hy = (worldH - (building.y + building.h + margin)) * scale;
    const hw = (building.w + margin * 2) * scale;
    const hh = (building.h + margin * 2) * scale;
    fillGroundPattern(view, dirtGround, THEME.dirtFloor, { x: hx, y: hy, w: hw, h: hh });
  }
  ctx.restore();
}

function drawLabel(view, text, position, color) {
  const { ctx } = view;
  const [x, y] = worldToCanvas(view, position);
  ctx.save();
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.fillText(text, x + 8, y - 8);
  ctx.restore();
}

// --- Metrics panel -----------------------------------------------------------

function renderMetrics() {
  metrics.innerHTML = "";
  addMetric("Step", `${state.step} (${(state.step * state.dt).toFixed(1)}s)`);
  addMetric("Detections", state.detections.length);
  addMetric("Blue Territory Scouted", `${Math.round(getScoutedFraction(state) * 100)}%`);
}

function addMetric(label, value) {
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value;
  metrics.append(dt, dd);
}
