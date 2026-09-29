const sprites = {};
const spriteNames = ['ant', 'spider', 'sugar', 'seed', 'anthill', 'toile', 'pond', 'rock_background'];
spriteNames.forEach(name => {
  const img = new Image();
  img.src = `/assets/${name}.png`;
  img.onload = () => { sprites[name] = img; };
});

const bgDirt = new Image();
bgDirt.src = '/assets/background.jpg';
bgDirt.onload = () => { sprites['background'] = bgDirt; };

let canvas, ctx;
let camera = { x: 0, y: 0, zoom: 1 };
let cachedDirtGrad = null;
let cachedRockGrad = null;
let cachedWorldSize = null;
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let cachedZonePath = null;

window.Renderer = {
  init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    
    // Resize handling
    const resize = () => {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', resize);
    resize();
  },

  render(snap, camState, showSensors, pureMode = false) {
    if (camState) camera = camState;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    // Camera transform
    ctx.translate(canvas.width / 2 + camera.x, canvas.height / 2 + camera.y);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-snap.world.width / 2, -snap.world.height / 2);

    if (pureMode) {
      if (window.ZONE_BOUNDARY_MAP) {
        // Draw right background (rock) as full base layer to prevent any anti-aliasing hairline gaps
        ctx.fillStyle = '#1c1f26';
        ctx.fillRect(0, 0, snap.world.width, snap.world.height);
        
        // Draw left background (dirt) on top using the jagged boundary
        ctx.fillStyle = '#2a1f14';
        
        if (!cachedZonePath) {
          cachedZonePath = new Path2D();
          cachedZonePath.moveTo(0, 0);
          for (let y = 0; y < window.ZONE_BOUNDARY_MAP.length; y++) {
            cachedZonePath.lineTo(window.ZONE_BOUNDARY_MAP[y], y);
          }
          cachedZonePath.lineTo(0, snap.world.height);
          cachedZonePath.closePath();
        }
        
        ctx.fill(cachedZonePath);
      } else {
        // Fallback if not loaded
        ctx.fillStyle = '#2a1f14';
        ctx.fillRect(0, 0, snap.world.width / 2, snap.world.height);
        ctx.fillStyle = '#1c1f26';
        ctx.fillRect(snap.world.width / 2, 0, snap.world.width / 2, snap.world.height);
      }
    } else {
      // Draw left background (dirt)
      if (sprites['background']) {
        ctx.drawImage(sprites['background'], 0, 0, snap.world.width, snap.world.height);
        // Darken the dirt background slightly for better contrast
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, snap.world.width, snap.world.height);
      } else {
        ctx.fillStyle = '#2a1f14';
        ctx.fillRect(0, 0, snap.world.width, snap.world.height);
      }

      // Draw right background (rock)
      if (sprites['rock_background']) {
        // The rock background has transparency for the dirt side, so we draw it over the entire world size
        ctx.save();
        //Darken the rock image by 80% (brightness 20%) to match the rgba(0, 0, 0, 0.8) intent
        ctx.filter = 'brightness(40%)';
        ctx.drawImage(sprites['rock_background'], 0, 0, snap.world.width, snap.world.height);
        ctx.restore();
      } else {
        // Fallback
        ctx.fillStyle = '#1c1f26';
        ctx.fillRect(snap.world.width / 2, 0, snap.world.width / 2, snap.world.height);
      }
    }

    // Pheromones
    if (snap.pheromones && snap.pheromones.data) {
      snap.pheromones.data.forEach(([gx, gy, strength]) => {
        const cs = snap.pheromones.cellSize;
        const alpha = Math.min(1.0, strength * 1.8);
        ctx.fillStyle = `rgba(255, 255, 138, ${alpha})`;
        ctx.fillRect(gx * cs, gy * cs, cs, cs);
      });
    }

    // Lakes
    if (snap.lakes) {
      snap.lakes.forEach(l => {
        if (!pureMode && sprites['pond']) {
          ctx.save();
          ctx.translate(l.x, l.y);
          ctx.drawImage(sprites['pond'], -l.radius, -l.radius, l.radius * 2, l.radius * 2);
          ctx.restore();
        } else {
          ctx.fillStyle = "#1e3a8a";
          ctx.beginPath();
          ctx.arc(l.x, l.y, l.radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#3b82f6";
          ctx.stroke();
        }
      });
    }

    // Kingdoms (using sprites)
    if (snap.kingdoms) {
      snap.kingdoms.forEach(k => {
        ctx.save();
        ctx.translate(k.x, k.y);
        const sprite = k.species === "Ant" ? sprites['anthill'] : sprites['toile'];
        if (sprite) {
          ctx.drawImage(sprite, -k.spawnRadius, -k.spawnRadius, k.spawnRadius * 2, k.spawnRadius * 2);
        } else {
          ctx.fillStyle = k.species === "Ant" ? "rgba(111, 184, 122, 0.2)" : "rgba(201, 74, 74, 0.2)";
          ctx.beginPath();
          ctx.arc(0, 0, k.spawnRadius, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      });
    }

    // Food (batched — no save/restore)
    if (snap.food) {
      snap.food.forEach(f => {
        if (f.carried) return;
        const sprite = f.type === "sugar" ? sprites['sugar'] : sprites['seed'];
        if (sprite) {
          ctx.drawImage(sprite, f.x - 4, f.y - 4, 8, 8);
        } else {
          ctx.fillStyle = f.type === "sugar" ? "#38bdf8" : "#f59e0b";
          ctx.beginPath();
          ctx.arc(f.x, f.y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    // Number of sensors (global constant)
    let numSensors = 8;
    if (window.getConstant) {
       numSensors = window.getConstant('NN_NUM_SENSORS') || 8;
    }

    // Creatures
    if (snap.creatures) {
      for (const [spName, list] of Object.entries(snap.creatures)) {
        const sprite = spName === "Ant" ? sprites['ant'] : sprites['spider'];
        const topIndices = snap.topFit ? (snap.topFit[spName] || []) : [];

        list.forEach((c, idx) => {
          if (!c.alive) return;
          ctx.save();
          ctx.translate(c.x, c.y);

          // Top Fit Halo
          if (topIndices.includes(idx)) {
            ctx.strokeStyle = "#c4a35a";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(0, 0, c.radius + 2, 0, Math.PI * 2);
            ctx.stroke();
          }

          // Sensor Rays
          if (showSensors) {
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = 1;
            const rayLength = c.visionRange || (spName === "Ant" ? 180 : 120);
            const halfFovRad = ((c.fov || 160) / 2) * (Math.PI / 180);
            const startAngle = c.dir - halfFovRad;
            const step = (halfFovRad * 2) / Math.max(1, numSensors - 1);

            for (let i = 0; i < numSensors; i++) {
              const a = startAngle + i * step;
              ctx.beginPath();
              ctx.moveTo(0, 0);
              ctx.lineTo(Math.cos(a) * rayLength, Math.sin(a) * rayLength);
              ctx.stroke();
            }
          }

          // Actions
          if (c.attacking) {
            ctx.shadowColor = 'rgba(255,50,50,0.3)';
            ctx.shadowBlur = 10;
          } else if (c.eating) {
            ctx.shadowColor = 'rgba(255,255,255,0.3)';
            ctx.shadowBlur = 8;
          }

          ctx.rotate(c.dir);
          
          if (sprite) {
            ctx.save();
            // Rotate sprite by 90 degrees (if native asset faces UP)
            // to align it with the walking direction (+X axis)
            ctx.rotate(Math.PI / 2);
            ctx.drawImage(sprite, -c.radius, -c.radius, c.radius * 2, c.radius * 2);
            ctx.restore();
          } else {
            ctx.fillStyle = spName === "Ant" ? "#6fb87a" : "#c94a4a";
            ctx.beginPath();
            ctx.arc(0, 0, c.radius, 0, Math.PI * 2);
            ctx.fill();
            // Heading line
            ctx.strokeStyle = "#ffffff";
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(c.radius + 3, 0);
            ctx.stroke();
          }

          // Carrying
          if (c.carrying && c.carriedType) {
            const carrySprite = c.carriedType === "sugar" ? sprites['sugar'] : sprites['seed'];
            if (carrySprite) {
              // Draw carried object at the creature's head/mandibles
              ctx.drawImage(carrySprite, c.radius - 2, -3, 6, 6);
            }
          }
          
          ctx.restore();
          
          ctx.save();
          ctx.translate(c.x, c.y);
          // Health Bar
          if (c.hp < c.maxHp) {
            const hpRatio = Math.max(0, c.hp / c.maxHp);
            const w = c.radius * 2;
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(-w/2, -c.radius - 3, w, 2);
            
            if (hpRatio > 0.5) ctx.fillStyle = '#6fb87a';
            else if (hpRatio > 0.25) ctx.fillStyle = '#c4a35a';
            else ctx.fillStyle = '#c94a4a';
            
            ctx.fillRect(-w/2, -c.radius - 3, w * hpRatio, 2);
          }
          ctx.restore();
        });
      }
    }

    ctx.restore();
  },

  zoomIn() {
    camera.zoom = Math.min(3.0, camera.zoom * 1.15);
  },

  zoomOut() {
    camera.zoom = Math.max(0.5, camera.zoom / 1.15);
  },

  getZoomPercent() {
    return Math.round(camera.zoom * 100) + '%';
  },

  startDrag(e) {
    isDragging = true;
    dragStart = { x: e.clientX - camera.x, y: e.clientY - camera.y };
  },

  moveDrag(e) {
    if (isDragging) {
      camera.x = e.clientX - dragStart.x;
      camera.y = e.clientY - dragStart.y;
    }
  },

  endDrag() {
    isDragging = false;
  },
  
  getCamera() {
    return camera;
  }
};
