const sprites = {};
const spriteNames = ['ant', 'spider', 'sugar', 'seed', 'anthill', 'toile'];
spriteNames.forEach(name => {
  const img = new Image();
  img.src = `/assets/${name}.png`;
  img.onload = () => { sprites[name] = img; };
});

let canvas, ctx;
let camera = { x: 0, y: 0, zoom: 1 };
let isDragging = false;
let dragStart = { x: 0, y: 0 };

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

  render(snap, camState, showSensors) {
    if (camState) camera = camState;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    // Camera transform
    ctx.translate(canvas.width / 2 + camera.x, canvas.height / 2 + camera.y);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-snap.world.width / 2, -snap.world.height / 2);

    // World background (Gradient)
    const dirtGrad = ctx.createLinearGradient(0, 0, snap.world.width / 2, 0);
    dirtGrad.addColorStop(0, '#3d2b1f');
    dirtGrad.addColorStop(1, '#2a1f14');
    
    const rockGrad = ctx.createLinearGradient(snap.world.width / 2, 0, snap.world.width, 0);
    rockGrad.addColorStop(0, '#2a2d32');
    rockGrad.addColorStop(1, '#1c1f26');

    ctx.fillStyle = dirtGrad;
    ctx.fillRect(0, 0, snap.world.width / 2, snap.world.height);
    ctx.fillStyle = rockGrad;
    ctx.fillRect(snap.world.width / 2, 0, snap.world.width / 2, snap.world.height);

    // Border
    ctx.strokeStyle = '#3d3228';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, snap.world.width, snap.world.height);

    // Pheromones
    if (snap.pheromones && snap.pheromones.data) {
      snap.pheromones.data.forEach(([gx, gy, strength]) => {
        const cs = snap.pheromones.cellSize;
        ctx.fillStyle = `rgba(255, 255, 138, ${Math.min(1.0, strength / 4)})`; // (strength / 4) for light color
        ctx.fillRect(gx * cs, gy * cs, cs, cs);
      });
    }

    // Lakes
    if (snap.lakes) {
      snap.lakes.forEach(l => {
        ctx.fillStyle = "#1e3a8a";
        ctx.beginPath();
        ctx.arc(l.x, l.y, l.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#3b82f6";
        ctx.stroke();
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

    // Food
    if (snap.food) {
      snap.food.forEach(f => {
        if (f.carried) return;
        ctx.save();
        ctx.translate(f.x, f.y);
        const sprite = f.type === "sugar" ? sprites['sugar'] : sprites['seed'];
        if (sprite) {
          ctx.drawImage(sprite, -4, -4, 8, 8);
        } else {
          ctx.fillStyle = f.type === "sugar" ? "#38bdf8" : "#f59e0b";
          ctx.beginPath();
          ctx.arc(0, 0, 4, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      });
    }

    // Dynamic Sensor Angle & FOV
    let sensorAngleRad = 1.396;
    let numSensors = 8;
    if (window.getConstant) {
       sensorAngleRad = window.getConstant('SENSOR_ANGLE') || 1.396;
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
            const rayLength = spName === "Ant" ? 180 : 120;
            const startAngle = c.dir - sensorAngleRad;
            const step = (sensorAngleRad * 2) / Math.max(1, numSensors - 1);

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
            ctx.drawImage(sprite, -c.radius, -c.radius, c.radius * 2, c.radius * 2);
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
              ctx.drawImage(carrySprite, -2, -2, 4, 4);
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
    camera.zoom = Math.min(5.0, camera.zoom + 0.2);
  },

  zoomOut() {
    camera.zoom = Math.max(0.3, camera.zoom - 0.2);
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
