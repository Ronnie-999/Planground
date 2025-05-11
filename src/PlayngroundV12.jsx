import React, { useRef, useState, useEffect } from "react";

/**
 * PLANGROUND — wall/column sketch-optimiser + cursor-centred wheel-zoom + Rhythmizer
 * --------------------------------------------------------
 *  • Wall / Columns(centroid) / Eraser / Move object
 *  • Optimize (robust algorithm)
 *  • Mouse-wheel zoom (view transform only; geometry unchanged)
 *  • Rhythmizer: snap grid & objects to uniform pitch
 *  • Grid toggle, Undo / Redo, Clear, Looseness slider
 */
export default function Planground() {
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  const [view, setView]         = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [tool, setTool]         = useState("wall");
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);

  const [columns, setColumns] = useState([]);   // [{x,y},…]
  const [walls,   setWalls]   = useState([]);   // [[{x,y},…],…]

  const [grid, setGrid]           = useState(null); // { xs, ys, dirA, dirB }
  const [showGrid, setShowGrid]   = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);

   // ─── wall‑simplification tweakables (from Code F) ───────────────
 const RDP_EPS      = 8;     // px – tolerance for Ramer‑Douglas‑Peucker
 const JUNC_THRESH  = 12;    // px – node snapped only if this close

 // ─── Ramer–Douglas–Peucker (2‑D) ────────────────────────────────
 const rdpIndices = (pts, eps) => {
   const keep = Array(pts.length).fill(false);
   keep[0] = keep[pts.length - 1] = true;
   const stack = [[0, pts.length - 1]];
   while (stack.length) {
     const [s, e] = stack.pop();
     const a = pts[s], b = pts[e];
     const segx = b.x - a.x, segy = b.y - a.y;
     const segLen = Math.hypot(segx, segy) || 1;
     let idx = -1, maxD = -1;
     for (let i = s + 1; i < e; i++) {
       const p = pts[i];
       const d = Math.abs((p.x - a.x) * segy - (p.y - a.y) * segx) / segLen;
       if (d > maxD) { maxD = d; idx = i; }
     }
     if (maxD > eps) {
       keep[idx] = true;
       stack.push([s, idx], [idx, e]);
     }
   }
   return keep.map((k, i) => k ? i : -1).filter(i => i !== -1);
 };



 // ─── principal‑axis finder (PCA 1‑D) ────────────────────────────
 const principalAxis = run => {
   const n = run.length;
   const mean = run.reduce((s, p) => ({ x: s.x + p.x, y: s.y + p.y }),
                           { x: 0, y: 0 });
   mean.x /= n; mean.y /= n;
   let sxx = 0, sxy = 0, syy = 0;
   run.forEach(p => {
     const dx = p.x - mean.x, dy = p.y - mean.y;
     sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
   });
   const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
   return { centre: mean, dir: { x: Math.cos(angle), y: Math.sin(angle) } };
 };

 // ─── snap a stroke to piece‑wise axes (+ optional corner node) ───
 const simplifyWall = pts => {
   if (pts.length < 3) return pts;
   const verts = rdpIndices(pts, RDP_EPS);
   
   const out   = pts.slice();                 // copy to mutate
   const axes  = [];

   for (let k = 0; k < verts.length - 1; k++) {
     const run = pts.slice(verts[k], verts[k + 1] + 1);
     if (run.length < 3) continue;
     const { centre, dir } = principalAxis(run);
     axes.push({ start: verts[k], end: verts[k + 1], centre, dir });
     // project each point of the run on its axis
     for (let i = verts[k]; i <= verts[k + 1]; i++) {
       const p  = pts[i];
       const t  = (p.x - centre.x) * dir.x + (p.y - centre.y) * dir.y;
       out[i] = { x: centre.x + t * dir.x, y: centre.y + t * dir.y };
     }
   }

   // optional junction node between consecutive runs
   const addNodes = [];
   for (let k = 0; k < axes.length - 1; k++) {
     const A = axes[k], B = axes[k + 1];
     // solve cA + s*dirA = cB + t*dirB
     const det = A.dir.x * B.dir.y - A.dir.y * B.dir.x;
     if (Math.abs(det) < 1e-6) continue;
     const dx = B.centre.x - A.centre.x;
     const dy = B.centre.y - A.centre.y;
     const s  = (dx * B.dir.y - dy * B.dir.x) / det;
     const X  = { x: A.centre.x + s * A.dir.x, y: A.centre.y + s * A.dir.y };
     const ea = out[A.end], sb = out[B.start];
     if (Math.hypot(X.x - ea.x, X.y - ea.y) < JUNC_THRESH &&
         Math.hypot(X.x - sb.x, X.y - sb.y) < JUNC_THRESH)
       addNodes.push({ idx: A.end + 0.5, pt: X });
   }
   if (addNodes.length) {
     let offset = 0;
     addNodes.sort((a, b) => a.idx - b.idx).forEach(n => {
       out.splice(Math.round(n.idx + offset), 0, n.pt);
       offset += 1;
     });
   }
   return out;
 };








  const [hasOptimized, setHasOptimized] = useState(false);

  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);

  const [moveSel, setMoveSel] = useState(null); // { kind, index, prev }

  const deepCopy = obj => JSON.parse(JSON.stringify(obj));

  // ─── history ───────────────────────────────────────────────────────────────
  const recordState = () => {
    setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
    setFuture([]);
  };
  const undo = () => {
    if (!history.length) return;
    setHistory(h => {
      const prev = h.at(-1);
      setFuture(f => [...f, deepCopy({ columns, walls, grid })]);
      setColumns(prev.columns);
      setWalls(prev.walls);
      setGrid(prev.grid);
      return h.slice(0, -1);
    });
  };
  const redo = () => {
    if (!future.length) return;
    setFuture(f => {
      const nxt = f.at(-1);
      setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
      setColumns(nxt.columns);
      setWalls(nxt.walls);
      setGrid(nxt.grid);
      return f.slice(0, -1);
    });
  };

  // ─── coordinate transforms ─────────────────────────────────────────────────
  const screenToWorld = ({ x, y }) => ({
    x: (x - view.offsetX) / view.scale,
    y: (y - view.offsetY) / view.scale
  });
  const worldToScreen = ({ x, y }) => ({
    x: x * view.scale + view.offsetX,
    y: y * view.scale + view.offsetY
  });
  const getScreenXY = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const r = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  // ─── canvas init ───────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.8;
    drawFrame(canvas.getContext("2d"), canvas);
    document.title = "PLANGROUND";

    // avoid stray <h1>
    const purge = () =>
      document.querySelectorAll("h1").forEach(el => {
        if (el.id !== "planground-title") el.remove();
      });
    purge();
    const obs = new MutationObserver(purge);
    obs.observe(document.body, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, []);

  const drawFrame = (ctx, canvas) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.lineWidth   = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  // ─── wheel-zoom ────────────────────────────────────────────────────────────
  const handleWheel = e => {
    e.preventDefault();
    const scr    = getScreenXY(e);
    const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
    const newScale = view.scale * factor;
    const { x: wx, y: wy } = screenToWorld(scr);
    setView({
      scale:   newScale,
      offsetX: scr.x - wx * newScale,
      offsetY: scr.y - wy * newScale
    });
  };

  // ─── helper from Code D: recover pitch & snapped offsets ───────────────────
  const recoverPitchAndOffsets = (offsets, tol = 0.25) => {
    const offs = [...offsets].sort((a,b) => a-b);
    if (!offs.length) return [0, []];
    const gaps = offs.slice(1).map((v,i) => v - offs[i]);
    const init = gaps.length
      ? [...gaps].sort((a,b)=>a-b)[Math.floor(gaps.length/2)]
      : 0;
    const parts = [];
    gaps.forEach(g => {
      const n = init > 0 ? Math.max(1, Math.round(g/init)) : 1;
      if (init > 0 && Math.abs(g/n - init)/init < tol) {
        for(let i=0; i<n; i++) parts.push(g/n);
      } else {
        parts.push(g);
      }
    });
    const pitch = parts.reduce((s,v)=>s+v,0) / parts.length;
    const centre = offs.reduce((s,v)=>s+v,0) / offs.length;
    const snapped = offs.map(o =>
      Math.round((o-centre)/pitch)*pitch + centre
    );
    return [pitch, snapped];
  };

  // ─── Rhythmizer: snap grid AND objects to nearest grid intersections ──────
  const rhythmize = () => {
    if (!grid) return;
    recordState();
    const { xs, ys, dirA, dirB } = grid;
    const [, snapXs] = recoverPitchAndOffsets(xs);
    const [, snapYs] = recoverPitchAndOffsets(ys);

    // build every intersection point
    const intersections = snapXs.flatMap(xv =>
      snapYs.map(yv => ({
        x: dirA.x*xv + dirB.x*yv,
        y: dirA.y*xv + dirB.y*yv
      }))
    );

    // snap each column
    const newCols = columns.map(col => {
      let best = intersections[0], bd = Infinity;
      intersections.forEach(pt => {
        const d = (pt.x-col.x)**2 + (pt.y-col.y)**2;
        if (d < bd) { bd = d; best = pt; }
      });
      return { x: best.x, y: best.y };
    });

    // snap every vertex of every wall
    const newWalls = walls.map(stk =>
      stk.map(pt => {
        let best = intersections[0], bd = Infinity;
        intersections.forEach(ip => {
          const d = (ip.x-pt.x)**2 + (ip.y-pt.y)**2;
          if (d < bd) { bd = d; best = ip; }
        });
        return { x: best.x, y: best.y };
      })
    );

    setColumns(newCols);
    setWalls(newWalls);
    setGrid({ xs: snapXs, ys: snapYs, dirA, dirB });
  };

  // ─── Optimize (identical to Code A) ────────────────────────────────────────
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;
    recordState();

    // 1) nearest-neighbour edges
    const dot  = (a,b) => a.x*b.x + a.y*b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const vec  = (p,q) => ({ x:q.x-p.x, y:q.y-p.y });
    const ang  = v => (Math.atan2(v.y, v.x) + 2*Math.PI) % Math.PI;

    const edges = pts.map((_, i) => {
      let best = { d: Infinity, j: -1 };
      for (let j=0; j<pts.length; j++) if (i!==j) {
        const d = norm(vec(pts[i], pts[j]));
        if (d < best.d) best = { d, j };
      }
      return { v: vec(pts[i], pts[best.j]) };
    });

    // 2) orientation histogram
    const bins = 180, hist = Array(bins).fill(0);
    edges.forEach(e => hist[Math.floor(ang(e.v)/Math.PI * bins)]++);
    const p1 = hist.indexOf(Math.max(...hist));
    const masked = hist.map((v,i) => {
      const d = Math.min(Math.abs(i-p1), bins - Math.abs(i-p1));
      return d <= 45 ? 0 : v;
    });
    const p2 = masked.indexOf(Math.max(...masked));

    let th1 = (p1 + 0.5)*Math.PI/bins;
    let th2 = (p2 + 0.5)*Math.PI/bins;
    const delta = ((th2 - th1 + Math.PI) % Math.PI) - Math.PI/2;
    th1 += delta/2;
    th2 -= delta/2;

    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };

    // 3) project
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));

    // 4) cluster to grid lines
    const typicalGap = arr => {
      const uniq = [...new Set(arr)].sort((a,b)=>a-b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v,i)=>v-uniq[i]).sort((a,b)=>a-b);
      const i0 = Math.floor(gaps.length*0.05);
      const i1 = Math.floor(gaps.length*0.95);
      const core = gaps.slice(i0,i1);
      return core[Math.floor(core.length/2)];
    };
    const epsX = Math.max(typicalGap(xProj)*epsFactor, 1);
    const epsY = Math.max(typicalGap(yProj)*epsFactor, 1);

    const cluster1D = (arr, eps) => {
      const srt = [...arr].sort((a,b)=>a-b);
      const out = []; let grp = [srt[0]];
      for (let i=1; i<srt.length; i++) {
        if (Math.abs(srt[i]-srt[i-1]) <= eps) grp.push(srt[i]);
        else { out.push(grp); grp=[srt[i]]; }
      }
      out.push(grp);
      return out.map(g => g.reduce((s,v)=>s+v,0)/g.length);
    };

    const xs = cluster1D(xProj, epsX);
    const ys = cluster1D(yProj, epsY);

    // 5) snap pts
    const nearest = (arr,v) => arr.reduce((p,c)=>Math.abs(c-v)<Math.abs(p-v)?c:p);
    const snapped = pts.map((p,i) => {
      const sx = nearest(xs, xProj[i]);
      const sy = nearest(ys, yProj[i]);
      return { x: dirA.x*sx + dirB.x*sy, y: dirA.y*sx + dirB.y*sy };
    });

    const newColumns = snapped.slice(0, columns.length);
    const flatWalls  = snapped.slice(columns.length);
    let k=0;
    const newWalls = walls.map(stk => {
      const seg = flatWalls.slice(k, k+stk.length);
      k += stk.length;
      return seg;
    });

    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
    setHasOptimized(true);
  };

  // ─── drawing / move / erase ───────────────────────────────────────────────
  const startDrawing = e => {
    e.preventDefault();
    const scr = getScreenXY(e);
    const wpt = screenToWorld(scr);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;
    recordState();

    if (tool === "move") {
      const pick = pickObject(scr);
      if (!pick) return;
      setMoveSel({ ...pick, prev: scr });
      setIsDrawing(true);
      return;
    }

    if (tool === "eraser") {
      ctx.save();
      ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = 24 / view.scale;
      ctx.beginPath(); ctx.moveTo(wpt.x, wpt.y);
      ctx.restore();
      setIsDrawing(true);
      return;
    }

    // wall or columns
    ctx.save();
    ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
    ctx.lineWidth   = 2 / view.scale;
    ctx.lineCap     = "round";
    ctx.strokeStyle = tool === "columns" ? "rgba(0,0,0,0.4)" : "black";
    ctx.beginPath(); ctx.moveTo(wpt.x, wpt.y);
    setStrokePts([wpt]);
    setIsDrawing(true);
    ctx.restore();
  };

  const draw = e => {
    if (!isDrawing) return;
    const scr = getScreenXY(e);
    const wpt = screenToWorld(scr);
    const ctx = canvasRef.current.getContext("2d");

    if (tool === "move") {
      const { kind, index, prev } = moveSel;
      const dx = (scr.x - prev.x) / view.scale;
      const dy = (scr.y - prev.y) / view.scale;
      setMoveSel(m => ({ ...m, prev: scr }));
      if (kind === "column") {
        setColumns(cols =>
          cols.map((p,i) => i===index ? { x: p.x+dx, y: p.y+dy } : p)
        );
      } else {
        setWalls(ws =>
          ws.map((stk,i) =>
            i===index
              ? stk.map(pt => ({ x: pt.x+dx, y: pt.y+dy }))
              : stk
          )
        );
      }
      return;
    }

    if (tool === "eraser") {
      ctx.save();
      ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
      ctx.lineTo(wpt.x, wpt.y);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // continue stroke
    setStrokePts(pts => [...pts, wpt]);
    renderCanvas(columns, walls, grid, [...strokePts, wpt], tool === "wall");
  };

  const stopDrawing = e => {
    e.preventDefault();
    if (!isDrawing) return;
    setIsDrawing(false);
    const ctx = ctxRef.current;

    if (tool === "move") {
      setMoveSel(null);
      return;
    }
    if (tool === "eraser") {
      ctx.globalCompositeOperation = "source-over";
      renderCanvas();
      return;
    }
    if (tool === "wall") {
      const simple = simplifyWall(strokePts);      // ▼ NEW
      setWalls(w => [...w, simple]);
    }
    if (tool === "columns") {
      const cx = strokePts.reduce((s,p)=>s+p.x,0) / strokePts.length;
      const cy = strokePts.reduce((s,p)=>s+p.y,0) / strokePts.length;
      setColumns(c => [...c, { x: cx, y: cy }]);
    }
    setStrokePts([]);
  };

  const pickObject = scr => {
    // screen‐space picking
    const colIdx = columns.findIndex(p => {
      const s = worldToScreen(p);
      return (s.x-scr.x)**2 + (s.y-scr.y)**2 < (8)**2;
    });
    if (colIdx !== -1) return { kind: "column", index: colIdx };

    for (let i=0; i<walls.length; i++) {
      const stk = walls[i].map(worldToScreen);
      for (let j=0; j<stk.length-1; j++) {
        const a = stk[j], b = stk[j+1];
        const t = Math.max(0, Math.min(1,
          ((scr.x-a.x)*(b.x-a.x) + (scr.y-a.y)*(b.y-a.y)) /
          ((b.x-a.x)**2 + (b.y-a.y)**2)));
        const px = a.x + t*(b.x-a.x), py = a.y + t*(b.y-a.y);
        if (Math.hypot(scr.x-px, scr.y-py) < 6) {
          return { kind: "wall", index: i };
        }
      }
    }
    return null;
  };

  const clearCanvas = () => {
    recordState();
    setColumns([]);
    setWalls([]);
    setGrid(null);
  };

  // ─── render ────────────────────────────────────────────────────────────────
  const renderCanvas = (
    cols = columns,
    ws  = walls,
    g   = grid,
    tempStroke = null,
    isWall     = false
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // clear + frame
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    drawFrame(ctx, canvas);

    // world → screen
    ctx.save();
    ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);

    // draw grid
    if (g && showGrid) {
      ctx.lineWidth   = 1/view.scale;
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      const reach = (Math.hypot(canvas.width,canvas.height)+20)/view.scale;
      const { xs, ys, dirA, dirB } = g;
      xs.forEach(xc => {
        ctx.beginPath();
        ctx.moveTo(dirA.x*xc - dirB.x*reach, dirA.y*xc - dirB.y*reach);
        ctx.lineTo(dirA.x*xc + dirB.x*reach, dirA.y*xc + dirB.y*reach);
        ctx.stroke();
      });
      ys.forEach(yc => {
        ctx.beginPath();
        ctx.moveTo(dirB.x*yc - dirA.x*reach, dirB.y*yc - dirA.y*reach);
        ctx.lineTo(dirB.x*yc + dirA.x*reach, dirB.y*yc + dirA.y*reach);
        ctx.stroke();
      });
    }

    // draw walls
    ctx.lineWidth   = 2/view.scale;
    ctx.lineCap     = "round";
    ctx.strokeStyle = "black";
    ws.forEach(stk => {
      if (stk.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(stk[0].x, stk[0].y);
      stk.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });

    // draw preview stroke
    if (tempStroke && tempStroke.length > 1) {
      ctx.beginPath();
      ctx.moveTo(tempStroke[0].x, tempStroke[0].y);
      tempStroke.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      if (isWall) {
        ctx.strokeStyle = "black";
      } else {
        ctx.strokeStyle = "rgba(0,0,0,0.4)";
        ctx.lineWidth   = 2/view.scale;
      }
      ctx.stroke();
    }

    // draw columns
    ctx.fillStyle = "black";
    cols.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4/view.scale, 0, Math.PI*2);
      ctx.fill();
    });

    ctx.restore();
  };
  useEffect(() => renderCanvas(), [columns, walls, grid, showGrid, view]);

  // ─── UI ───────────────────────────────────────────────────────────────────
  const Btn = ({ children, active, disabled, onClick }) => (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`p-2 rounded-2xl shadow  
        ${disabled
          ? "bg-gray-100 text-gray-400"
          : active
          ? "bg-blue-600 text-white"
          : "bg-gray-200"}`}
    >
      {children}
    </button>
  );

  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">
        PLANGROUND
      </h1>

      <div
        className="border-4 border-black rounded-lg p-2 shadow-md"
        style={{ touchAction: "none" }}
      >
        <canvas
          ref={canvasRef}
          className="cursor-crosshair rounded-md"
          onWheel={handleWheel}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>

      <div className="mt-4 flex flex-wrap gap-4 justify-center">
        {["wall", "columns", "eraser", "move"].map(id => (
          <Btn
            key={id}
            active={tool === id}
            onClick={() => setTool(id)}
          >
            {id === "move" ? "Move object" : id.charAt(0).toUpperCase() + id.slice(1)}
          </Btn>
        ))}

        <Btn onClick={optimizeDrawing}>Optimize</Btn>
        <Btn onClick={rhythmize} disabled={!hasOptimized}>
          Rhythmizer
        </Btn>
        <Btn disabled={!history.length} onClick={undo}>Undo</Btn>
        <Btn disabled={!future.length} onClick={redo}>Redo</Btn>
        <Btn disabled={!grid} onClick={() => setShowGrid(g => !g)}>
          {showGrid ? "Hide grid" : "Show grid"}
        </Btn>
        <Btn onClick={clearCanvas}>Clear</Btn>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <label className="whitespace-nowrap">Looseness:</label>
        <input
          type="range"
          min="1"
          max="10"
          value={epsFactor}
          onChange={e => setEpsFactor(+e.target.value)}
          className="accent-blue-600 w-48"
        />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
 }