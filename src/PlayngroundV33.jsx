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



  const [showTempGridHighlight, setShowTempGridHighlight] = useState(false);






  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  const moveSelRef = useRef(null);
  const panRef = useRef(null);

  const [wallBreaks, setWallBreaks] = useState({});


  const [recentlyHighlighted, setRecentlyHighlighted] = useState(false);






  const [view, setView]         = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [tool, setTool]         = useState("wall");
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);
  const [mode, setMode] = useState("secondary"); // or "primary"


  const [primaryColumns, setPrimaryColumns] = useState([]);
  const [primaryWalls, setPrimaryWalls] = useState([]);
  const [secondaryColumns, setSecondaryColumns] = useState([]);
  const [secondaryWalls, setSecondaryWalls] = useState([]);

  const [primaryOptimized, setPrimaryOptimized] = useState(false);
  const [secondaryOptimized, setSecondaryOptimized] = useState(false);



  const columns = mode === "primary" ? primaryColumns : secondaryColumns;
  const walls   = mode === "primary" ? primaryWalls   : secondaryWalls;

  const setColumns = mode === "primary" ? setPrimaryColumns : setSecondaryColumns;
  const setWalls   = mode === "primary" ? setPrimaryWalls   : setSecondaryWalls;



  const [primaryGrid, setPrimaryGrid] = useState(null);
  const [secondaryGrid, setSecondaryGrid] = useState(null);
  const [showGrid, setShowGrid] = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);
  

   // ─── wall‑simplification tweakables (from Code F) ───────────────
 const RDP_EPS      = 8;     // px – tolerance for Ramer‑Douglas‑Peucker
 const JUNC_THRESH  = 12;    // px – node snapped only if this close

 const grid = mode === "primary" ? primaryGrid : secondaryGrid;
 const setGrid = mode === "primary" ? setPrimaryGrid : setSecondaryGrid;
 
 const hasOptimized = mode === "primary" ? primaryOptimized : secondaryOptimized;
 const setHasOptimized = mode === "primary" ? setPrimaryOptimized : setSecondaryOptimized;
  





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
  if (!pts || pts.length < 2) return pts;
  return pts;
};


// --- break points where the SECOND derivative of the heading spikes ---
function fragmentPolyline(raw, ACC_THRESHOLD = 0.25) {
  if (!raw || raw.length < 4) return [];

  const sub  = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
  const atan = v   => Math.atan2(v.y, v.x);

  /* 1) headings ---------------------------------------------------- */
  const theta = [];
  for (let i = 1; i < raw.length; i++) theta.push(atan(sub(raw[i], raw[i - 1])));

  /* unwrap so the sequence is continuous -------------------------- */
  for (let i = 1; i < theta.length; i++) {
    while (theta[i] - theta[i - 1] >  Math.PI) theta[i] -= 2 * Math.PI;
    while (theta[i] - theta[i - 1] < -Math.PI) theta[i] += 2 * Math.PI;
  }

  /* 2) first derivative  (Δθ) ------------------------------------- */
  const d1 = [];
  for (let i = 1; i < theta.length; i++) d1.push(theta[i] - theta[i - 1]);

  /* 3) second derivative (Δ²θ)  ----------------------------------- */
  const d2 = [];
  for (let i = 1; i < d1.length; i++) d2.push(d1[i] - d1[i - 1]);

  /* 4) find local peaks of |Δ²θ| above threshold ------------------ */
  const breaks = [];
  for (let i = 1; i < d2.length - 1; i++) {
    const cur  = Math.abs(d2[i]);
    const prev = Math.abs(d2[i - 1]);
    const next = Math.abs(d2[i + 1]);
    if (cur > ACC_THRESHOLD && cur > prev && cur > next) {
      const ptIdx = i + 2;          // Δ²θ[i] maps to raw[i+2]
      breaks.push(raw[ptIdx]);
    }
  }

  return breaks;
}





         

 









  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);

  const [moveSel, setMoveSel] = useState(null); // { kind, index, prev }

  const deepCopy = obj => JSON.parse(JSON.stringify(obj));

  // ─── history ───────────────────────────────────────────────────────────────
  const recordState = () => {
    setHistory(h => [
      ...h,
      deepCopy({
        primaryColumns, primaryWalls,
        secondaryColumns, secondaryWalls,
        primaryGrid, secondaryGrid

      })
    ]);
    setFuture([]);
  };
  
  const undo = () => {
    if (!history.length) return;
    const prev = history.at(-1);
  
    setFuture(f => [
      ...f,
      deepCopy({
        primaryColumns, primaryWalls,
        secondaryColumns, secondaryWalls,
        primaryGrid, secondaryGrid
      })
    ]);
  
    setPrimaryColumns(prev.primaryColumns);
    setPrimaryWalls(prev.primaryWalls);
    setSecondaryColumns(prev.secondaryColumns);
    setSecondaryWalls(prev.secondaryWalls);
    setPrimaryGrid(prev.primaryGrid);
    setSecondaryGrid(prev.secondaryGrid);
    setHistory(h => h.slice(0, -1));
  };
  
  
  const redo = () => {
    if (!future.length) return;
  
    setFuture(f => {
      const nxt = f.at(-1);
  
      setHistory(h => [
        ...h,
        deepCopy({
          primaryColumns, primaryWalls,
          secondaryColumns, secondaryWalls,
          primaryGrid, secondaryGrid
        })
      ]);
  
      setPrimaryColumns(nxt.primaryColumns);
      setPrimaryWalls(nxt.primaryWalls);
      setSecondaryColumns(nxt.secondaryColumns);
      setSecondaryWalls(nxt.secondaryWalls);
      setPrimaryGrid(nxt.primaryGrid);
      setSecondaryGrid(nxt.secondaryGrid);
  
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
    if (mode !== "secondary" || !grid) return;
  
    // Show light grid highlight briefly
    setShowTempGridHighlight(true);
    setRecentlyHighlighted(true);
    
    if (showGrid) {
      setShowTempGridHighlight(true);
      setTimeout(() => setShowTempGridHighlight(false), 300);
    }
    
    
  
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
    if (mode !== "secondary") return;
  
    // Show light grid highlight briefly
    setShowTempGridHighlight(true);
    setTimeout(() => setShowTempGridHighlight(false), 300);
  
    const pts = [...secondaryColumns, ...secondaryWalls.flat()];
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
  
    const newColumns = snapped.slice(0, secondaryColumns.length);
    const flatWalls  = snapped.slice(secondaryColumns.length);
    let k=0;
    const newWalls = secondaryWalls.map(stk => {
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


  /* ─── Idealize (PRIMARY only) ──────────────────────────────── */
  const idealizePrimaryWalls = () => {
    if (mode !== "primary" || !primaryWalls.length) return;

    recordState();                       // for Undo

    // build new walls list
    const newWalls = primaryWalls.flatMap(idealizePolyline);

    setPrimaryWalls(newWalls);
  };

  

  

  // ─── drawing / move / erase ───────────────────────────────────────────────
  const startDrawing = e => {
    if (tool === "hand") {
      const scr = getScreenXY(e);
      panRef.current = {
        startX: scr.x,
        startY: scr.y,
        startView: { ...view }
      };
      setIsDrawing(true);
      return;
    }
    
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
      const wpt = screenToWorld(getScreenXY(e));
      eraseAt(wpt);
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
    if (tool === "hand") {
      const scr = getScreenXY(e);
      const { startX, startY, startView } = panRef.current;
      setView({
        scale: startView.scale,
        offsetX: startView.offsetX + (scr.x - startX),
        offsetY: startView.offsetY + (scr.y - startY)
      });
      return;
    }
  
    if (!isDrawing) return;
  
    const scr = getScreenXY(e);
    const wpt = screenToWorld(scr);
  
    if (tool === "move") {
      const { kind, index, prev } = moveSel;
      const dx = (scr.x - prev.x) / view.scale;
      const dy = (scr.y - prev.y) / view.scale;
      setMoveSel(m => ({ ...m, prev: scr }));
      if (kind === "column") {
        setColumns(cols =>
          cols.map((p, i) => (i === index ? { x: p.x + dx, y: p.y + dy } : p))
        );
      } else {
        setWalls(ws =>
          ws.map((stk, i) =>
            i === index ? stk.map(pt => ({ x: pt.x + dx, y: pt.y + dy })) : stk
          )
        );
      }
      return;
    }
  
    if (tool === "eraser") {
      const wpt = screenToWorld(getScreenXY(e));
      eraseAt(wpt);
      return;
    }
  
    // ✅ Live update stroke and redraw
    setStrokePts(prev => {
      const newPts = [...prev, wpt];
      renderCanvas(newPts, tool === "wall");
      return newPts;
    });
  };



  const resampleLine = (pts, segmentLength = 4) => {
    if (pts.length < 2) return pts;
  
    const resampled = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const a = resampled[resampled.length - 1];
      const b = pts[i];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy);
      const steps = Math.floor(dist / segmentLength);
  
      for (let j = 1; j <= steps; j++) {
        const t = j / steps;
        resampled.push({
          x: a.x + t * dx,
          y: a.y + t * dy,
        });
      }
    }
  
    return resampled;
  };
  
  








  const stopDrawing = e => {
    if (tool === "hand") {
      panRef.current = null;
      return;
    }
    
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
      // Densify the stroke for smoother curves (smaller = more segments)
      const denseStroke = resampleLine(strokePts, 4);
    
      // Optionally simplify the dense stroke (you can skip this if not needed)
      const simplified = simplifyWall(denseStroke);
    
      // Store in the appropriate mode
      if (mode === "primary") {
        setPrimaryWalls(walls => [...walls, simplified]);
      } else {
        setSecondaryWalls(walls => [...walls, simplified]);
      }
    }
    
    if (tool === "columns") {
      const cx = strokePts.reduce((s,p)=>s+p.x,0) / strokePts.length;
      const cy = strokePts.reduce((s,p)=>s+p.y,0) / strokePts.length;
      if (mode === "primary") {
        setPrimaryColumns(c => [...c, { x: cx, y: cy }]);
      } else {
        setSecondaryColumns(c => [...c, { x: cx, y: cy }]);
      }
      
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

  const eraseAt = wpt => {
    const hitTol = 8 / view.scale; // in world coords
  
    // Columns
    const ci = columns.findIndex(p =>
      (p.x - wpt.x) ** 2 + (p.y - wpt.y) ** 2 < hitTol ** 2
    );
    if (ci !== -1) {
      setColumns(cols => cols.filter((_, i) => i !== ci));
      return;
    }
  
    // Walls
    const ptSegDist = (p, a, b) => {
      const t = Math.max(0, Math.min(1,
        ((p.x - a.x)*(b.x - a.x) + (p.y - a.y)*(b.y - a.y)) /
        ((b.x - a.x)**2 + (b.y - a.y)**2)
      ));
      const px = a.x + t * (b.x - a.x);
      const py = a.y + t * (b.y - a.y);
      return Math.hypot(p.x - px, p.y - py);
    };
  
    for (let wi = 0; wi < walls.length; wi++) {
      const stk = walls[wi];
      for (let j = 0; j < stk.length - 1; j++) {
        if (ptSegDist(wpt, stk[j], stk[j + 1]) < hitTol) {
          setWalls(ws => ws.filter((_, ii) => ii !== wi));
          return;
        }
      }
    }
  };
  







  const clearCanvas = () => {
    recordState();
    setColumns([]);
    setWalls([]);
    setGrid(null);
  };

  const renderCanvas = (
    tempStroke = null,
    isWall = false
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
  
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
  
    ctx.save();
    ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);
  
    // Draw grid (one pass only, based on active color)
    const gridColor = showTempGridHighlight
      ? "rgba(0,0,0,0.08)"
      : (showGrid ? "rgba(0,0,0,0.08)" : null);

    if (grid && gridColor) {
      const { xs, ys, dirA, dirB } = grid;
      ctx.lineWidth = 1 / view.scale;
      ctx.strokeStyle = gridColor;

      for (const x of xs) {
        ctx.beginPath();
        ctx.moveTo(dirA.x * x + dirB.x * -10000, dirA.y * x + dirB.y * -10000);
        ctx.lineTo(dirA.x * x + dirB.x * 10000, dirA.y * x + dirB.y * 10000);
        ctx.stroke();
      }

      for (const y of ys) {
        ctx.beginPath();
        ctx.moveTo(dirB.x * y + dirA.x * -10000, dirB.y * y + dirA.y * -10000);
        ctx.lineTo(dirB.x * y + dirA.x * 10000, dirB.y * y + dirA.y * 10000);
        ctx.stroke();
      }
    }




  
    const renderWalls = (walls, color) => {
      ctx.lineWidth = 2 / view.scale;
      ctx.lineCap = "round";
      ctx.strokeStyle = color;
      walls.forEach(stk => {
        if (stk.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(stk[0].x, stk[0].y);
        stk.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
        ctx.stroke();
      });
    };
  
    const renderCols = (cols, color) => {
      ctx.fillStyle = color;
      cols.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4 / view.scale, 0, Math.PI * 2);
        ctx.fill();
      });
    };
  
    if (mode === "primary") {
      renderWalls(secondaryWalls, "rgba(0,0,0,0.2)");
      renderCols(secondaryColumns, "rgba(0,0,0,0.2)");
      renderWalls(primaryWalls, "black");
      renderCols(primaryColumns, "black");
    } else {
      renderWalls(primaryWalls, "rgba(0,0,0,0.2)");
      renderCols(primaryColumns, "rgba(0,0,0,0.2)");
      renderWalls(secondaryWalls, "black");
      renderCols(secondaryColumns, "black");
    }
  
    // ✅ Draw temporary stroke (in-progress wall or column)
    if (tempStroke && tempStroke.length > 1) {
      ctx.beginPath();
      ctx.moveTo(tempStroke[0].x, tempStroke[0].y);
      tempStroke.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.strokeStyle = isWall ? "black" : "rgba(0,0,0,0.4)";
      ctx.stroke();
    }
  
    // ✅ Draw curvature breakpoints
    if (mode === "primary") {
      ctx.fillStyle = "black";
      Object.values(wallBreaks).forEach(group => {
        group.forEach(p => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 6 / view.scale, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    }
    
  
    ctx.restore();
  };
  
  

  useEffect(() => {
    const isWall = tool === "wall";
    renderCanvas(strokePts.length > 1 ? strokePts : null, isWall);
  }, [
    primaryWalls, primaryColumns,
    secondaryWalls, secondaryColumns,
    strokePts, view, mode, grid,
    showGrid, showTempGridHighlight, wallBreaks, tool

  ]);
  
  


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
  
    const handleWheel = e => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const scr = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  
      setView(prev => {
        const newScale = prev.scale * factor;
        const wx = (scr.x - prev.offsetX) / prev.scale;
        const wy = (scr.y - prev.offsetY) / prev.scale;
        return {
          scale: newScale,
          offsetX: scr.x - wx * newScale,
          offsetY: scr.y - wy * newScale
        };
      });
    };
  
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, []); // ✅ Use empty dependency array to attach only once
  
  





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
  
      {/* Mode toggle and indicator */}
      <div className="mb-4 flex items-center gap-4">
        <span className="text-lg font-semibold">
          Active mode:
          <span className={`ml-2 px-2 py-1 rounded-xl text-white ${mode === "primary" ? "bg-blue-600" : "bg-gray-600"}`}>
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </span>
        </span>
  
        <div className="flex gap-2 ml-4">
          <button
            className={`px-4 py-1 rounded-xl shadow border transition
              ${mode === "primary" ? "bg-blue-600 text-white border-blue-800" : "bg-gray-200 text-black border-gray-400"}`}
            onClick={() => setMode("primary")}
          >
            Primary
          </button>
          <button
            className={`px-4 py-1 rounded-xl shadow border transition
              ${mode === "secondary" ? "bg-blue-600 text-white border-blue-800" : "bg-gray-200 text-black border-gray-400"}`}
            onClick={() => setMode("secondary")}
          >
            Secondary
          </button>
        </div>
      </div>
  
      {/* Canvas */}
      <div
        className="border-4 border-black rounded-lg p-2 shadow-md"
        style={{ touchAction: "none", overscrollBehavior: "contain" }}
      >
        <canvas
          ref={canvasRef}
          className="cursor-crosshair rounded-md"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>
  
      {/* Tool Buttons */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center">
        {["wall", "columns", "eraser", "move", "hand"].map(id => (
          <Btn
            key={id}
            active={tool === id}
            onClick={() => setTool(id)}
          >
            {id === "move" ? "Move object" : id.charAt(0).toUpperCase() + id.slice(1)}
          </Btn>
        ))}

        {/* Idealize appears only in Primary and acts immediately */}
        <Btn
          active={false}              // one-click action, not a “mode”
          disabled={mode!=="primary" || !primaryWalls.length}
          onClick={() => {
            const allBreaks = {};
            primaryWalls.forEach((wall, idx) => {
              allBreaks[idx] = fragmentPolyline(wall);
            });
            setWallBreaks(allBreaks);
          }}
          
          
        >
          Idealize
        </Btn>

  
        <Btn onClick={optimizeDrawing} disabled={mode === "primary"}>
          Optimize
        </Btn>
        <Btn onClick={rhythmize} disabled={mode === "primary" || !hasOptimized}>
          Rhythmizer
        </Btn>
        <Btn disabled={!history.length} onClick={undo}>Undo</Btn>
        <Btn disabled={!future.length} onClick={redo}>Redo</Btn>
        <Btn disabled={!grid} onClick={() => setShowGrid(g => !g)}>
          {showGrid ? "Hide grid" : "Show grid"}
        </Btn>
        <Btn onClick={clearCanvas}>Clear</Btn>
      </div>
  
      {/* Looseness Slider */}
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