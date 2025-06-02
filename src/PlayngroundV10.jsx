import React, { useRef, useState, useEffect } from "react";

/**
 * PLANGROUND — wall/column sketch-optimiser + cursor-centred wheel-zoom + Rhythmizer
 * --------------------------------------------------------
 *  • Wall / Columns(centroid) / Eraser / Move object
 *  • Optimize (robust algorithm)
 *  • Mouse-wheel zoom (view transform only; geometry unchanged)
 *  • Rhythmizer: snap grid offsets to uniform pitch
 *  • Grid toggle, Undo / Redo, Clear, Looseness slider
 */
export default function Planground() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  const [view, setView] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [tool, setTool] = useState("wall");
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);
  const [columns, setColumns] = useState([]);
  const [walls, setWalls] = useState([]);
  const [grid, setGrid] = useState(null);
  const [showGrid, setShowGrid] = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);
  const [hasOptimized, setHasOptimized] = useState(false);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [moveSel, setMoveSel] = useState(null);
  const deepCopy = o => JSON.parse(JSON.stringify(o));

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

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.8;
    drawFrame(canvas.getContext("2d"), canvas);
    document.title = "PLANGROUND";
    const purge = () => document.querySelectorAll("h1").forEach(el => {
      if (el.id !== "planground-title") el.remove();
    });
    purge();
    const obs = new MutationObserver(purge);
    obs.observe(document.body, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, []);

  const drawFrame = (ctx, canvas) => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const handleWheel = e => {
    e.preventDefault();
    const scr = getScreenXY(e);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = view.scale * factor;
    const { x: wx, y: wy } = screenToWorld(scr);
    setView({
      scale: newScale,
      offsetX: scr.x - wx * newScale,
      offsetY: scr.y - wy * newScale
    });
  };

  const recoverPitchAndOffsets = (offsets, tol = 0.25) => {
    const offs = [...offsets].sort((a, b) => a - b);
    if (!offs.length) return [0, []];
    const gaps = offs.slice(1).map((v, i) => v - offs[i]);
    const init = gaps.length ? gaps.sort((a, b) => a - b)[Math.floor(gaps.length / 2)] : 0;
    const parts = [];
    gaps.forEach(g => {
      const n = init > 0 ? Math.max(1, Math.round(g / init)) : 1;
      if (init > 0 && Math.abs(g / n - init) / init < tol) {
        for (let i = 0; i < n; i++) parts.push(g / n);
      } else parts.push(g);
    });
    const pitch = parts.reduce((s, v) => s + v, 0) / parts.length;
    const centre = offs.reduce((s, v) => s + v, 0) / offs.length;
    const snapped = offs.map(o => Math.round((o - centre) / pitch) * pitch + centre);
    return [pitch, snapped];
  };

  const rhythmize = () => {
    if (!grid) return;
    recordState();
    const { xs, ys, dirA, dirB } = grid;
    const [, snapA] = recoverPitchAndOffsets(xs);
    const [, snapB] = recoverPitchAndOffsets(ys);
    setGrid({ xs: snapA, ys: snapB, dirA, dirB });
  };

  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;
    recordState();
    const dot = (a, b) => a.x * b.x + a.y * b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const vec = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });
    const ang = v => (Math.atan2(v.y, v.x) + Math.PI * 2) % Math.PI;
    const edges = pts.map((_, i) => {
      let best = { d: Infinity, j: -1 };
      for (let j = 0; j < pts.length; j++) if (i !== j) {
        const d = norm(vec(pts[i], pts[j]));
        if (d < best.d) best = { d, j };
      }
      return { v: vec(pts[i], pts[best.j]) };
    });
    const bins = 180;
    const hist = Array(bins).fill(0);
    edges.forEach(e => hist[Math.floor(ang(e.v) / Math.PI * bins)]++);
    const p1 = hist.indexOf(Math.max(...hist));
    const masked = hist.map((v, i) => {
      const d = Math.min(Math.abs(i - p1), bins - Math.abs(i - p1));
      return d <= 45 ? 0 : v;
    });
    const p2 = masked.indexOf(Math.max(...masked));
    let th1 = (p1 + .5) * Math.PI / bins;
    let th2 = (p2 + .5) * Math.PI / bins;
    const delta = ((th2 - th1 + Math.PI) % Math.PI) - Math.PI / 2;
    th1 += delta / 2;
    th2 -= delta / 2;
    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));
    const typicalGap = arr => {
      const uniq = [...new Set(arr)].sort((a, b) => a - b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v, i) => v - uniq[i]).sort((a, b) => a - b);
      const i0 = Math.floor(gaps.length * 0.05);
      const i1 = Math.floor(gaps.length * 0.95);
      const core = gaps.slice(i0, i1);
      return core[Math.floor(core.length / 2)];
    };
    const epsX = Math.max(typicalGap(xProj) * epsFactor, 1);
    const epsY = Math.max(typicalGap(yProj) * epsFactor, 1);
    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const out = [];
      let grp = [sorted[0]];
      for (let i = 1; i < sorted.length; i++) {
        if (Math.abs(sorted[i] - sorted[i - 1]) <= eps) grp.push(sorted[i]);
        else { out.push(grp); grp = [sorted[i]]; }
      }
      out.push(grp);
      return out.map(g => g.reduce((s, v) => s + v, 0) / g.length);
    };
    const xs_ = cluster1D(xProj, epsX);
    const ys_ = cluster1D(yProj, epsY);
    const nearest = (arr, v) => arr.reduce((p, c) => (Math.abs(c - v) < Math.abs(p - v) ? c : p));
    const snapped = pts.map((p, i) => {
      const sx = nearest(xs_, xProj[i]);
      const sy = nearest(ys_, yProj[i]);
      return { x: dirA.x * sx + dirB.x * sy, y: dirA.y * sx + dirB.y * sy };
    });
    const newCols = snapped.slice(0, columns.length);
    const flatW = snapped.slice(columns.length);
    let k = 0;
    const newWalls = walls.map(stk => {
      const seg = flatW.slice(k, k + stk.length);
      k += stk.length;
      return seg;
    });
    setColumns(newCols);
    setWalls(newWalls);
    setGrid({ xs: xs_, ys: ys_, dirA, dirB });
    setShowGrid(true);
    setHasOptimized(true);
  };

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
      ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = 24 / view.scale;
      ctx.beginPath();
      ctx.moveTo(wpt.x, wpt.y);
      ctx.restore();
      setIsDrawing(true);
      return;
    }
    ctx.save();
    ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);
    ctx.lineWidth = tool === "columns" ? 2 / view.scale : 2 / view.scale;
    ctx.strokeStyle = tool === "columns" ? "rgba(0,0,0,0.4)" : "black";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(wpt.x, wpt.y);
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
        setColumns(cols => cols.map((p, i) => i === index ? { x: p.x + dx, y: p.y + dy } : p));
      } else {
        setWalls(ws => ws.map((stk, i) => i === index ? stk.map(pt => ({ x: pt.x + dx, y: pt.y + dy })) : stk));
      }
      return;
    }
    if (tool === "eraser") {
      ctx.save();
      ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);
      ctx.lineTo(wpt.x, wpt.y);
      ctx.stroke();
      ctx.restore();
      return;
    }
    setStrokePts(pts => [...pts, wpt]);
    renderCanvas(columns, walls, grid, [...strokePts, wpt], tool === "wall");
  };

  const stopDrawing = e => {
    e.preventDefault();
    if (!isDrawing) return;
    setIsDrawing(false);
    const ctx = ctxRef.current;
    if (tool === "move") { setMoveSel(null); return; }
    if (tool === "eraser") { ctx.globalCompositeOperation = "source-over"; renderCanvas(); return; }
    if (tool === "wall") { setWalls(w => [...w, strokePts]); }
    if (tool === "columns") {
      const cx = strokePts.reduce((s,p)=>s+p.x,0) / strokePts.length;
      const cy = strokePts.reduce((s,p)=>s+p.y,0) / strokePts.length;
      setColumns(c => [...c, { x: cx, y: cy }]);
    }
    setStrokePts([]);
  };

  const pickObject = p => {
    const col = columns.findIndex(q => (q.x - p.x) ** 2 + (q.y - p.y) ** 2 < (8 / view.scale) ** 2);
    if (col !== -1) return { kind: 'column', index: col, prev: p };
    for (let i = 0; i < walls.length; i++) {
      const stk = walls[i];
      for (let j = 0; j < stk.length - 1; j++) {
        const a = worldToScreen(stk[j]);
        const b = worldToScreen(stk[j+1]);
        const t = Math.max(0, Math.min(1, ((p.x - a.x)*(b.x - a.x) + (p.y - a.y)*(b.y - a.y)) / ((b.x - a.x)**2 + (b.y - a.y)**2)));
        const px = a.x + t*(b.x - a.x);
        const py = a.y + t*(b.y - a.y);
        if (Math.hypot(p.x - px, p.y - py) < 6) return { kind: 'wall', index: i, prev: p };
      }
    }
    return null;
  };

  const clearCanvas = () => { recordState(); setColumns([]); setWalls([]); setGrid(null); };

  const renderCanvas = (cols=columns, ws=walls, g=grid, tempStroke=null, strokeIsWall=false) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height);
    drawFrame(ctx, canvas);
    ctx.save(); ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
    if (g && showGrid) {
      ctx.lineWidth = 1 / view.scale; ctx.strokeStyle = "rgba(0,0,0,0.2)";
      const reach = (Math.hypot(canvas.width,canvas.height) + 20) / view.scale;
      const { xs, ys, dirA, dirB } = g;
      xs.forEach(xc => { ctx.beginPath(); ctx.moveTo(dirA.x*xc - dirB.x*reach, dirA.y*xc - dirB.y*reach); ctx.lineTo(dirA.x*xc + dirB.x*reach, dirA.y*xc + dirB.y*reach); ctx.stroke(); });
      ys.forEach(yc => { ctx.beginPath(); ctx.moveTo(dirB.x*yc - dirA.x*reach, dirB.y*yc - dirA.y*reach); ctx.lineTo(dirB.x*yc + dirA.x*reach, dirB.y*yc + dirA.y*reach); ctx.stroke(); });
    }
    ctx.lineWidth = 2 / view.scale; ctx.lineCap = "round"; ctx.strokeStyle = "black";
    ws.forEach(stk => { if (stk.length < 2) return; ctx.beginPath(); ctx.moveTo(stk[0].x, stk[0].y); stk.slice(1).forEach(p => ctx.lineTo(p.x, p.y)); ctx.stroke(); });
    if (tempStroke && tempStroke.length > 1) { ctx.beginPath(); ctx.moveTo(tempStroke[0].x, tempStroke[0].y); tempStroke.slice(1).forEach(p => ctx.lineTo(p.x, p.y)); if (strokeIsWall) ctx.strokeStyle = "black"; else { ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 2 / view.scale; } ctx.stroke(); }
    ctx.fillStyle = "black"; cols.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 4 / view.scale, 0, 2*Math.PI); ctx.fill(); });
    ctx.restore();
  };

  useEffect(() => { renderCanvas(); }, [columns, walls, grid, showGrid, view]);

  const Btn = ({ children, active, disabled, onClick }) => (
    <button disabled={disabled} onClick={onClick} className={`p-2 rounded-2xl shadow ${disabled?"bg-gray-100 text-gray-400": active?"bg-blue-600 text-white":"bg-gray-200"}`}>{children}</button>
  );

  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">PLANGROUND</h1>
      <div className="border-4 border-black rounded-lg p-2 shadow-md" style={{ touchAction:"none" }}>
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
        {["wall","columns","eraser","move"].map(id => (
          <Btn key={id} active={tool===id} disabled={false} onClick={() => setTool(id)}>
            {id==="move"?"Move object": id.charAt(0).toUpperCase()+id.slice(1)}
          </Btn>
        ))}
        <Btn onClick={optimizeDrawing}>Optimize</Btn>
        <Btn onClick={rhythmize} disabled={!hasOptimized}>Rhythmizer</Btn>
        <Btn disabled={!history.length} onClick={undo}>Undo</Btn>
        <Btn disabled={!future.length} onClick={redo}>Redo</Btn>
        <Btn disabled={!grid} onClick={() => setShowGrid(g => !g)}>{showGrid?"Hide grid":"Show grid"}</Btn>
        <Btn onClick={clearCanvas}>Clear</Btn>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <label className="whitespace-nowrap">Looseness:</label>
        <input type="range" min="1" max="10" value={epsFactor} onChange={e => setEpsFactor(+e.target.value)} className="accent-blue-600 w-48" />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
}
