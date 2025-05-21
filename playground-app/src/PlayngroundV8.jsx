import { useRef, useState, useEffect } from "react";

/**
 * PLANGROUND — wall/column sketch‑optimiser with MOVE tool
 * --------------------------------------------------------
 *  • Wall / Columns(centroid) / Eraser / Move object
 *  • Optimize (restored robust algorithm), Grid toggle
 *  • Undo / Redo, Clear, Looseness slider
 */
export default function Planground() {
  /* ─── refs ─── */
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  /* ─── state ─── */
  const [isDrawing, setIsDrawing]  = useState(false);
  const [tool, setTool]            = useState("wall");     // wall | columns | eraser | move
  const [strokePts, setStrokePts]  = useState([]);

  const [columns, setColumns]      = useState([]);         // [{x,y}]
  const [walls,   setWalls]        = useState([]);         // [[{x,y}, …], …]

  const [grid,    setGrid]         = useState(null);       // {xs, ys, dirA, dirB}
  const [showGrid, setShowGrid]    = useState(true);
  const [epsFactor, setEpsFactor]  = useState(4);

  /* history */
  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);

  /* move‑tool selection */
  const [moveSel, setMoveSel] = useState(null);            // {kind,index,prev}

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

  /* ─── canvas init ─── */
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.8;
    drawFrame(canvas.getContext("2d"), canvas);

    document.title = "PLANGROUND";
    const purge = () => document.querySelectorAll("h1")
      .forEach(el => { if (el.id !== "planground-title") el.remove(); });
    purge();
    const obs = new MutationObserver(purge);
    obs.observe(document.body, { childList: true, subtree: true });
    return () => obs.disconnect();
  }, []);

  const drawFrame = (ctx, canvas) => {
    ctx.lineWidth   = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const getXY = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const r = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  /* ─── pointer handlers ─── */
  const startDrawing = e => {
    e.preventDefault();
    const { x, y } = getXY(e);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    if (tool === "move") {
      const pick = pickObject({ x, y });
      if (!pick) return;
      recordState();
      setMoveSel({ ...pick, prev: { x, y } });
      setIsDrawing(true);
      return;
    }

    switch (tool) {
      case "eraser":
        recordState();
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = 24;
        ctx.beginPath();
        ctx.moveTo(x, y);
        setIsDrawing(true);
        break;

      case "columns":
        recordState();
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = 2;
        ctx.lineCap   = "round";
        ctx.strokeStyle = "rgba(0,0,0,0.4)";
        ctx.beginPath();
        ctx.moveTo(x, y);
        setStrokePts([{ x, y }]);
        setIsDrawing(true);
        break;

      default: // wall
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = 2;
        ctx.lineCap   = "round";
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.moveTo(x, y);
        setStrokePts([{ x, y }]);
        setIsDrawing(true);
        break;
    }
  };

  const draw = e => {
    if (!isDrawing) return;
    const { x, y } = getXY(e);
    const ctx = ctxRef.current;

    if (tool === "move") {
      const { kind, index, prev } = moveSel;
      const dx = x - prev.x, dy = y - prev.y;
      setMoveSel(sel => ({ ...sel, prev: { x, y } }));
      if (kind === "column") {
        setColumns(cols =>
          cols.map((p, i) =>
            i === index ? { x: p.x + dx, y: p.y + dy } : p
          ));
      } else {
        setWalls(ws =>
          ws.map((stk, i) =>
            i === index ? stk.map(pt => ({ x: pt.x + dx, y: pt.y + dy })) : stk
          ));
      }
      return;
    }

    if (tool === "eraser") {
      ctx.lineTo(x, y);
      ctx.stroke();
      return;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    setStrokePts(pts => [...pts, { x, y }]);
  };

  const stopDrawing = e => {
    e.preventDefault();
    if (!isDrawing) return;
    setIsDrawing(false);

    if (tool === "move") {
      setMoveSel(null);
      return;
    }

    const ctx = ctxRef.current;
    if (tool === "eraser") {
      ctx.globalCompositeOperation = "source-over";
      drawFrame(ctx, canvasRef.current);
      renderCanvas(columns, walls, grid);
      return;
    }

    if (tool === "wall") {
      setWalls(w => [...w, strokePts]);
      setStrokePts([]);
    }

    if (tool === "columns") {
      const cx = strokePts.reduce((s, p) => s + p.x, 0) / strokePts.length;
      const cy = strokePts.reduce((s, p) => s + p.y, 0) / strokePts.length;
      setColumns(c => [...c, { x: cx, y: cy }]);
      setStrokePts([]);
    }
  };

  /* ─── object picking ─── */
  const pickObject = p => {
    const colIdx = columns.findIndex(
      q => (q.x - p.x) ** 2 + (q.y - p.y) ** 2 < 8 ** 2
    );
    if (colIdx !== -1) return { kind: "column", index: colIdx };

    const distSeg = (pt, a, b) => {
      const t = Math.max(
        0,
        Math.min(
          1,
          ((pt.x - a.x) * (b.x - a.x) + (pt.y - a.y) * (b.y - a.y)) /
            ((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        )
      );
      const px = a.x + t * (b.x - a.x);
      const py = a.y + t * (b.y - a.y);
      return Math.hypot(pt.x - px, pt.y - py);
    };
    for (let i = 0; i < walls.length; i++) {
      const stk = walls[i];
      for (let j = 0; j < stk.length - 1; j++) {
        if (distSeg(p, stk[j], stk[j + 1]) < 6) return { kind: "wall", index: i };
      }
    }
    return null;
  };

  /* ─── clear ─── */
  const clearCanvas = () => {
    recordState();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    setColumns([]);
    setWalls([]);
    setGrid(null);
  };

  /* ─── restore: robust OPTIMISE (same as earlier “good” version) ─── */
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;
    recordState();

    /* helpers */
    const dot = (a, b) => a.x * b.x + a.y * b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const vec  = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });
    const ang  = v => (Math.atan2(v.y, v.x) + Math.PI * 2) % Math.PI;

    /* 1) nearest‑neighbour edges (exactly one per point) */
    const edges = [];
    for (let i = 0; i < pts.length; i++) {
      let best = { d: Infinity, j: -1 };
      for (let j = 0; j < pts.length; j++) if (i !== j) {
        const d = norm(vec(pts[i], pts[j]));
        if (d < best.d) best = { d, j };
      }
      edges.push({ v: vec(pts[i], pts[best.j]) });
    }

    /* 2) orientation histogram and two‑peak extraction */
    const bins = 180;
    const hist = Array(bins).fill(0);
    edges.forEach(e => hist[Math.floor(ang(e.v) / Math.PI * bins)]++);
    const p1 = hist.indexOf(Math.max(...hist));
    const masked = hist.map((v, i) => {
      const d = Math.min(Math.abs(i - p1), bins - Math.abs(i - p1));
      return d <= 45 ? 0 : v;
    });
    const p2 = masked.indexOf(Math.max(...masked));

    let th1 = (p1 + 0.5) * Math.PI / bins;
    let th2 = (p2 + 0.5) * Math.PI / bins;
    const delta = (((th2 - th1 + Math.PI) % Math.PI) - Math.PI / 2);
    th1 += delta / 2;
    th2 -= delta / 2;

    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };

    /* 3) project coordinates */
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));

    /* typical gap (median of middle 90 % gaps) */
    const typicalGap = arr => {
      const uniq = [...new Set(arr)].sort((a, b) => a - b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v, i) => v - uniq[i]);
      gaps.sort((a, b) => a - b);
      const i0 = Math.floor(gaps.length * 0.05);
      const i1 = Math.floor(gaps.length * 0.95);
      const core = gaps.slice(i0, i1);
      return core[Math.floor(core.length / 2)];
    };

    /* cluster by simple 1‑D DBSCAN‑like merge */
    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const out = [];
      let group = [sorted[0]];
      for (let i = 1; i < sorted.length; i++) {
        if (Math.abs(sorted[i] - sorted[i - 1]) <= eps) group.push(sorted[i]);
        else { out.push(group); group = [sorted[i]]; }
      }
      out.push(group);
      return out.map(g => g.reduce((s, v) => s + v, 0) / g.length);
    };

    const epsX = Math.max(typicalGap(xProj) * epsFactor, 1);
    const epsY = Math.max(typicalGap(yProj) * epsFactor, 1);
    const xs = cluster1D(xProj, epsX);
    const ys = cluster1D(yProj, epsY);

    /* 4) snap each point */
    const nearest = (arr, v) =>
      arr.reduce((p, c) => (Math.abs(c - v) < Math.abs(p - v) ? c : p));
    const snapped = pts.map((p, i) => {
      const sx = nearest(xs, xProj[i]);
      const sy = nearest(ys, yProj[i]);
      return {
        x: dirA.x * sx + dirB.x * sy,
        y: dirA.y * sx + dirB.y * sy
      };
    });

    /* split back into columns & walls */
    const newColumns = snapped.slice(0, columns.length);
    const flatWalls = snapped.slice(columns.length);
    let k = 0;
    const newWalls = walls.map(stk => {
      const seg = flatWalls.slice(k, k + stk.length);
      k += stk.length;
      return seg;
    });

    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
  };

  /* ─── render ─── */
  const renderCanvas = (cols = columns, ws = walls, g = grid) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);

    /* grid */
    if (g && showGrid) {
      ctx.save();
      ctx.lineWidth   = 1;
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      const reach = Math.hypot(canvas.width, canvas.height) + 20;
      const { xs, ys, dirA, dirB } = g;
      xs.forEach(xc => {
        ctx.beginPath();
        ctx.moveTo(dirA.x * xc - dirB.x * reach, dirA.y * xc - dirB.y * reach);
        ctx.lineTo(dirA.x * xc + dirB.x * reach, dirA.y * xc + dirB.y * reach);
        ctx.stroke();
      });
      ys.forEach(yc => {
        ctx.beginPath();
        ctx.moveTo(dirB.x * yc - dirA.x * reach, dirB.y * yc - dirA.y * reach);
        ctx.lineTo(dirB.x * yc + dirA.x * reach, dirB.y * yc + dirA.y * reach);
        ctx.stroke();
      });
      ctx.restore();
    }

    /* walls */
    ctx.lineWidth = 2;
    ctx.lineCap   = "round";
    ctx.strokeStyle = "black";
    ws.forEach(stk => {
      if (stk.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(stk[0].x, stk[0].y);
      stk.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });

    /* columns */
    ctx.fillStyle = "black";
    cols.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  useEffect(() => {
    renderCanvas(columns, walls, grid);
  }, [columns, walls, grid, showGrid]);

  /* ─── UI helper ─── */
  const Btn = ({ children, active, disabled, onClick }) => (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`p-2 rounded 2xl:rounded-2xl shadow
        ${disabled ? "bg-gray-100 text-gray-400"
                   : active   ? "bg-blue-600 text-white"
                              : "bg-gray-200"}`}
    >
      {children}
    </button>
  );

  /* ─── JSX ─── */
  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">
        PLANGROUND
      </h1>

      <div className="border-4 border-black rounded-lg p-2 shadow-md">
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

      {/* controls */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center">
        {[
          { id: "wall",    label: "Wall" },
          { id: "columns", label: "Columns" },
          { id: "eraser",  label: "Eraser" },
          { id: "move",    label: "Move object" }
        ].map(b => (
          <Btn
            key={b.id}
            active={tool === b.id}
            onClick={() => setTool(b.id)}
          >
            {b.label}
          </Btn>
        ))}

        <Btn onClick={optimizeDrawing}>Optimize</Btn>
        <Btn disabled={!history.length} onClick={undo}>Undo</Btn>
        <Btn disabled={!future.length}  onClick={redo}>Redo</Btn>
        <Btn disabled={!grid} onClick={() => setShowGrid(g => !g)}>
          {showGrid ? "Hide grid" : "Show grid"}
        </Btn>
        <Btn onClick={clearCanvas}>Clear</Btn>
      </div>

      {/* looseness */}
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
