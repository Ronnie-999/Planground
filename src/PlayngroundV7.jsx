import { useRef, useState, useEffect } from "react";

/**
 * Planground — grid‑aware sketch canvas with history
 * --------------------------------------------------
 *  ▸  Tools:  Wall  |  Columns (centroid)  |  Eraser
 *  ▸  Optimize  →  detect square grid, snap & (optionally) show faint grid
 *  ▸  Grid toggle, Undo / Redo, Clear
 *  ▸  “Playground” heading auto‑removed on every load
 */
export default function Planground() {
  /* ───────── refs ───────── */
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  /* ───────── state ───────── */
  const [isDrawing, setIsDrawing]  = useState(false);
  const [tool, setTool]            = useState("wall");         // wall | columns | eraser
  const [strokePts, setStrokePts]  = useState([]);             // points of current drag

  const [columns, setColumns]      = useState([]);             // [{x,y}]
  const [walls,   setWalls]        = useState([]);             // [[{x,y}, …], …]
  const [grid,    setGrid]         = useState(null);           // { xs, ys, dirA, dirB }
  const [showGrid, setShowGrid]    = useState(true);           // toggle visibility

  const [epsFactor, setEpsFactor]  = useState(4);              // 1 … 10

  /* history */
  const [history, setHistory] = useState([]);                  // past states
  const [future,  setFuture]  = useState([]);                  // redo stack

  /* ───────── helpers ───────── */
  const deepCopy = obj => JSON.parse(JSON.stringify(obj));

  const recordState = () => {
    setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
    setFuture([]);                                             // clear redo path
  };

  const undo = () => {
    if (!history.length) return;
    setHistory(h => {
      const prev = h[h.length - 1];
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
      const next = f[f.length - 1];
      setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
      setColumns(next.columns);
      setWalls(next.walls);
      setGrid(next.grid);
      return f.slice(0, -1);
    });
  };

  /* ───────── canvas init ───────── */
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth  * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
    document.title = "Planground";

    // remove any stray “Playground” heading the host page might inject
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
    ctx.lineWidth   = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const getXY = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const rect = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - rect.left, y: t.clientY - rect.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  /* ───────── pointer handlers ───────── */
  const startDrawing = e => {
    e.preventDefault();
    const { x, y } = getXY(e);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

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
        ctx.strokeStyle = "rgba(0,0,0,0.4)";                   // faint scribble
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

    const ctx = ctxRef.current;

    if (tool === "eraser") {
      ctx.globalCompositeOperation = "source-over";
      drawFrame(ctx, canvasRef.current);
      renderCanvas(columns, walls, grid);                      // redraw vector data
      return;
    }

    if (tool === "wall") {
      setWalls(w => [...w, strokePts]);
      setStrokePts([]);
    }

    if (tool === "columns") {
      // centroid of scribble
      const cx = strokePts.reduce((s, p) => s + p.x, 0) / strokePts.length;
      const cy = strokePts.reduce((s, p) => s + p.y, 0) / strokePts.length;
      setColumns(c => [...c, { x: cx, y: cy }]);
      setStrokePts([]);
    }
  };

  /* ───────── clear ───────── */
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

  /* ───────── optimize & grid detection ───────── */
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;
    recordState();

    /* helper functions */
    const dot = (a, b) => a.x * b.x + a.y * b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const vec  = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });
    const ang  = v => {
      const a = Math.atan2(v.y, v.x);
      return a < 0 ? a + Math.PI : a;                          // 0..π
    };

    /* 1) nearest-neighbour edges */
    const edges = [];
    for (let i = 0; i < pts.length; i++) {
      let best = { d: Infinity, j: -1 };
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        const d = norm(vec(pts[i], pts[j]));
        if (d < best.d) best = { d, j };
      }
      edges.push({ i, j: best.j, v: vec(pts[i], pts[best.j]) });
    }

    /* 2) orientation histogram — pick two orthogonal peaks */
    const bins = 180;
    const hist = Array(bins).fill(0);
    edges.forEach(e => hist[Math.floor(ang(e.v) / Math.PI * bins)]++);
    const p1 = hist.indexOf(Math.max(...hist));
    const mask = hist.map((v, i) => {
      const d = Math.min(Math.abs(i - p1), bins - Math.abs(i - p1));
      return d <= 45 ? 0 : v;
    });
    const p2 = mask.indexOf(Math.max(...mask));

    let th1 = (p1 + 0.5) * Math.PI / bins;
    let th2 = (p2 + 0.5) * Math.PI / bins;
    const delta = (((th2 - th1 + Math.PI) % Math.PI) - Math.PI / 2);
    th1 += delta / 2;
    th2 -= delta / 2;

    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };

    /* 3) projections onto custom axes */
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));

    const medianGap = arr => {
      const uniq = [...new Set(arr)].sort((a, b) => a - b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v, i) => v - uniq[i]).sort((a, b) => a - b);
      return gaps[Math.floor(gaps.length / 2)];
    };

    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const groups = [];
      let g = [sorted[0]];
      for (let i = 1; i < sorted.length; i++) {
        if (Math.abs(sorted[i] - sorted[i - 1]) <= eps) g.push(sorted[i]);
        else {
          groups.push(g);
          g = [sorted[i]];
        }
      }
      groups.push(g);
      return groups.map(gr => gr.reduce((s, v) => s + v, 0) / gr.length);
    };

    const epsX = Math.max(medianGap(xProj) * epsFactor * 0.2, 1);
    const epsY = Math.max(medianGap(yProj) * epsFactor * 0.2, 1);
    const xs = cluster1D(xProj, epsX);
    const ys = cluster1D(yProj, epsY);

    /* 4) snap all points */
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

    const newColumns = snapped.slice(0, columns.length);
    const flatWalls = snapped.slice(columns.length);
    const newWalls = [];
    let k = 0;
    walls.forEach(stk => {
      newWalls.push(flatWalls.slice(k, k + stk.length));
      k += stk.length;
    });

    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
  };

  /* ───────── render canvas ───────── */
  const renderCanvas = (cols = columns, ws = walls, g = grid) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);

    /* grid first */
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
    ctx.lineWidth   = 2;
    ctx.lineCap     = "round";
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

  /* re‑render whenever data or grid visibility changes */
  useEffect(() => {
    renderCanvas(columns, walls, grid);
  }, [columns, walls, grid, showGrid]);

  /* ───────── UI helpers ───────── */
  const Button = ({ children, active, disabled, onClick }) => (
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

  /* ───────── JSX ───────── */
  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">
        Planground
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

      {/* primary controls */}
      <div className="mt-4 flex flex-wrap gap-4 justify-center">
        {[
          { id: "wall",    label: "Wall" },
          { id: "columns", label: "Columns" },
          { id: "eraser",  label: "Eraser" }
        ].map(b => (
          <Button
            key={b.id}
            active={tool === b.id}
            onClick={() => setTool(b.id)}
          >
            {b.label}
          </Button>
        ))}

        <Button onClick={optimizeDrawing}>Optimize</Button>
        <Button disabled={!history.length} onClick={undo}>Undo</Button>
        <Button disabled={!future.length}  onClick={redo}>Redo</Button>
        <Button disabled={!grid} onClick={() => setShowGrid(g => !g)}>
          {showGrid ? "Hide grid" : "Show grid"}
        </Button>
        <Button onClick={clearCanvas}>Clear</Button>
      </div>

      {/* looseness slider */}
      <div className="mt-4 flex items-center gap-2">
        <label className="whitespace-nowrap">Looseness:</label>
        <input
          type="range"
          min="1"
          max="10"
          value={epsFactor}
          onChange={e => setEpsFactor(parseInt(e.target.value, 10))}
          className="accent-blue-600 w-48"
        />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
}
