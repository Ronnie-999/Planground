import { useRef, useState, useEffect } from "react";

/**
 * Planground – free‑hand sketch canvas ➜ grid optimiser
 * Tools:
 *  • Wall     – continuous lines
 *  • Columns  – discrete points
 *  • Eraser   – remove parts
 * Extra:
 *  • Optimize – detect square grid & snap everything to it
 *  • epsFactor slider – loosen/tighten clustering when building the grid
 */
export default function Planground() {
  const canvasRef   = useRef(null);
  const ctxRef      = useRef(null);

  // drawing‑state
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool]           = useState("wall"); // wall | columns | eraser
  const [currentStroke, setCurrentStroke] = useState([]);

  // data‑models (all in canvas space, y ↓)
  const [columns, setColumns]     = useState([]);       // [{x,y}]
  const [walls, setWalls]         = useState([]);       // [[{x,y}, …], …]

  // optimiser settings
  const [epsFactor, setEpsFactor] = useState(4);        // 1…10 ish

  // ───────────────── canvas init ─────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth  * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
    document.title = "Planground";
    // remove stray <h1>Playground
    document.querySelectorAll("h1").forEach(el => {
      if (el.textContent.trim() === "Playground") el.remove();
    });
  }, []);

  const drawFrame = (ctx, canvas) => {
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const getCoordinates = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const rect = canvasRef.current.getBoundingClientRect();
      const touch = e.nativeEvent.touches[0];
      return { offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top };
    }
    return { offsetX: e.nativeEvent.offsetX, offsetY: e.nativeEvent.offsetY };
  };

  // ───────────────── drawing handlers ─────────────────
  const startDrawing = e => {
    e.preventDefault();
    const { offsetX, offsetY } = getCoordinates(e);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    switch (tool) {
      case "eraser":
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = 24;
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
        setIsDrawing(true);
        break;

      case "columns":
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "black";
        ctx.beginPath();
        ctx.arc(offsetX, offsetY, 4, 0, Math.PI * 2);
        ctx.fill();
        setColumns(cols => [...cols, { x: offsetX, y: offsetY }]);
        setIsDrawing(false);
        break;

      default: // wall
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth  = 2;
        ctx.lineCap    = "round";
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
        setCurrentStroke([{ x: offsetX, y: offsetY }]);
        setIsDrawing(true);
        break;
    }
  };

  const draw = e => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);
    const ctx = ctxRef.current;

    if (tool === "eraser") {
      ctx.lineTo(offsetX, offsetY);
      ctx.stroke();
      return;
    }
    if (tool === "wall") {
      ctx.lineTo(offsetX, offsetY);
      ctx.stroke();
      setCurrentStroke(stk => [...stk, { x: offsetX, y: offsetY }]);
    }
  };

  const stopDrawing = e => {
    e.preventDefault();
    if (!isDrawing) return;
    setIsDrawing(false);

    if (tool === "eraser") {
      ctxRef.current.globalCompositeOperation = "source-over";
      drawFrame(ctxRef.current, canvasRef.current);
    } else if (tool === "wall") {
      setWalls(all => [...all, currentStroke]);
      setCurrentStroke([]);
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    setColumns([]);
    setWalls([]);
  };

  // ───────────────── OPTIMISER CORE ─────────────────
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return; // nothing to do

    // helper maths
    const dot = (a, b) => a.x * b.x + a.y * b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const toVec = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });

    // 1) local edge set (nearest neighbour for each pt)
    const edges = [];
    for (let i = 0; i < pts.length; i++) {
      let minD = Infinity, minJ = -1;
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        const d = norm(toVec(pts[i], pts[j]));
        if (d < minD) { minD = d; minJ = j; }
      }
      if (minJ >= 0) edges.push({ a: i, b: minJ, vec: toVec(pts[i], pts[minJ]) });
    }

    // 2) orientation histogram (0..π)
    const histBins = 180;
    const hist = new Array(histBins).fill(0);
    const toAngle = v => {
      const ang = Math.atan2(v.y, v.x);
      return (ang < 0 ? ang + Math.PI : ang);
    };
    edges.forEach(e => {
      const ang = toAngle(e.vec);
      const bi  = Math.floor(ang / Math.PI * histBins);
      hist[bi] += 1;
    });

    const peak1 = hist.indexOf(Math.max(...hist));
    // zero out ±45° around peak1 and find second peak
    const maskHist = hist.map((v, i) => {
      const diff = Math.abs(i - peak1);
      const wrap = Math.min(diff, histBins - diff);
      return wrap <= 45 ? 0 : v;
    });
    const peak2 = maskHist.indexOf(Math.max(...maskHist));

    const ang1 = (peak1 + 0.5) * Math.PI / histBins;
    const ang2 = (peak2 + 0.5) * Math.PI / histBins;

    // enforce perpendicularity
    const delta = (((ang2 - ang1 + Math.PI) % Math.PI) - Math.PI/2);
    const thetaA = (ang1 + delta / 2);
    const thetaB = (ang2 - delta / 2);
    const dirA = { x: Math.cos(thetaA), y: Math.sin(thetaA) };
    const dirB = { x: Math.cos(thetaB), y: Math.sin(thetaB) };

    // 3) project points onto custom axes
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));

    // helper – group 1D coords into clusters based on eps
    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const groups = [];
      let group = [sorted[0]];
      for (let i = 1; i < sorted.length; i++) {
        if (Math.abs(sorted[i] - sorted[i-1]) <= eps) {
          group.push(sorted[i]);
        } else {
          groups.push(group);
          group = [sorted[i]];
        }
      }
      groups.push(group);
      return groups.map(g => g.reduce((s,v)=>s+v,0)/g.length);
    };

    // typical spacing (median gap)
    const typicalGap = arr => {
      const uniq = [...new Set(arr)].sort((a,b)=>a-b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v,i)=>v-uniq[i]);
      gaps.sort((a,b)=>a-b);
      return gaps[Math.floor(gaps.length/2)];
    };

    const spacingX = typicalGap(xProj);
    const spacingY = typicalGap(yProj);
    const epsX = Math.max(spacingX * epsFactor * 0.2, 1); // scaled
    const epsY = Math.max(spacingY * epsFactor * 0.2, 1);

    const axisXs = cluster1D(xProj, epsX);
    const axisYs = cluster1D(yProj, epsY);

    // snap every point to nearest axis intersection
    const snappedPts = pts.map((p, idx) => {
      const sx = axisXs.reduce((prev, curr) => Math.abs(curr - xProj[idx]) < Math.abs(prev - xProj[idx]) ? curr : prev);
      const sy = axisYs.reduce((prev, curr) => Math.abs(curr - yProj[idx]) < Math.abs(prev - yProj[idx]) ? curr : prev);
      return {
        x: dirA.x * sx + dirB.x * sy,
        y: dirA.y * sx + dirB.y * sy
      };
    });

    // 4) update columns & walls data
    const newColumns = snappedPts.slice(0, columns.length);
    const wallFlat   = snappedPts.slice(columns.length);
    const newWalls = [];
    let idx = 0;
    for (const stroke of walls) {
      newWalls.push(wallFlat.slice(idx, idx + stroke.length));
      idx += stroke.length;
    }
    setColumns(newColumns);
    setWalls(newWalls);

    // 5) redraw everything
    renderCanvas(newColumns, newWalls);
  };

  const renderCanvas = (cols = columns, ws = walls) => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);

    // draw walls
    ctx.lineWidth = 2;
    ctx.strokeStyle = "black";
    ctx.lineCap = "round";
    ws.forEach(stroke => {
      if (stroke.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      stroke.slice(1).forEach(pt => ctx.lineTo(pt.x, pt.y));
      ctx.stroke();
    });

    // draw columns
    ctx.fillStyle = "black";
    cols.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  // keep canvas up‑to‑date when walls/columns arrays change (after snapping etc.)
  useEffect(() => { renderCanvas(); }, [columns, walls]);

  // ───────────────── JSX UI ─────────────────
  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 className="text-3xl font-bold mb-4">Planground</h1>

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

      {/* Tools */}
      <div className="mt-4 flex gap-4 flex-wrap justify-center">
        {[
          { id: "wall",    label: "Wall" },
          { id: "columns", label: "Columns" },
          { id: "eraser",  label: "Eraser" },
        ].map(btn => (
          <button
            key={btn.id}
            onClick={() => setTool(btn.id)}
            className={`p-2 rounded shadow 2xl:rounded-2xl ${
              tool === btn.id ? "bg-blue-600 text-white" : "bg-gray-200"
            }`}
          >
            {btn.label}
          </button>
        ))}
        <button
          onClick={optimizeDrawing}
          className="p-2 bg-green-600 text-white rounded shadow 2xl:rounded-2xl"
        >
          Optimize
        </button>
        <button
          onClick={clearCanvas}
          className="p-2 bg-red-500 text-white rounded shadow 2xl:rounded-2xl"
        >
          Clear
        </button>
      </div>

      {/* epsFactor slider */}
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
