import React, { useRef, useState, useEffect } from "react";

/**
 * PLANGROUND — wall/column sketch‑optimiser + cursor‑centred wheel‑zoom
 * --------------------------------------------------------
 *  • Wall / Columns(centroid) / Eraser / Move object
 *  • Optimize (restored robust algorithm)
 *  • Mouse‑wheel zoom (view transform only; geometry unchanged)
 *  • Grid toggle, Undo / Redo, Clear, Looseness slider
 */
export default function Planground() {
  /* refs */
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  /* view transform (world↔screen) */
  const [view, setView] = useState({ scale: 1, offsetX: 0, offsetY: 0 });

  /* drawing state */
  const [tool, setTool]           = useState("wall");  // wall | columns | eraser | move
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);        // world coords

  const [columns, setColumns]     = useState([]);        // world [{x,y}]
  const [walls,   setWalls]       = useState([]);        // world [[{x,y}, …], …]

  const [grid,    setGrid]        = useState(null);      // {xs, ys, dirA, dirB}
  const [showGrid, setShowGrid]   = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);

  /* history */
  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);

  /* move‑tool selection */
  const [moveSel, setMoveSel] = useState(null);         // {kind,index,prevMouse}

  const deepCopy = o => JSON.parse(JSON.stringify(o));

  /* coordinate helpers */
  const screenToWorld = ({ x, y }) => ({ x: (x - view.offsetX) / view.scale, y: (y - view.offsetY) / view.scale });
  const worldToScreen = ({ x, y }) => ({ x: x * view.scale + view.offsetX, y: y * view.scale + view.offsetY });
  const getScreenXY = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const r = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  /* history helpers */
  const recordState = () => {
    setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
    setFuture([]);
  };
  const undo = () => {
    if (!history.length) return;
    setHistory(h => {
      const prev = h.at(-1);
      setFuture(f => [...f, deepCopy({ columns, walls, grid })]);
      setColumns(prev.columns); setWalls(prev.walls); setGrid(prev.grid);
      return h.slice(0, -1);
    });
  };
  const redo = () => {
    if (!future.length) return;
    setFuture(f => {
      const nxt = f.at(-1);
      setHistory(h => [...h, deepCopy({ columns, walls, grid })]);
      setColumns(nxt.columns); setWalls(nxt.walls); setGrid(nxt.grid);
      return f.slice(0, -1);
    });
  };

  /* canvas init */
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth * 0.9;
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
    ctx.setTransform(1,0,0,1,0,0);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  /* wheel‑zoom */
  const handleWheel = e => {
    e.preventDefault();
    const scr = getScreenXY(e);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = view.scale * factor;
    const wx = screenToWorld(scr).x;
    const wy = screenToWorld(scr).y;
    setView({
      scale: newScale,
      offsetX: scr.x - wx * newScale,
      offsetY: scr.y - wy * newScale
    });
  };

  /* pointer handlers */
  const startDrawing = e => {
    e.preventDefault();
    const scr = getScreenXY(e);
    const wpt = screenToWorld(scr);
    recordState();
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    if (tool === "move") {
      const pick = pickObject(scr);
      if (!pick) return;
      setMoveSel({ ...pick, prevMouse: scr });
      setIsDrawing(true);
      return;
    }

    if (tool === "eraser") {
      ctx.save();
      ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = 24 / view.scale;
      ctx.beginPath();
      ctx.moveTo(wpt.x, wpt.y);
      ctx.restore();
      setIsDrawing(true);
      return;
    }

    // wall or columns
    setStrokePts([wpt]);
    setIsDrawing(true);
  };

  const draw = e => {
    if (!isDrawing) return;
    const scr = getScreenXY(e);
    const wpt = screenToWorld(scr);

    if (tool === "move") {
      const { kind, index, prevMouse } = moveSel;
      const dx = (scr.x - prevMouse.x) / view.scale;
      const dy = (scr.y - prevMouse.y) / view.scale;
      setMoveSel(sel => ({ ...sel, prevMouse: scr }));
      if (kind === "column") {
        setColumns(cols =>
          cols.map((p,i) => i===index ? { x: p.x+dx, y: p.y+dy } : p)
        );
      } else {
        setWalls(ws =>
          ws.map((stk,i) =>
            i===index ? stk.map(pt => ({ x: pt.x+dx, y: pt.y+dy })) : stk
          )
        );
      }
      return;
    }

    if (tool === "eraser") {
      const ctx = canvasRef.current.getContext("2d");
      ctx.save();
      ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
      ctx.lineTo(wpt.x, wpt.y);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // wall or columns drawing
    setStrokePts(pts => [...pts, wpt]);
    renderCanvas(columns, walls, grid, [...strokePts, wpt], tool === "wall");
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);

    if (tool === "move") {
      setMoveSel(null);
      return;
    }
    if (tool === "eraser") {
      renderCanvas();
      return;
    }

    if (tool === "wall") {
      setWalls(w => [...w, strokePts]);
    }
    if (tool === "columns") {
      const cx = strokePts.reduce((s,p)=>s+p.x,0) / strokePts.length;
      const cy = strokePts.reduce((s,p)=>s+p.y,0) / strokePts.length;
      setColumns(c => [...c, { x: cx, y: cy }]);
    }
    setStrokePts([]);
  };

  /* object pick (screen) */
  const pickObject = scr => {
    const colIdx = columns.findIndex(p => {
      const s = worldToScreen(p);
      return (s.x - scr.x)**2 + (s.y - scr.y)**2 < (8*view.scale)**2;
    });
    if (colIdx !== -1) return { kind: 'column', index: colIdx };

    for (let i=0; i<walls.length; i++) {
      const stkS = walls[i].map(worldToScreen);
      for (let j=0; j<stkS.length-1; j++) {
        const a = stkS[j], b = stkS[j+1];
        const t = Math.max(0, Math.min(1,
          ((scr.x - a.x)*(b.x-a.x) + (scr.y - a.y)*(b.y-a.y)) /
          ((b.x-a.x)**2 + (b.y-a.y)**2)
        ));
        const px = a.x + t*(b.x-a.x), py = a.y + t*(b.y-a.y);
        if (Math.hypot(scr.x-px, scr.y-py) < 6) return { kind: 'wall', index: i };
      }
    }
    return null;
  };

  /* clear */
  const clearCanvas = () => {
    recordState();
    setColumns([]); setWalls([]); setGrid(null);
  };

  /* Optimize (identical to Code‑A) */
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;
    recordState();
    const dot = (a,b) => a.x*b.x + a.y*b.y;
    const norm = v => Math.hypot(v.x, v.y);
    const vec  = (p,q) => ({ x: q.x-p.x, y: q.y-p.y });
    const ang  = v => (Math.atan2(v.y, v.x) + Math.PI*2) % Math.PI;

    const edges = pts.map((_,i) => {
      let best = { d: Infinity, j: -1 };
      for (let j=0;j<pts.length;j++) if(i!==j){
        const d = norm(vec(pts[i], pts[j]));
        if (d < best.d) best = { d, j };
      }
      return { v: vec(pts[i], pts[best.j]) };
    });

    const bins = 180;
    const hist = Array(bins).fill(0);
    edges.forEach(e => hist[Math.floor(ang(e.v)/Math.PI*bins)]++);
    const p1 = hist.indexOf(Math.max(...hist));
    const masked = hist.map((v,i) => {
      const d = Math.min(Math.abs(i-p1), bins-Math.abs(i-p1));
      return d <= 45 ? 0 : v;
    });
    const p2 = masked.indexOf(Math.max(...masked));

    let th1 = (p1+0.5)*Math.PI/bins;
    let th2 = (p2+0.5)*Math.PI/bins;
    const delta = ((th2-th1+Math.PI)%Math.PI) - Math.PI/2;
    th1 += delta/2; th2 -= delta/2;

    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };

    const xProj = pts.map(p=>dot(p, dirA));
    const yProj = pts.map(p=>dot(p, dirB));

    const typicalGap = arr => {
      const uniq = [...new Set(arr)].sort((a,b)=>a-b);
      if (uniq.length < 2) return 1;
      const gaps = uniq.slice(1).map((v,i) => v-uniq[i]).sort((a,b)=>a-b);
      const i0 = Math.floor(gaps.length * 0.05);
      const i1 = Math.floor(gaps.length * 0.95);
      const core = gaps.slice(i0, i1);
      return core[Math.floor(core.length/2)];
    };

    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a,b)=>a-b);
      const out = [];
      let group = [sorted[0]];
      for (let i = 1; i < sorted.length; i++) {
        if (Math.abs(sorted[i] - sorted[i-1]) <= eps) group.push(sorted[i]);
        else { out.push(group); group = [sorted[i]]; }
      }
      out.push(group);
      return out.map(g => g.reduce((s,v)=>s+v,0)/g.length);
    };

    const epsX = Math.max(typicalGap(xProj) * epsFactor, 1);
    const epsY = Math.max(typicalGap(yProj) * epsFactor, 1);
    const xs = cluster1D(xProj, epsX);
    const ys = cluster1D(yProj, epsY);

    const nearest = (arr, v) => arr.reduce((p,c)=>(Math.abs(c-v)<Math.abs(p-v)?c:p));
    const snapped = pts.map((p,i)=>{
      const sx = nearest(xs, xProj[i]);
      const sy = nearest(ys, yProj[i]);
      return { x: dirA.x*sx+dirB.x*sy, y: dirA.y*sx+dirB.y*sy };
    });

    const newColumns = snapped.slice(0, columns.length);
    const flatWalls = snapped.slice(columns.length);
    let k=0;
    const newWalls = walls.map(stk => {
      const seg = flatWalls.slice(k, k+stk.length);
      k += stk.length; return seg;
    });

    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
  };

  /* render */
  const renderCanvas = (
    cols = columns,
    ws = walls,
    g = grid,
    tempStroke = null,
    strokeIsWall = false
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    drawFrame(ctx, canvas);

    ctx.save();
    ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);

    /* grid */
    if (g && showGrid) {
      ctx.lineWidth = 1/view.scale;
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      const reach = (Math.hypot(canvas.width,canvas.height) + 20) / view.scale;
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

    /* walls */
    ctx.lineWidth = 2/view.scale;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ws.forEach(stk => {
      if (stk.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(stk[0].x, stk[0].y);
      stk.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });

    /* preview stroke for tool */
    if (tempStroke) {
      ctx.beginPath();
      ctx.moveTo(tempStroke[0].x, tempStroke[0].y);
      tempStroke.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      if (strokeIsWall) {
        ctx.strokeStyle = "black";
        ctx.stroke();
      } else {
        ctx.strokeStyle = "rgba(0,0,0,0.4)";
        ctx.lineWidth = 2/view.scale;
        ctx.stroke();
      }
    }

    /* columns */
    ctx.fillStyle = "black";
    columns.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4/view.scale, 0, Math.PI*2);
      ctx.fill();
    });

    ctx.restore();
  };

  useEffect(() => { renderCanvas(); }, [columns, walls, grid, showGrid, view]);

  /* UI */
  const Btn = ({ children, active, disabled, onClick }) => (
    <button disabled={disabled} onClick={onClick}
      className={`p-2 rounded-2xl shadow
        ${disabled ? "bg-gray-100 text-gray-400"
          : active ? "bg-blue-600 text-white"
                   : "bg-gray-200"}`}
    >{children}</button>
  );

  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">PLANGROUND</h1>
      <div className="border-4 border-black rounded-lg p-2 shadow-md" style={{ touchAction: "none" }}>
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
          <Btn key={id}
            active={tool===id}
            disabled={false}
            onClick={() => setTool(id)}
          >
            {id==="move"?"Move object": id.charAt(0).toUpperCase()+id.slice(1)}
          </Btn>
        ))}
        <Btn onClick={optimizeDrawing}>Optimize</Btn>
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
