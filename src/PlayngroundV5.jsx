import { useRef, useState, useEffect } from "react";

/**
 * Planground – grid‑aware optimiser
 * ‑ Wall / Columns tools + Eraser
 * ‑ Optimize button → detects square grid, snaps, *and* draws faint grid across canvas
 * ‑ Looseness slider (epsFactor)
 */
export default function Planground() {
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  /* drawing state */
  const [isDrawing, setIsDrawing]   = useState(false);
  const [tool, setTool]             = useState("wall");          // wall | columns | eraser
  const [currentStroke, setStroke]  = useState([]);
  const [columns, setColumns]       = useState([]);               // [{x,y}]
  const [walls,   setWalls]         = useState([]);               // [[{x,y},…],…]

  /* optimiser */
  const [epsFactor, setEpsFactor]   = useState(4);
  const [grid, setGrid]             = useState(null);             // {xs, ys, dirA, dirB}

  // ─── canvas init ───
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth  * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
    document.title = "Planground";
    document.querySelectorAll("h1").forEach(el => {
      if (el.textContent.trim() === "Playground") el.remove();
    });
  }, []);

  const drawFrame = (ctx, canvas) => {
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const getCoords = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const rect = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - rect.left, y: t.clientY - rect.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  // ─── pointer handlers ───
  const startDrawing = e => {
    e.preventDefault();
    const { x, y } = getCoords(e);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    switch (tool) {
      case "eraser":
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = 24;
        ctx.beginPath();
        ctx.moveTo(x, y);
        setIsDrawing(true);
        break;
      case "columns":
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "black";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        setColumns(c => [...c, { x, y }]);
        setIsDrawing(false);
        break;
      default: // wall
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = 2;
        ctx.lineCap  = "round";
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.moveTo(x, y);
        setStroke([{ x, y }]);
        setIsDrawing(true);
        break;
    }
  };

  const draw = e => {
    if (!isDrawing) return;
    const { x, y } = getCoords(e);
    const ctx = ctxRef.current;

    if (tool === "eraser") {
      ctx.lineTo(x, y);
      ctx.stroke();
      return;
    }
    if (tool === "wall") {
      ctx.lineTo(x, y);
      ctx.stroke();
      setStroke(s => [...s, { x, y }]);
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
      setWalls(w => [...w, currentStroke]);
      setStroke([]);
    }
  };

  /* clear */
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    setColumns([]);
    setWalls([]);
    setGrid(null);
  };

  // ─── OPTIMISE & GRID DETECTION ───
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;

    const dot   = (a, b) => a.x * b.x + a.y * b.y;
    const norm  = v => Math.hypot(v.x, v.y);
    const vec   = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });
    const angle = v => { const a = Math.atan2(v.y, v.x); return a < 0 ? a + Math.PI : a; };

    /* 1) nearest‑neighbour edges */
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

    /* 2) orientation histogram to find two orthogonal axes */
    const bins = 180, hist = Array(bins).fill(0);
    edges.forEach(e => { hist[Math.floor(angle(e.v)/Math.PI*bins)]++; });
    const idx1 = hist.indexOf(Math.max(...hist));
    const hist2 = hist.map((v,i)=>{
      const diff = Math.min(Math.abs(i-idx1), bins - Math.abs(i-idx1));
      return diff <= 45 ? 0 : v;
    });
    const idx2 = hist2.indexOf(Math.max(...hist2));
    let th1 = (idx1+0.5)*Math.PI/bins, th2 = (idx2+0.5)*Math.PI/bins;
    const delta = (((th2-th1+Math.PI)%Math.PI)-Math.PI/2);
    th1 += delta/2; th2 -= delta/2;
    const dirA = { x: Math.cos(th1), y: Math.sin(th1) };
    const dirB = { x: Math.cos(th2), y: Math.sin(th2) };

    /* 3) projections */
    const xProj = pts.map(p => dot(p, dirA));
    const yProj = pts.map(p => dot(p, dirB));

    const medianGap = arr => {
      const uniq = [...new Set(arr)].sort((a,b)=>a-b);
      if (uniq.length<2) return 1;
      const gaps = uniq.slice(1).map((v,i)=>v-uniq[i]);
      gaps.sort((a,b)=>a-b);
      return gaps[Math.floor(gaps.length/2)];
    };

    const cluster1D = (arr, eps) => {
      const sorted = [...arr].sort((a,b)=>a-b);
      const out = [];
      let group=[sorted[0]];
      for(let i=1;i<sorted.length;i++){
        if(Math.abs(sorted[i]-sorted[i-1])<=eps){group.push(sorted[i]);}
        else {out.push(group);group=[sorted[i]];}
      }
      out.push(group);
      return out.map(g=>g.reduce((s,v)=>s+v,0)/g.length);
    };

    const epsX = Math.max(medianGap(xProj)*epsFactor*0.2,1);
    const epsY = Math.max(medianGap(yProj)*epsFactor*0.2,1);
    const xs = cluster1D(xProj, epsX);
    const ys = cluster1D(yProj, epsY);

    /* 4) snap points */
    const snapped = pts.map((p, i)=>{
      const nearest = (arr, val)=>arr.reduce((pr,cu)=>Math.abs(cu-val)<Math.abs(pr-val)?cu:pr);
      const sx=nearest(xs,xProj[i]), sy=nearest(ys,yProj[i]);
      return { x: dirA.x*sx + dirB.x*sy, y: dirA.y*sx + dirB.y*sy };
    });

    const newColumns = snapped.slice(0, columns.length);
    const flatWalls  = snapped.slice(columns.length);
    const newWalls   = [];
    let k=0; walls.forEach(stk=>{newWalls.push(flatWalls.slice(k,k+stk.length));k+=stk.length;});

    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
  };

  // ─── rendering helper ───
  const renderCanvas = (cols=columns, ws=walls, g=grid) => {
    const canvas = canvasRef.current, ctx = canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    drawFrame(ctx, canvas);

    /* draw grid (if any) */
    if (g) {
      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      const reach = Math.hypot(canvas.width, canvas.height) + 20;
      const { xs, ys, dirA, dirB } = g;
      xs.forEach(xc=>{
        ctx.beginPath();
        ctx.moveTo(dirA.x*xc - dirB.x*reach, dirA.y*xc - dirB.y*reach);
        ctx.lineTo(dirA.x*xc + dirB.x*reach, dirA.y*xc + dirB.y*reach);
        ctx.stroke();
      });
      ys.forEach(yc=>{
        ctx.beginPath();
        ctx.moveTo(dirB.x*yc - dirA.x*reach, dirB.y*yc - dirA.y*reach);
        ctx.lineTo(dirB.x*yc + dirA.x*reach, dirB.y*yc + dirA.y*reach);
        ctx.stroke();
      });
      ctx.restore();
    }

    /* walls */
    ctx.lineWidth=2; ctx.strokeStyle="black"; ctx.lineCap="round";
    ws.forEach(stk=>{if(stk.length<2)return; ctx.beginPath(); ctx.moveTo(stk[0].x,stk[0].y); stk.slice(1).forEach(p=>ctx.lineTo(p.x,p.y)); ctx.stroke();});

    /* columns */
    ctx.fillStyle="black";
    cols.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fill();});
  };

  // re‑render on state change
  useEffect(()=>{renderCanvas(columns,walls,grid);},[columns,walls,grid]);

  // ─── UI ───
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

      {/* controls */}
      <div className="mt-4 flex gap-4 flex-wrap justify-center">
        {[{id:"wall",label:"Wall"},{id:"columns",label:"Columns"},{id:"eraser",label:"Eraser"}].map(b=>(
          <button key={b.id} onClick={()=>setTool(b.id)} className={`p-2 rounded shadow 2xl:rounded-2xl ${tool===b.id?"bg-blue-600 text-white":"bg-gray-200"}`}>{b.label}</button>
        ))}
        <button onClick={optimizeDrawing} className="p-2 bg-green-600 text-white rounded shadow 2xl:rounded-2xl">Optimize</button>
        <button onClick={clearCanvas} className="p-2 bg-red-500 text-white rounded shadow 2xl:rounded-2xl">Clear</button>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <label className="whitespace-nowrap">Looseness:</label>
        <input type="range" min="1" max="10" value={epsFactor} onChange={e=>setEpsFactor(parseInt(e.target.value,10))} className="accent-blue-600 w-48" />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
}
