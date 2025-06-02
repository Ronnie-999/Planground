import { useRef, useState, useEffect } from "react";

/**
 * Planground – grid‑aware optimiser with history
 * ▸ Tools: Wall / Columns / Eraser
 * ▸ Undo / Redo for any mutation (draw, optimise, clear)
 * ▸ Optimize → detect square grid, snap & (optionally) show faint grid over full canvas
 * ▸ Grid toggle to show/hide the detected grid
 * ▸ Looseness slider for clustering
 */
export default function Planground() {
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  /* drawing state */
  const [isDrawing, setIsDrawing]  = useState(false);
  const [tool, setTool]            = useState("wall"); // wall | columns | eraser
  const [currentStroke, setStroke] = useState([]);

  const [columns, setColumns]      = useState([]);      // [{x,y}]
  const [walls,   setWalls]        = useState([]);      // [[{x,y},…],…]
  const [grid,    setGrid]         = useState(null);    // {xs,ys,dirA,dirB}
  const [showGrid, setShowGrid]    = useState(true);

  /* optimiser */
  const [epsFactor, setEpsFactor]  = useState(4);

  /* history for Undo / Redo */
  const [history, setHistory] = useState([]);           // stack ← past states
  const [future,  setFuture]  = useState([]);           // stack → redo states

  // deep‑copy helper (structuredClone polyfill)
  const copy = (obj) => JSON.parse(JSON.stringify(obj));

  const recordState = () => {
    setHistory(h => [...h, copy({ columns, walls, grid })]);
    setFuture([]); // clear redo stack after fresh action
  };

  const undo = () => {
    setHistory(h => {
      if (!h.length) return h;           // nothing to undo
      const prev = h[h.length - 1];
      setFuture(f => [...f, copy({ columns, walls, grid })]);
      setColumns(prev.columns);
      setWalls(prev.walls);
      setGrid(prev.grid);
      return h.slice(0, -1);
    });
  };

  const redo = () => {
    setFuture(f => {
      if (!f.length) return f;           // nothing to redo
      const next = f[f.length - 1];
      setHistory(h => [...h, copy({ columns, walls, grid })]);
      setColumns(next.columns);
      setWalls(next.walls);
      setGrid(next.grid);
      return f.slice(0, -1);
    });
  };

  // ─── canvas init ───
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width  = window.innerWidth  * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
    document.title = "Planground";
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
      recordState();
      setWalls(w => [...w, currentStroke]);
      setStroke([]);
    }
  };

  /* clear */
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

  // ─── OPTIMISE & GRID DETECTION ───
  const optimizeDrawing = () => {
    const pts = [...columns, ...walls.flat()];
    if (pts.length < 4) return;

    recordState();

    const dot   = (a, b) => a.x * b.x + a.y * b.y;
    const norm  = v => Math.hypot(v.x, v.y);
    const vec   = (p, q) => ({ x: q.x - p.x, y: q.y - p.y });
    const angle = v => { const a = Math.atan2(v.y, v.x); return a < 0 ? a + Math.PI : a; };

    /* 1) nearest neighbour edges */
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

    /* 2) orientation histogram */
    const bins=180, hist=Array(bins).fill(0);
    edges.forEach(e=>{hist[Math.floor(angle(e.v)/Math.PI*bins)]++;});
    const idx1=hist.indexOf(Math.max(...hist));
    const hist2=hist.map((v,i)=>{const d=Math.min(Math.abs(i-idx1),bins-Math.abs(i-idx1));return d<=45?0:v;});
    const idx2=hist2.indexOf(Math.max(...hist2));
    let th1=(idx1+0.5)*Math.PI/bins, th2=(idx2+0.5)*Math.PI/bins;
    const delta=(((th2-th1+Math.PI)%Math.PI)-Math.PI/2);
    th1+=delta/2; th2-=delta/2;
    const dirA={x:Math.cos(th1),y:Math.sin(th1)};
    const dirB={x:Math.cos(th2),y:Math.sin(th2)};

    /* 3) projections */
    const xProj=pts.map(p=>dot(p,dirA));
    const yProj=pts.map(p=>dot(p,dirB));

    const medianGap=a=>{const u=[...new Set(a)].sort((a,b)=>a-b);if(u.length<2)return 1;const g=u.slice(1).map((v,i)=>v-u[i]).sort((a,b)=>a-b);return g[Math.floor(g.length/2)];};
    const cluster=(arr,eps)=>{const s=[...arr].sort((a,b)=>a-b);const out=[];let g=[s[0]];for(let i=1;i<s.length;i++){if(Math.abs(s[i]-s[i-1])<=eps)g.push(s[i]);else{out.push(g);g=[s[i]];}}out.push(g);return out.map(g=>g.reduce((s,v)=>s+v,0)/g.length);};

    const epsX=Math.max(medianGap(xProj)*epsFactor*0.2,1);
    const epsY=Math.max(medianGap(yProj)*epsFactor*0.2,1);
    const xs=cluster(xProj,epsX);
    const ys=cluster(yProj,epsY);

    /* 4) snap */
    const near=(arr,val)=>arr.reduce((p,c)=>Math.abs(c-val)<Math.abs(p-val)?c:p);
    const snapped=pts.map((p,i)=>{const sx=near(xs,xProj[i]), sy=near(ys,yProj[i]);return{ x:dirA.x*sx+dirB.x*sy, y:dirA.y*sx+dirB.y*sy};});

    const newCols=snapped.slice(0,columns.length);
    const flatWalls=snapped.slice(columns.length);
    const newWalls=[];let k=0;walls.forEach(stk=>{newWalls.push(flatWalls.slice(k,k+stk.length));k+=stk.length;});

    setColumns(newCols);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
  };

  // ─── render helper ───
  const renderCanvas = (cols=columns, ws=walls, g=grid) => {
    const canvas=canvasRef.current, ctx=canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    drawFrame(ctx,canvas);

    /* grid */
    if(g && showGrid){
      ctx.save();ctx.lineWidth=1;ctx.strokeStyle="rgba(0,0,0,0.2)";
      const reach=Math.hypot(canvas.width,canvas.height)+20;
      const {xs,ys,dirA,dirB}=g;
      xs.forEach(xc=>{ctx.beginPath();ctx.moveTo(dirA.x*xc-dirB.x*reach,dirA.y*xc-dirB.y*reach);ctx.lineTo(dirA.x*xc+dirB.x*reach,dirA.y*xc+dirB.y*reach);ctx.stroke();});
      ys.forEach(yc=>{ctx.beginPath();ctx.moveTo(dirB.x*yc-dirA.x*reach,dirB.y*yc-dirA.y*reach);ctx.lineTo(dirB.x*yc+dirA.x*reach,dirB.y*yc+dirA.y*reach);ctx.stroke();});
      ctx.restore();
    }

    /* walls */
    ctx.lineWidth=2;ctx.strokeStyle="black";ctx.lineCap="round";
    ws.forEach(stk=>{if(stk.length<2)return;ctx.beginPath();ctx.moveTo(stk[0].x,stk[0].y);stk.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));ctx.stroke();});

    /* columns */
    ctx.fillStyle="black";
    cols.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,4,0,Math.PI*2);ctx.fill();});
  };

  useEffect(()=>{renderCanvas(columns,walls,grid);},[columns,walls,grid,showGrid]);

  // ─── UI ───
  const Button = ({children,onClick,active=false,disabled=false})=>(
    <button onClick={onClick} disabled={disabled} className={`p-2 rounded shadow 2xl:rounded-2xl ${disabled?"bg-gray-100 text-gray-400":active?"bg-blue-600 text-white":"bg-gray-200"}`}>{children}</button>
  );

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

      {/* primary controls */}
      <div className="mt-4 flex gap-4 flex-wrap justify-center">
        {[
          { id: "wall", label: "Wall" },
          { id: "columns", label: "Columns" },
          { id: "eraser", label: "Eraser" },
        ].map(b => (
          <Button key={b.id} active={tool===b.id} onClick={()=>setTool(b.id)}>{b.label}</Button>
        ))}
        <Button onClick={optimizeDrawing}>Optimize</Button>
        <Button onClick={undo} disabled={!history.length}>Undo</Button>
        <Button onClick={redo} disabled={!future.length}>Redo</Button>
        <Button onClick={()=>setShowGrid(g=>!g)} disabled={!grid}>{showGrid?"Hide Grid":"Show Grid"}</Button>
        <Button onClick={clearCanvas}>Clear</Button>
      </div>

      {/* loosness slider */}
      <div className="mt-4 flex items-center gap-2">
        <label className="whitespace-nowrap">Looseness:</label>
        <input type="range" min="1" max="10" value={epsFactor} onChange={e=>setEpsFactor(parseInt(e.target.value,10))} className="accent-blue-600 w-48" />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
}
