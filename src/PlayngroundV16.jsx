import React, { useRef, useState, useEffect } from "react";

/**
 * PLANGROUND — two-layer sketcher + zoom + pan + eraser + optimizers
 * -------------------------------------------------------------
 *  • Tools: Wall, Columns, Eraser, Move, Hand-pan
 *  • Layers: Primary vs Secondary (toggle switch)
 *  • Recognize Shape (Primary only): RDP + micro-segment → ideal polylines/arcs
 *  • Optimize (Secondary only): snap to primary grid axes
 *  • Rhythmizer: snap grid pitch & both layers
 *  • Undo/Redo, Clear, Grid toggle, Looseness slider
 */
export default function Planground() {
  // ─── Refs ───────────────────────────────────────────────────────────
  const canvasRef = useRef(null);
  const ctxRef    = useRef(null);

  // ─── View / Zoom / Pan ──────────────────────────────────────────────
  const [view, setView]       = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [panning, setPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // ─── Tools / Modes ──────────────────────────────────────────────────
  const TOOLS = ["wall","columns","eraser","move","hand"];
  const [tool, setTool]       = useState("wall");
  const [layer, setLayer]     = useState("secondary"); // "primary" | "secondary"
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);

  // ─── Layer data ────────────────────────────────────────────────────
  const [primaryWalls,   setPrimaryWalls]   = useState([]);
  const [primaryCols,    setPrimaryCols]    = useState([]);
  const [secondaryWalls, setSecondaryWalls] = useState([]);
  const [secondaryCols,  setSecondaryCols]  = useState([]);

  // ─── Grid & Rhythm ──────────────────────────────────────────────────
  const [grid, setGrid]         = useState(null);
  const [showGrid, setShowGrid] = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);

  // ─── History ────────────────────────────────────────────────────────
  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);
  const deepCopy = o => JSON.parse(JSON.stringify(o));
  const recordState = () => {
    setHistory(h => [...h, deepCopy({
      primaryWalls, primaryCols,
      secondaryWalls, secondaryCols,
      grid
    })]);
    setFuture([]);
  };
  const undo = () => {
    if (!history.length) return;
    setHistory(h => {
      const prev = h.at(-1);
      setFuture(f => [...f, deepCopy({
        primaryWalls, primaryCols,
        secondaryWalls, secondaryCols,
        grid
      })]);
      setPrimaryWalls(prev.primaryWalls);
      setPrimaryCols(prev.primaryCols);
      setSecondaryWalls(prev.secondaryWalls);
      setSecondaryCols(prev.secondaryCols);
      setGrid(prev.grid);
      return h.slice(0,-1);
    });
  };
  const redo = () => {
    if (!future.length) return;
    setFuture(f => {
      const nxt = f.at(-1);
      setHistory(h => [...h, deepCopy({
        primaryWalls, primaryCols,
        secondaryWalls, secondaryCols,
        grid
      })]);
      setPrimaryWalls(nxt.primaryWalls);
      setPrimaryCols(nxt.primaryCols);
      setSecondaryWalls(nxt.secondaryWalls);
      setSecondaryCols(nxt.secondaryCols);
      setGrid(nxt.grid);
      return f.slice(0,-1);
    });
  };

  // ─── Coord Transforms ────────────────────────────────────────────────
  const screenToWorld = ({ x,y }) => ({
    x: (x - view.offsetX)/view.scale,
    y: (y - view.offsetY)/view.scale
  });
  const worldToScreen = ({ x,y }) => ({
    x: x*view.scale + view.offsetX,
    y: y*view.scale + view.offsetY
  });
  const getScreenXY = e => {
    if (e.nativeEvent instanceof TouchEvent) {
      const r = canvasRef.current.getBoundingClientRect();
      const t = e.nativeEvent.touches[0];
      return { x: t.clientX - r.left, y: t.clientY - r.top };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  // ─── Canvas Init ────────────────────────────────────────────────────
  useEffect(() => {
    const c = canvasRef.current;
    c.width  = window.innerWidth * 0.9;
    c.height = window.innerHeight * 0.8;
    drawFrame(c.getContext("2d"), c);
    document.title = "PLANGROUND";
  }, []);
  const drawFrame = (ctx, c) => {
    ctx.setTransform(1,0,0,1,0,0);
    ctx.lineWidth   = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0,0,c.width,c.height);
  };

  // ─── Wheel Zoom (canvas only) ────────────────────────────────────────
  const handleWheel = e => {
    e.stopPropagation(); e.preventDefault();
    const scr = getScreenXY(e);
    const f = e.deltaY<0?1.15:1/1.15;
    const s = view.scale * f;
    const { x:wx, y:wy } = screenToWorld(scr);
    setView({ scale: s,
      offsetX: scr.x - wx*s,
      offsetY: scr.y - wy*s
    });
  };

  // ─── Eraser: delete whole element ───────────────────────────────────
  const eraseAt = scr => {
    const wpt = screenToWorld(scr);
    const check = (wallsArr, colsArr, setWA, setCA) => {
      // columns
      let ci = colsArr.findIndex(c => (c.x-wpt.x)**2+(c.y-wpt.y)**2 < 64/view.scale**2);
      if (ci!==-1) { setCA(a=>a.filter((_,i)=>i!==ci)); return true; }
      // walls
      for (let i=0;i<wallsArr.length;i++){
        const stk = wallsArr[i];
        for (let j=0;j<stk.length-1;j++){
          const a=stk[j], b=stk[j+1];
          const t = Math.max(0,Math.min(1,
            ((wpt.x-a.x)*(b.x-a.x)+(wpt.y-a.y)*(b.y-a.y))/
            ((b.x-a.x)**2+(b.y-a.y)**2)
          ));
          const px=a.x+t*(b.x-a.x), py=a.y+t*(b.y-a.y);
          if ((px-wpt.x)**2+(py-wpt.y)**2<36/view.scale**2){
            setWA(a=>a.filter((_,ii)=>ii!==i));
            return true;
          }
        }
      }
      return false;
    };
    const primaryFirst = layer==="primary"
      ? [ [primaryWalls, setPrimaryWalls, primaryCols, setPrimaryCols],
          [secondaryWalls, setSecondaryWalls, secondaryCols, setSecondaryCols] ]
      : [ [secondaryWalls, setSecondaryWalls, secondaryCols, setSecondaryCols],
          [primaryWalls, setPrimaryWalls, primaryCols, setPrimaryCols] ];
    for (let [[wA,swA,cA,scA]] of [primaryFirst]) {
      if (check(wA,cA,swA,scA)) break;
    }
    renderCanvas();
  };

  // ─── Hand Pan ────────────────────────────────────────────────────────
  const panStartHandler = scr => { setPanning(true); setPanStart(scr); };
  const panMoveHandler = scr => {
    if (!panning) return;
    const dx = scr.x - panStart.x, dy = scr.y - panStart.y;
    setPanStart(scr);
    setView(v=>({
      scale: v.scale,
      offsetX: v.offsetX+dx,
      offsetY: v.offsetY+dy
    }));
  };
  const panEndHandler = ()=>setPanning(false);

  // ─── Recognize Shape (stub) ───────────────────────────────────────────
  const recognizeShape = () => {
    recordState();
    // TODO: implement RDP + micro-seg + arc detection on primaryWalls
    // e.g. setPrimaryWalls(ws=>ws.map(simplifyWall))
  };

  // ─── Secondary Optimize ──────────────────────────────────────────────
  const secondaryOptimize = () => {
    if (!grid) return;
    recordState();
    const { xs, ys, dirA, dirB } = grid;
    const ints = xs.flatMap(xv=> ys.map(yv=>({
      x: dirA.x*xv + dirB.x*yv,
      y: dirA.y*xv + dirB.y*yv
    })));
    // snap cols
    setSecondaryCols(cols=>cols.map(col=>{
      let best=ints[0], bd=Infinity;
      for (let pt of ints){
        const d=(pt.x-col.x)**2+(pt.y-col.y)**2;
        if(d<bd){bd=d; best=pt;}
      }
      return {x:best.x, y:best.y};
    }));
    // snap walls
    setSecondaryWalls(ws=>ws.map(stk=>
      stk.map(pt=>{
        let best=ints[0], bd=Infinity;
        for (let ip of ints){
          const d=(ip.x-pt.x)**2+(ip.y-pt.y)**2;
          if(d<bd){bd=d; best=ip;}
        }
        return {x:best.x, y:best.y};
      })
    ));
    renderCanvas();
  };

  // ─── Rhythmizer ───────────────────────────────────────────────────────
  const [recoverPitchAndOffsets] = [ // placeholder, reuse from above
    (offs,tol=0.25)=>[0,offs]
  ];
  const rhythmize = () => {
    if (!grid) return;
    recordState();
    // similar to secondaryOptimize but for grid pitch
    // TODO: implement actual pitch snapping
  };

  // ─── Drawing / Erase / Move / Pan Handlers ────────────────────────────
  const startDrawing = e => {
    e.preventDefault();
    const scr = getScreenXY(e);
    ctxRef.current = canvasRef.current.getContext("2d");
    recordState();

    if (tool==="hand") { panStartHandler(scr); return; }
    if (tool==="move") {
      const pick = pickObject(scr);
      if (!pick) return;
      setMoveSel({ ...pick, prev: scr });
      setIsDrawing(true);
      return;
    }
    if (tool==="eraser") { setIsDrawing(true); return; }

    // begin wall/columns
    const wpt = screenToWorld(scr);
    const ctx = ctxRef.current;
    ctx.save();
    ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);
    ctx.lineWidth   = 2/view.scale;
    ctx.lineCap     = "round";
    ctx.strokeStyle = tool==="columns"? "rgba(0,0,0,0.4)" : "black";
    ctx.beginPath(); ctx.moveTo(wpt.x,wpt.y);
    setStrokePts([wpt]);
    setIsDrawing(true);
    ctx.restore();
  };

  const draw = e => {
    const scr = getScreenXY(e);
    if (tool==="hand")  { panMoveHandler(scr); return; }
    if (!isDrawing)    return;
    if (tool==="eraser") { eraseAt(scr); return; }
    if (tool==="move")   { /* move logic */ return; }
    // continue stroke
    const wpt = screenToWorld(scr);
    setStrokePts(pts => [...pts,wpt]);
    renderCanvas();
  };

  const stopDrawing = e => {
    e.preventDefault();
    if (tool==="hand")  { panEndHandler(); return; }
    if (tool==="eraser"){ setIsDrawing(false); return; }
    if (tool==="move")  { setIsDrawing(false); setMoveSel(null); return; }

    setIsDrawing(false);
    if (tool==="wall") {
      if (layer==="primary")   setPrimaryWalls(ws => [...ws, strokePts]);
      else                     setSecondaryWalls(ws=>[...ws,strokePts]);
    }
    if (tool==="columns") {
      const cx = strokePts.reduce((s,p)=>s+p.x,0)/strokePts.length;
      const cy = strokePts.reduce((s,p)=>s+p.y,0)/strokePts.length;
      const nc = {x:cx,y:cy};
      if (layer==="primary")   setPrimaryCols(cs=>[...cs,nc]);
      else                     setSecondaryCols(cs=>[...cs,nc]);
    }
    setStrokePts([]);
  };

  const pickObject = scr => {
    const wpt = screenToWorld(scr);
    const layers = layer==="primary"
      ? [[primaryWalls,setPrimaryWalls,primaryCols,setPrimaryCols],
         [secondaryWalls,setSecondaryWalls,secondaryCols,setSecondaryCols]]
      : [[secondaryWalls,setSecondaryWalls,secondaryCols,setSecondaryCols],
         [primaryWalls,setPrimaryWalls,primaryCols,setPrimaryCols]];
    for (let [[wA, swA, cA, scA]] of [layers]) {
      for (let i=0;i<cA.length;i++){
        const c = cA[i];
        if ((c.x-wpt.x)**2+(c.y-wpt.y)**2<64/view.scale**2)
          return { kind:"column", index:i, cols:cA, setCols:scA };
      }
      for (let i=0;i<wA.length;i++){
        const stk=wA[i];
        for (let j=0;j<stk.length-1;j++){
          const a=worldToScreen(stk[j]), b=worldToScreen(stk[j+1]);
          const t= Math.max(0,Math.min(1,
            ((scr.x-a.x)*(b.x-a.x)+(scr.y-a.y)*(b.y-a.y))/
            ((b.x-a.x)**2+(b.y-a.y)**2)
          ));
          const px=a.x+t*(b.x-a.x), py=a.y+t*(b.y-a.y);
          if ((px-scr.x)**2+(py-scr.y)**2<36){
            return { kind:"wall", index:i, walls:wA, setWalls:swA };
          }
        }
      }
    }
    return null;
  };

  const clearCanvas = () => {
    recordState();
    setPrimaryWalls([]); setPrimaryCols([]);
    setSecondaryWalls([]); setSecondaryCols([]);
    setGrid(null);
  };

  // ─── Render ────────────────────────────────────────────────────────────
  const renderCanvas = () => {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    // clear + frame
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,c.width,c.height);
    drawFrame(ctx,c);

    // world → screen
    ctx.save();
    ctx.setTransform(view.scale,0,0,view.scale,view.offsetX,view.offsetY);

    // draw each layer (fading the other)
    const drawWalls=(arr,alpha)=>{
      ctx.globalAlpha=alpha;
      ctx.lineWidth=2/view.scale; ctx.strokeStyle="black";
      arr.forEach(stk=>{
        if(stk.length<2) return;
        ctx.beginPath(); ctx.moveTo(stk[0].x,stk[0].y);
        stk.slice(1).forEach(p=>ctx.lineTo(p.x,p.y)); ctx.stroke();
      });
    };
    const drawCols=(arr,alpha)=>{
      ctx.globalAlpha=alpha;
      ctx.fillStyle="black";
      arr.forEach(p=>{
        ctx.beginPath(); ctx.arc(p.x,p.y,4/view.scale,0,2*Math.PI); ctx.fill();
      });
    };
    drawWalls(primaryWalls,   layer==="secondary"?0.3:1);
    drawCols (primaryCols,    layer==="secondary"?0.3:1);
    drawWalls(secondaryWalls, layer==="primary"?0.3:1);
    drawCols (secondaryCols,  layer==="primary"?0.3:1);

    // preview stroke
    if(isDrawing && strokePts.length>1){
      ctx.globalAlpha=1;
      ctx.beginPath();
      ctx.moveTo(strokePts[0].x,strokePts[0].y);
      strokePts.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));
      ctx.strokeStyle=tool==="columns"?"rgba(0,0,0,0.4)":"black";
      ctx.lineWidth=2/view.scale;
      ctx.stroke();
    }

    ctx.globalAlpha=1;
    ctx.restore();
  };
  useEffect(renderCanvas, [
    primaryWalls, primaryCols,
    secondaryWalls, secondaryCols,
    grid, showGrid, view,
    strokePts, isDrawing, layer
  ]);

  // ─── UI ───────────────────────────────────────────────────────────────
  const Btn=({active,disabled,onClick,children})=>(
    <button disabled={disabled} onClick={onClick}
      className={`p-2 rounded-2xl shadow
        ${disabled?"bg-gray-100 text-gray-400"
          :active?"bg-blue-600 text-white":"bg-gray-200"}`}>
      {children}
    </button>
  );

  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <h1 id="planground-title" className="text-3xl font-bold mb-4">
        PLANGROUND
      </h1>

      {/* Layer toggle */}
      <div className="mb-2">
        <Btn active={layer==="primary"} onClick={()=>setLayer("primary")}>
          Primary Layer
        </Btn>
        <Btn active={layer==="secondary"} onClick={()=>setLayer("secondary")}>
          Secondary Layer
        </Btn>
      </div>

      {/* Recognize / Optimize */}
      <div className="mb-2">
        {layer==="primary" ? (
          <Btn onClick={recognizeShape}>Recognize Shape</Btn>
        ) : (
          <Btn onClick={secondaryOptimize} disabled={!grid}>
            Optimize Secondary
          </Btn>
        )}
        <Btn onClick={rhythmize} disabled={!grid}>
          Rhythmizer
        </Btn>
      </div>

      {/* Canvas */}
      <div className="border-4 border-black rounded-lg p-2 shadow-md" style={{touchAction:"none"}}>
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

      {/* Tools */}
      <div className="mt-4 flex flex-wrap gap-2 justify-center">
        {TOOLS.map(id=>(
          <Btn key={id} active={tool===id} onClick={()=>setTool(id)}>
            {id.charAt(0).toUpperCase()+id.slice(1)}
          </Btn>
        ))}
        <Btn onClick={undo}    disabled={!history.length}>Undo</Btn>
        <Btn onClick={redo}    disabled={!future.length}>Redo</Btn>
        <Btn onClick={clearCanvas}>Clear</Btn>
        <Btn onClick={()=>setShowGrid(g=>!g)} disabled={!grid}>
          {showGrid?"Hide Grid":"Show Grid"}
        </Btn>
      </div>

      {/* Looseness */}
      <div className="mt-4 flex items-center gap-2">
        <label>Looseness:</label>
        <input
          type="range" min="1" max="10"
          value={epsFactor}
          onChange={e=>setEpsFactor(+e.target.value)}
          className="accent-blue-600 w-48"
        />
        <span>{epsFactor}</span>
      </div>
    </div>
  );
}
