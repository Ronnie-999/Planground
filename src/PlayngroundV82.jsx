  import React, { useRef, useState, useEffect } from "react";
import * as pdfjsLib from "pdfjs-dist";
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.mjs",   // ✅ exists in v4.x
  import.meta.url
).toString();




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
  const [offsetVisuals, setOffsetVisuals] = useState([]);

  const [gridCount, setGridCount] = useState(1);
  const [bgImage, setBgImage] = useState(null);

  const handlePdfUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const fileReader = new FileReader();
    fileReader.onload = async function () {
      const typedarray = new Uint8Array(this.result);

      const pdf = await pdfjsLib.getDocument({ data: typedarray }).promise;
      const page = await pdf.getPage(1);

      const originalViewport = page.getViewport({ scale: 1 });

      // Compute scale to fit PDF page within canvas
      const scale = Math.min(
        canvasWidth / originalViewport.width,
        canvasHeight / originalViewport.height
      );

      const viewport = page.getViewport({ scale });
      const offscreenCanvas = document.createElement("canvas");
      const context = offscreenCanvas.getContext("2d");
      offscreenCanvas.width = viewport.width;
      offscreenCanvas.height = viewport.height;

      await page.render({ canvasContext: context, viewport }).promise;

      setBgImage(offscreenCanvas);


    };

    fileReader.readAsArrayBuffer(file);
  };

  const deleteBackground = () => {
    setBgImage(null);
  };












  const [view, setView]         = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [tool, setTool]         = useState("wall");
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokePts, setStrokePts] = useState([]);
  const [mode, setMode] = useState("primary"); // or "secondary"
  

  


  const [primaryColumns, setPrimaryColumns] = useState([]);
  const [primaryWalls, setPrimaryWalls] = useState([]);
  const [secondaryColumns, setSecondaryColumns] = useState([]);
  const [secondaryWalls, setSecondaryWalls] = useState([]);

  const [primaryDoors,   setPrimaryDoors]   = useState([]);
  const [secondaryDoors, setSecondaryDoors] = useState([]);


  const [primaryOptimized, setPrimaryOptimized] = useState(false);
  const [primaryIdealized, setPrimaryIdealized] = useState(false);

  const [secondaryOptimized, setSecondaryOptimized] = useState(false);


  



  const columns = mode === "primary" ? primaryColumns : secondaryColumns;
  const walls   = mode === "primary" ? primaryWalls   : secondaryWalls;

  const setColumns = mode === "primary" ? setPrimaryColumns : setSecondaryColumns;
  const setWalls   = mode === "primary" ? setPrimaryWalls   : setSecondaryWalls;



  const [primaryGrid, setPrimaryGrid] = useState(null);
  const [secondaryGrid, setSecondaryGrid] = useState(null);
  const [secondaryGridInherited, setSecondaryGridInherited] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [epsFactor, setEpsFactor] = useState(4);
  

   // ─── wall‑simplification tweakables (from Code F) ───────────────
 const RDP_EPS      = 8;     // px – tolerance for Ramer‑Douglas‑Peucker
 const JUNC_THRESH  = 12;    // px – node snapped only if this close

 const grid = mode === "primary" ? primaryGrid : secondaryGrid;
 const setGrid = mode === "primary" ? setPrimaryGrid : setSecondaryGrid;
 
 const hasOptimized = mode === "primary" ? primaryOptimized : secondaryOptimized;
 const doorEnabled =
  mode === "primary" ? primaryIdealized : hasOptimized;

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




function splitWallByBreakpoints(wall, breakPts) {
  if (!breakPts.length) return [wall];

  const dist = (a, b) => (a.x - b.x)**2 + (a.y - b.y)**2;

  // For each breakpoint, find the closest index in the wall polyline
  const indices = breakPts.map(bp => {
    let best = 0;
    let minD = Infinity;
    for (let i = 0; i < wall.length; i++) {
      const d = dist(bp, wall[i]);
      if (d < minD) {
        minD = d;
        best = i;
      }
    }
    return best;
  });

  // Sort and remove duplicates
  const unique = [...new Set(indices)].sort((a, b) => a - b);

  // Now split the wall
  const result = [];
  let startIdx = 0;
  for (let idx of unique) {
    if (idx > startIdx + 1) {
      result.push(wall.slice(startIdx, idx + 1));
      startIdx = idx;
    }
  }
  if (startIdx < wall.length - 1) {
    result.push(wall.slice(startIdx));
  }

  return result;
}




function offsetPolyline(stroke, offsetDist) {
  if (stroke.length < 2) return stroke.slice();

  // 1.  compute a unit normal for every individual segment
  const segNormals = [];
  for (let i = 0; i < stroke.length - 1; i++) {
    const dx = stroke[i + 1].x - stroke[i].x;
    const dy = stroke[i + 1].y - stroke[i].y;
    const len = Math.hypot(dx, dy) || 1;
    segNormals.push({               // ⟂ to segment (i → i+1)
      nx: -dy / len,
      ny:  dx / len,
      len                             // keep the segment length too
    });
  }

  // 2.  build the offset poly-line point-by-point
  const out = [];

  for (let i = 0; i < stroke.length; i++) {

    /* choose the normal of the *shorter* adjacent segment
       – guarantees it is perpendicular to the “smallest” local piece   */
    let nx, ny;

    if (i === 0) {                       // first point → use first segment
      ({ nx, ny } = segNormals[0]);

    } else if (i === stroke.length - 1) { // last point → use last segment
      ({ nx, ny } = segNormals[segNormals.length - 1]);

    } else {
      const prev = segNormals[i - 1];
      const next = segNormals[i];
      const usePrev = prev.len <= next.len;
      ({ nx, ny } = usePrev ? prev : next);
    }

    out.push({
      x: stroke[i].x + nx * offsetDist,
      y: stroke[i].y + ny * offsetDist
    });
  }

  return out;
}



function offsetPolylineLite(pts, offsetDist) {
  if (pts.length < 2) return pts.slice();

  const { dir } = principalAxis(pts); // Global axis
  const nx = -dir.y;
  const ny =  dir.x;

  return pts.map(p => ({
    x: p.x + nx * offsetDist,
    y: p.y + ny * offsetDist
  }));
}





         

function classifyFragment(stroke, offsetRatio = 0.3, lengthTolerance = 0.27) {
  if (stroke.length < 2) return { type: "line", offsets: [] };

  const totalLength = pts => {
    let sum = 0;
    for (let i = 1; i < pts.length; i++) {
      const dx = pts[i].x - pts[i - 1].x;
      const dy = pts[i].y - pts[i - 1].y;
      sum += Math.hypot(dx, dy);
    }
    return sum;
  };

  const baseLen = totalLength(stroke);
  const offsetMag = offsetRatio * baseLen;

  const isShort = stroke.length < 5;
  const perturbed = isShort
    ? offsetPolylineLite(stroke, offsetMag)
    : offsetPolyline(stroke, offsetMag);

  const offsetLen = totalLength(perturbed);
  const deviation = Math.abs(offsetLen - baseLen) / baseLen;
  const type = deviation < lengthTolerance ? "line" : "arc";

  return { type, offsets: stroke.map((pt, i) => ({ from: pt, to: perturbed[i] })) };
}




function findMidpointForArc(seg) {
  return seg[Math.floor(seg.length / 2)];
}



const distToSeg2 = (p, a, b) => {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx*dx + dy*dy || 1;
  let t = ((p.x - a.x)*dx + (p.y - a.y)*dy) / len2;
  t = Math.max(0, Math.min(1, t));                      // clamp to segment
  const px = a.x + t*dx, py = a.y + t*dy;
  return (p.x - px)**2 + (p.y - py)**2;
};




/* --- smallest circle through 3 pts --------------------------------- */
function circleFrom3(a, b, c) {
  const A = b.x - a.x, B = b.y - a.y;
  const C = c.x - a.x, D = c.y - a.y;
  const E = A * (a.x + b.x) + B * (a.y + b.y);
  const F = C * (a.x + c.x) + D * (a.y + c.y);
  const G = 2 * (A * (c.y - b.y) - B * (c.x - b.x));
  if (Math.abs(G) < 1e-6) return null;          // almost colinear
  return {                           // centre & radius
    cx: (D * E - B * F) / G,
    cy: (A * F - C * E) / G
  };
}






// === Logic-(3) geometry =========================================
function circumcenter(A, B, C) {
  const [x1,y1] = [A.x, A.y], [x2,y2] = [B.x, B.y], [x3,y3] = [C.x, C.y];
  const d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
  if (Math.abs(d) < 1e-9) return null;
  const ux = ((x1**2+y1**2)*(y2-y3) + (x2**2+y2**2)*(y3-y1)
             +(x3**2+y3**2)*(y1-y2)) / d;
  const uy = ((x1**2+y1**2)*(x3-x2) + (x2**2+y2**2)*(x1-x3)
             +(x3**2+y3**2)*(x2-x1)) / d;
  return [ux, uy];
}

const normAngle = a => ((a % (2*Math.PI)) + 2*Math.PI) % (2*Math.PI);

/** buildArcThroughMid(p0, pm, p1 [,steps])
 *  Returns a sampled arc that starts at p0, ends at p1,
 *  and necessarily passes through pm */
function buildArcThroughMid(p0, pm, p1, steps = 64) {
  const cen = circumcenter(p0, pm, p1);
  if (!cen) return [p0, p1];

  const [cx, cy] = cen, R = Math.hypot(p0.x-cx, p0.y-cy);
  const ang = pt => Math.atan2(pt.y-cy, pt.x-cx);

  let a0 = normAngle(ang(p0));
  let am = normAngle(ang(pm));
  let a1 = normAngle(ang(p1));

  let cw = !((a0 < am && am < a1) || (a0 > a1 && (am > a0 || am < a1)));
  const span = cw ? (a0 - a1 + 2*Math.PI) % (2*Math.PI)
                  : (a1 - a0 + 2*Math.PI) % (2*Math.PI);

  const arc = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = cw ? a0 - span*t : a0 + span*t;
    arc.push({ x: cx + R*Math.cos(theta), y: cy + R*Math.sin(theta) });
  }
  return arc;
}




const snapToPrimaryGrid = (pt, gridList) => {
  let best = pt;
  let bestDist = Infinity;

  for (const grid of gridList) {
    if (grid.type === "linear") {
      const { xs, ys, dirA, dirB } = grid;
      const dot = (a, b) => a.x * b.x + a.y * b.y;
      const projA = dot(pt, dirA);
      const projB = dot(pt, dirB);

      xs.forEach(xv => {
        const candidate = {
          x: dirA.x * xv + dirB.x * projB,
          y: dirA.y * xv + dirB.y * projB
        };
        const d = (candidate.x - pt.x)**2 + (candidate.y - pt.y)**2;
        if (d < bestDist) { bestDist = d; best = candidate; }
      });

      ys.forEach(yv => {
        const candidate = {
          x: dirA.x * projA + dirB.x * yv,
          y: dirA.y * projA + dirB.y * yv
        };
        const d = (candidate.x - pt.x)**2 + (candidate.y - pt.y)**2;
        if (d < bestDist) { bestDist = d; best = candidate; }
      });
    }

    if (grid.type === "arc") {
      const { cx, cy, radii, angles } = grid;
      const dx = pt.x - cx, dy = pt.y - cy;
      const r = Math.hypot(dx, dy);
      const theta = Math.atan2(dy, dx);

      radii.forEach(rv => {
        const candidate = {
          x: cx + rv * Math.cos(theta),
          y: cy + rv * Math.sin(theta)
        };
        const d = (candidate.x - pt.x)**2 + (candidate.y - pt.y)**2;
        if (d < bestDist) { bestDist = d; best = candidate; }
      });

      angles.forEach(ang => {
        const candidate = {
          x: cx + r * Math.cos(ang),
          y: cy + r * Math.sin(ang)
        };
        const d = (candidate.x - pt.x)**2 + (candidate.y - pt.y)**2;
        if (d < bestDist) { bestDist = d; best = candidate; }
      });
    }
  }

  return best;
};




const segmentLength = pts => {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    len += Math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y);
  }
  return len;
};

const buildArcWithRadius = (cx, cy, r, pStart, pEnd, steps = 36) => {
  const angle = p => Math.atan2(p.y - cy, p.x - cx);
  let a0 = angle(pStart), a1 = angle(pEnd);
  if (a1 < a0) a1 += 2 * Math.PI;
  const arcPts = [];
  for (let i = 0; i <= steps; i++) {
    const t = a0 + (a1 - a0) * (i / steps);
    arcPts.push({ x: cx + r * Math.cos(t), y: cy + r * Math.sin(t) });
  }
  return arcPts;
};



// --- door helpers -------------------------------------------------
const pointAlongPolyline = (pts, ratio) => {
  const total = segmentLength(pts);
  let acc = 0;
  for (let i = 1; i < pts.length; i++) {
    const seg = Math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y);
    if (acc + seg >= ratio * total) {
      const t = (ratio * total - acc) / seg;
      return {
        x: pts[i-1].x + t * (pts[i].x - pts[i-1].x),
        y: pts[i-1].y + t * (pts[i].y - pts[i-1].y)
      };
    }
    acc += seg;
  }
  return pts[pts.length - 1];
};

const closestPointRatio = (pts, p) => {
  let best = 0, bestDist = Infinity, acc = 0;
  const total = segmentLength(pts);
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i-1], b = pts[i];
    const dx = b.x - a.x, dy = b.y - a.y;
    const len2 = dx*dx + dy*dy || 1;
    const t = Math.max(0, Math.min(1, ((p.x-a.x)*dx + (p.y-a.y)*dy) / len2));
    const proj = { x: a.x + t*dx, y: a.y + t*dy };
    const d2 = (proj.x - p.x)**2 + (proj.y - p.y)**2;
    if (d2 < bestDist) {
      bestDist = d2;
      best     = (acc + t * Math.hypot(dx,dy)) / total;
    }
    acc += Math.hypot(dx,dy);
  }
  return best;
};

const remapDoors = (doors, oldWalls, newWalls) =>
  doors.map(d => {
    const oldPt = pointAlongPolyline(oldWalls[d.wall].pts||oldWalls[d.wall], d.ratio);
    let bestWall = 0, bestRatio = 0, bestDist = Infinity;
    newWalls.forEach((w,i) => {
      const pts = w.pts || w;
      const r   = closestPointRatio(pts, oldPt);
      const np  = pointAlongPolyline(pts, r);
      const d2  = (np.x - oldPt.x)**2 + (np.y - oldPt.y)**2;
      if (d2 < bestDist) { bestDist = d2; bestWall = i; bestRatio = r; }
    });
    return { wall: bestWall, ratio: bestRatio };
  });






// Logic-(2) helpers
function circumcenter(A, B, C) {
  const [x1, y1] = [A.x, A.y],
        [x2, y2] = [B.x, B.y],
        [x3, y3] = [C.x, C.y];
  const d = 2 * (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
  if (Math.abs(d) < 1e-9) return null;
  const ux = ((x1**2 + y1**2)*(y2-y3) + (x2**2 + y2**2)*(y3-y1) +
              (x3**2 + y3**2)*(y1-y2)) / d;
  const uy = ((x1**2 + y1**2)*(x3-x2) + (x2**2 + y2**2)*(x1-x3) +
              (x3**2 + y3**2)*(x2-x1)) / d;
  return [ux, uy];
}

const normalizeAngle = a => ((a % (2*Math.PI)) + 2*Math.PI) % (2*Math.PI);

/** buildArcThroughMid(p0, pm, p1, steps)
 *  Guarantees the arc passes through `pm`
 *  and follows the shorter direction from p0 → p1 that hits pm */
function buildArcThroughMid(p0, pm, p1, steps = 64) {
  const cen = circumcenter(p0, pm, p1);
  if (!cen) return [p0, p1];

  const [cx, cy] = cen;
  const R = Math.hypot(p0.x - cx, p0.y - cy);
  const ang = (pt) => Math.atan2(pt.y - cy, pt.x - cx);
  let a0 = normalizeAngle(ang(p0));
  let am = normalizeAngle(ang(pm));
  let a1 = normalizeAngle(ang(p1));

  let clockwise = false;
  const goesThroughMid =
        (a0 < am && am < a1) || (a0 > a1 && (am > a0 || am < a1));
  if (!goesThroughMid) clockwise = true;

  const delta = clockwise
      ? (a0 - a1 + 2*Math.PI) % (2*Math.PI)
      : (a1 - a0 + 2*Math.PI) % (2*Math.PI);

  const arc = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const theta = clockwise ? a0 - delta*t : a0 + delta*t;
    arc.push({ x: cx + R*Math.cos(theta), y: cy + R*Math.sin(theta) });
  }
  return arc;
}










  const [history, setHistory] = useState([]);
  const [future,  setFuture]  = useState([]);

  const [moveSel, setMoveSel] = useState(null); // { kind, index, prev }

  const deepCopy = obj => JSON.parse(JSON.stringify(obj));
  // --- clone every field of every grid-definition (breaks shared refs) ----
  const cloneGrid = gridArr =>
    gridArr.map(g => ({
      ...g,
      xs: [...g.xs],
      ys: [...g.ys]
    }));


  // ─── history ───────────────────────────────────────────────────────────────
  const recordState = () => {
    setHistory(h => [
      ...h,
      deepCopy({
        primaryColumns, primaryWalls, primaryDoors,
        secondaryColumns, secondaryWalls, secondaryDoors,
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
        primaryColumns, primaryWalls, primaryDoors,
        secondaryColumns, secondaryWalls, secondaryDoors,
        primaryGrid, secondaryGrid
      })
    ]);

    setPrimaryColumns(prev.primaryColumns);
    setPrimaryWalls(prev.primaryWalls);
    setPrimaryDoors(prev.primaryDoors);
    setSecondaryColumns(prev.secondaryColumns);
    setSecondaryWalls(prev.secondaryWalls);
    setSecondaryDoors(prev.secondaryDoors);
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
          primaryColumns, primaryWalls, primaryDoors,
          secondaryColumns, secondaryWalls, secondaryDoors,
          primaryGrid, secondaryGrid
        })
      ]);

      setPrimaryColumns(nxt.primaryColumns);
      setPrimaryWalls(nxt.primaryWalls);
      setPrimaryDoors(nxt.primaryDoors);
      setSecondaryColumns(nxt.secondaryColumns);
      setSecondaryWalls(nxt.secondaryWalls);
      setSecondaryDoors(nxt.secondaryDoors);
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
    const newScale = prev.scale * factor;


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
    if (offs.length < 2) return [0, offs];      // guarantees at least one snap line

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
  
    const projectOntoGridLines = (pt, xs, ys, dirA, dirB) => {
      const dot = (p, q) => p.x * q.x + p.y * q.y;

      const projA = dot(pt, dirA);
      const projB = dot(pt, dirB);

      const nearestA = xs.reduce((acc, v) =>
        Math.abs(v - projA) < Math.abs(acc - projA) ? v : acc, xs[0]
      );
      const nearestB = ys.reduce((acc, v) =>
        Math.abs(v - projB) < Math.abs(acc - projB) ? v : acc, ys[0]
      );

      return {
        x: dirA.x * nearestA + dirB.x * nearestB,
        y: dirA.y * nearestA + dirB.y * nearestB
      };
    };

    // Apply projection snapping
    const newColumns = secondaryColumns.map(col =>
      projectOntoGridLines(col, useXs, useYs, dirA, dirB)
    );

    const newWalls = secondaryWalls.map(stk =>
      stk.map(pt => projectOntoGridLines(pt, useXs, useYs, dirA, dirB))
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

    // ✅ Check if primary grid can be used
    const canSnapToPrimaryGrid =
      primaryIdealized &&
      primaryGrid;


    if (canSnapToPrimaryGrid && Array.isArray(primaryGrid)) {
      const newColumns = secondaryColumns.map(pt => snapToPrimaryGrid(pt, primaryGrid));
      const newWalls = secondaryWalls.map(wall =>
        wall.map(pt => snapToPrimaryGrid(pt, primaryGrid))
      );

      setColumns(newColumns);

      const newDoorSet = remapDoors(secondaryDoors, secondaryWalls, newWalls);
      setSecondaryDoors(newDoorSet);



      setWalls(newWalls);
      const clonedGrid = primaryGrid.map(g => ({ ...g, xs: [...g.xs], ys: [...g.ys] }));
      setGrid(cloneGrid(primaryGrid));  // <- now the secondary gets its own copy
      setSecondaryGridInherited(true);   // 🆕 mark this as inherited

      setShowGrid(true);
      setHasOptimized(true);
      return;
    }





  
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
      const seg = flatWalls.slice(k, k + stk.length);
      k += stk.length;
      return seg.length >= 2 ? seg : null;
    }).filter(Boolean); // Remove null or too-short segments

  
    setColumns(newColumns);
    setWalls(newWalls);
    setGrid({ xs, ys, dirA, dirB });
    setShowGrid(true);
    setHasOptimized(true);
    setSecondaryGridInherited(false);   // 🆕 this was generated in secondary mode, not inherited

  };



  


  const idealizePrimaryWalls = () => {
    if (mode !== "primary" || !primaryWalls.length) return;

    const unidealizedWalls = primaryWalls.filter(w => !w.idealised);
    if (!unidealizedWalls.length) return;

    recordState();

    // Keep already idealized walls as-is
    const output = primaryWalls.filter(w => w.idealised);

    for (const wall of unidealizedWalls) {
      const raw = wall.raw || wall.pts || wall;
      const breaks = fragmentPolyline(raw);
      const segments = splitWallByBreakpoints(raw, breaks);

      const fragments = [];

      for (const seg of segments) {
        if (seg.length < 3) {
          fragments.push({ pts: seg, kind: "free", idealised: true });
          continue;
        }

        const { type } = classifyFragment(seg);

        if (type === "line") {
          fragments.push({
            pts: [seg[0], seg[seg.length - 1]],
            kind: "line",
            idealised: true
          });
        } else {
          const p0 = seg[0];
          const pm = findMidpointForArc(seg);
          const p1 = seg[seg.length - 1];
          const arcPts = buildArcThroughMid(p0, pm, p1, 64);
          fragments.push({
            pts: arcPts,
            kind: "arc",
            idealised: true
          });
        }
      }

      // === Local post-processing of short segments within this wall ===
      const cleaned = [];
      let i = 0;
      while (i < fragments.length) {
        const cur = fragments[i];
        const len = segmentLength(cur.pts || []);
        const left = i > 0 ? segmentLength(fragments[i - 1].pts || []) : Infinity;
        const right = i < fragments.length - 1 ? segmentLength(fragments[i + 1].pts || []) : Infinity;
        const maxNeighbor = Math.max(left, right);

        if (len < 0.1 * maxNeighbor && i > 0 && i < fragments.length - 1) {
          const prev = cleaned[cleaned.length - 1];
          const next = fragments[i + 1];

          const a = prev.pts.at(-2);
          const b = prev.pts.at(-1);
          const c = next.pts[0];
          const d = next.pts[1];

          const det = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
          if (Math.abs(det) > 1e-6) {
            const t = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / det;
            const ix = a.x + t * (b.x - a.x);
            const iy = a.y + t * (b.y - a.y);
            const intersection = { x: ix, y: iy };

            prev.pts[prev.pts.length - 1] = intersection;
            next.pts[0] = intersection;
          }

          i += 2;
          cleaned.push(next);
        } else {
          cleaned.push(cur);
          i += 1;
        }
      }


      // ➜ close the stroke if its ends nearly meet  (< 10 % of its perimeter)
      const gap = Math.hypot(
        raw[0].x - raw.at(-1).x,
        raw[0].y - raw.at(-1).y
      );
      if (gap < 0.05 * segmentLength(raw)) {
        cleaned.push({ pts:[ raw.at(-1), raw[0] ], kind:"line", idealised:true });
      }


      // Push each cleaned fragment as a separate wall (preserving independence)
      for (const frag of cleaned) {
        output.push(frag);
      }
    }

    const newDoorSet = remapDoors(primaryDoors, primaryWalls, output);
    setPrimaryDoors(newDoorSet);


    setPrimaryWalls(output);
    setPrimaryIdealized(true);
  };












  const findPrimaryGrid = (gridCount = 1) => {
    if (mode !== "primary" || !primaryIdealized) return;

    const idealSegments = primaryWalls
      .filter(w => w.idealised && w.pts?.length >= 2);
    if (!idealSegments.length) return;

    // Sort by descending length
    const sorted = [...idealSegments].sort(
      (a, b) => segmentLength(b.pts) - segmentLength(a.pts)
    );

    const allPrimary = primaryWalls
      .map(w => w.pts || w)
      .filter(pts => pts?.length >= 2);
    const avgWallLen = allPrimary.reduce((s, w) => s + segmentLength(w), 0) / allPrimary.length;
    const step = 0.1 * avgWallLen;
    const count = 20;
    const rayCount = 120;

    const offsets = [];

    // Optional: use the first grid for snapping
    let xs = [], ys = [], dirA = null, dirB = null;
    const gridDefs = [];


    for (let i = 0; i < Math.min(gridCount, sorted.length); i++) {
      const seg = sorted[i];

      if (seg.kind === "arc") {
        const arcPts = seg.pts;
        const p0 = arcPts[0];
        const p1 = arcPts[arcPts.length - 1];
        const pm = arcPts[Math.floor(arcPts.length / 2)];
        const cen = circleFrom3(p0, pm, p1);


        if (!cen) continue;

        const r0 = Math.hypot(p0.x - cen.cx, p0.y - cen.cy);

        if (i === 0 && cen) {
          const radii = Array.from({ length: 2 * count + 1 }, (_, j) => r0 + (j - count) * step);
          const angles = Array.from({ length: rayCount }, (_, j) => (2 * Math.PI * j) / rayCount);
          gridDefs.push({ type: "arc", cx: cen.cx, cy: cen.cy, radii, angles });
        }


        // Circles
        for (let j = -count; j <= count; j++) {
          const r = r0 + j * step;
          const circle = Array.from({ length: 129 }, (_, k) => {
            const t = (2 * Math.PI * k) / 128;
            return {
              x: cen.cx + r * Math.cos(t),
              y: cen.cy + r * Math.sin(t)
            };
          });
          offsets.push({ to: circle, color: "rgba(0,0,0,0.08)" });
        }

        // Rays
        for (let j = 0; j < rayCount; j++) {
          const angle = (2 * Math.PI * j) / rayCount;
          const p = {
            x: cen.cx + (r0 + count * step) * Math.cos(angle),
            y: cen.cy + (r0 + count * step) * Math.sin(angle)
          };
          offsets.push({ to: [{ x: cen.cx, y: cen.cy }, p], color: "rgba(0,0,0,0.08)" });
        }

        // For snapping grid (just once)
        if (i === 0) {
          xs = [0];
          ys = [0];
          dirA = { x: 1, y: 0 };
          dirB = { x: 0, y: 1 };
        }

      } else if (seg.kind === "line") {
        const pts = seg.pts;
        if (pts.length < 2) continue;

        const axis = principalAxis(pts);
        const dir = axis.dir;
        const nx = -dir.y, ny = dir.x;
        const mid = pts[Math.floor(pts.length / 2)];

        // Parallel offsets
        for (let j = -count; j <= count; j++) {
          const dx = j * step * nx;
          const dy = j * step * ny;
          const offset = pts.map(p => ({
            x: p.x + dx,
            y: p.y + dy
          }));
          offsets.push({ to: offset, color: "rgba(0,0,0,0.08)" });
        }

        // Perpendicular lines
        for (let j = -count; j <= count; j++) {
          const dx = j * step * dir.x;
          const dy = j * step * dir.y;

          const p1 = {
            x: mid.x + dx - nx * 2000,
            y: mid.y + dy - ny * 2000
          };
          const p2 = {
            x: mid.x + dx + nx * 2000,
            y: mid.y + dy + ny * 2000
          };
          offsets.push({ to: [p1, p2], color: "rgba(0,0,0,0.08)" });
        }

        // For snapping grid (just once)
        if (i === 0) {
          xs = Array.from({ length: 2 * count + 1 }, (_, i) => (i - count) * step);
          ys = [0];
          dirA = { x: nx, y: ny };
          dirB = { x: dir.x, y: dir.y };
        }
        if (i === 0) {
          gridDefs.push({ type: "linear", xs, ys, dirA, dirB });
        }



      }
    }
    if (gridDefs.length) {
      setPrimaryGrid(gridDefs);
      setShowGrid(true);
    }


    setOffsetVisuals(offsets);


  };













  




  
  
  const pickObject = scr => {
    const wpt = screenToWorld(scr);
    const hitTest = (a, b, r = 8 / view.scale) =>
      (a.x - b.x) ** 2 + (a.y - b.y) ** 2 < r * r;

    // Check columns first
    for (let i = 0; i < columns.length; i++) {
      if (hitTest(columns[i], wpt)) return { kind: "column", index: i };
    }

    // Then check walls
    // --- Walls ---------------------------------------------------------
    for (let i = 0; i < walls.length; i++) {
      const pts = walls[i].pts || walls[i];
      // quick reject: endpoints first (cheap) ---------------
      if (pts.some(p => hitTest(p, wpt))) return { kind:"wall", index:i };

      // full segment check ----------------------------------
      for (let j = 0; j < pts.length - 1; j++) {
        if (distToSeg2(wpt, pts[j], pts[j+1]) < (8/view.scale)**2)
          return { kind:"wall", index:i };
      }
    }


    return null;
  };




  const eraseAt = wpt => {
    const r = Math.max(10 / view.scale, 10);
    const hitTest = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2 < r * r;

    let erased = false;

    // Erase columns
    setColumns(cols => {
      const next = cols.filter(p => !hitTest(p, wpt));
      erased = erased || next.length !== cols.length;
      return next;
    });

    // Erase entire walls if any point is hit
    setWalls(walls => {
      const next = walls.filter(wall => {
        const pts = wall.pts || wall;
        return !pts.some(p => hitTest(p, wpt));
      });

      erased = erased || next.length !== walls.length;
      return next;
    });

    if (erased) recordState();
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


    // ─── Door ──────────────────────────────────────────────────────────
    if (tool === "door") {
      if (!doorEnabled) return;                 // guard – grid not ready yet?

      const pick = pickObject(scr);             // which wall did we click?
      if (!pick || pick.kind !== "wall") return;

      const wallPts = walls[pick.index].pts || walls[pick.index];
      const ratio   = closestPointRatio(wallPts, wpt);   // precise spot along it

      if (mode === "primary") {
        setPrimaryDoors(ds => [...ds, { wall: pick.index, ratio }]);
      } else {
        setSecondaryDoors(ds => [...ds, { wall: pick.index, ratio }]);
      }
      renderCanvas();                            // show the new square immediately
      return;                                    // stop the rest of startDrawing()
    }





    





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

    if (bgImage) {
      ctx.drawImage(bgImage, 0, 0);
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
          ws.map((stk, i) => {
            if (i !== index) return stk;

            const pts = stk.pts || stk;
            const moved = pts.map(pt => ({ x: pt.x + dx, y: pt.y + dy }));

            return stk.pts
              ? { ...stk, pts: moved }
              : moved;
          })
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
      const denseStroke = resampleLine(strokePts, 4);
      const simplified = simplifyWall(denseStroke);
  
    if (mode === "primary") {
      const wallEntry = {
        pts: simplified,
        raw: denseStroke // ✅ store original for later idealization
      };
      setPrimaryWalls(walls => [...walls, wallEntry]);
    } else {
      setSecondaryWalls(walls => [...walls, simplified]);
    }

    }
  
    if (tool === "columns") {
      const cx = strokePts.reduce((s, p) => s + p.x, 0) / strokePts.length;
      const cy = strokePts.reduce((s, p) => s + p.y, 0) / strokePts.length;
      const newCol = { x: cx, y: cy };
  
      if (mode === "primary") {
        setPrimaryColumns(cols => [...cols, newCol]);
      } else {
        setSecondaryColumns(cols => [...cols, newCol]);
      }
    }
  
    setStrokePts([]);
  };
  
  







  const clearCanvas = () => {
    recordState();
    if (mode === "primary") {
      // wipe everything primary
      setPrimaryColumns([]);
      setPrimaryWalls([]);
      setPrimaryDoors([]);

      setOffsetVisuals([]);
      setPrimaryGrid(null);
    } else {
      // wipe everything secondary
      setSecondaryColumns([]);
      setSecondaryWalls([]);
      setSecondaryDoors([]);

      setOffsetVisuals([]);
      // only clear the secondary grid if it wasn't inherited
      if (!secondaryGridInherited) {
        setSecondaryGrid(null);
      }
    }
  };







  
  const renderCanvas = (tempStroke = null, isWall = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");





    // Clear canvas and draw frame
    // Clear canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── background PDF (world space) ──
    ctx.save();
    ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);
    if (bgImage) ctx.drawImage(bgImage, 0, 0);  // ⬅️ now follows pan/zoom
    ctx.restore();

    drawFrame(ctx, canvas);                     // keep the black border

    // Apply zoom/pan transform for the rest
    ctx.save();
    ctx.setTransform(view.scale, 0, 0, view.scale, view.offsetX, view.offsetY);


    // ─── Grid ─────────────────────────────
    const shouldRenderGrayGrid = showGrid && !offsetVisuals.length;
    if (grid && shouldRenderGrayGrid) {
      const { xs, ys, dirA, dirB } = grid;
      ctx.lineWidth = 1 / view.scale;
      ctx.strokeStyle = "rgba(0,0,0,0.08)";

      xs.forEach(x => {
        ctx.beginPath();
        ctx.moveTo(dirA.x * x + dirB.x * -10000, dirA.y * x + dirB.y * -10000);
        ctx.lineTo(dirA.x * x + dirB.x * 10000, dirA.y * x + dirB.y * 10000);
        ctx.stroke();
      });

      ys.forEach(y => {
        ctx.beginPath();
        ctx.moveTo(dirB.x * y + dirA.x * -10000, dirB.y * y + dirA.y * -10000);
        ctx.lineTo(dirB.x * y + dirA.x * 10000, dirB.y * y + dirA.y * 10000);
        ctx.stroke();
      });
    }


    // ─── Walls ─────────────────────────────
    const renderWalls = (walls, isActive, fallbackColor) => {
      ctx.lineWidth = 2 / view.scale;
      ctx.lineCap = "round";

      walls.forEach(wall => {
        const pts = wall?.pts?.length >= 2 ? wall.pts : (Array.isArray(wall) && wall.length >= 2 ? wall : null);
        if (!pts) return;


        let color = fallbackColor;
        if (isActive) {
          if (wall.idealised)          color = "black"; // ← priority
          else if (wall.kind === "arc") color = "red";
          else if (wall.kind === "line") color = "blue";
        }



        ctx.strokeStyle = color;
        ctx.beginPath();

        // 🔁 If arc and has exactly 3 points: generate smooth arc
        if (wall.kind === "arc") {
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);             // arc points are pre-computed
          }
        } else {
          ctx.moveTo(pts[0].x, pts[0].y);
          pts.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
        }


        ctx.stroke();
      });
    };


    // ─── Columns ───────────────────────────
    const renderColumns = (cols, color) => {
      ctx.fillStyle = color;
      cols.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4 / view.scale, 0, 2 * Math.PI);
        ctx.fill();
      });
    };


    const renderDoors = (doors, wallSet) => {
      ctx.fillStyle = "black";
      doors.forEach(d => {
        const wall = wallSet[d.wall];
        if (!wall) return;
        const p = pointAlongPolyline(wall.pts || wall, d.ratio);
        const s = 6 / view.scale;
        ctx.fillRect(p.x - s/2, p.y - s/2, s, s);
      });
    };







    // ─── Draw Layers ───────────────────────
    if (mode === "primary") {
      renderWalls(secondaryWalls, false, "rgba(0,0,0,0.2)");
      renderColumns(secondaryColumns, "rgba(0,0,0,0.2)");
      renderWalls(primaryWalls, true, "black");
      renderColumns(primaryColumns, "black");
      renderDoors(primaryDoors, primaryWalls);
    } else {
      renderWalls(primaryWalls, false, "rgba(0,0,0,0.2)");
      renderColumns(primaryColumns, "rgba(0,0,0,0.2)");
      renderWalls(secondaryWalls, true, "black");
      renderColumns(secondaryColumns, "black");
      renderDoors(secondaryDoors, secondaryWalls);
    }

    // ─── Temp Stroke ───────────────────────
    if (tempStroke?.length > 1) {
      ctx.beginPath();
      ctx.moveTo(tempStroke[0].x, tempStroke[0].y);
      tempStroke.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.strokeStyle = isWall ? "black" : "rgba(0,0,0,0.4)";
      ctx.stroke();
    }

    // ─── Breakpoints (for primary mode) ────
    if (mode === "primary") {
      ctx.fillStyle = "black";
      Object.values(wallBreaks).forEach(group =>
        group.forEach(p => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 6 / view.scale, 0, 2 * Math.PI);
          ctx.fill();
        })
      );
    }

    // ─── Offset Visuals (e.g. debug or grid lines) ───────
    offsetVisuals.forEach(({ to, color = "magenta" }) => {
      if (!to || to.length < 2) return;
      ctx.lineWidth = 1.5 / view.scale;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(to[0].x, to[0].y);
      to.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });

    ctx.restore();
  };

  
  
  useEffect(() => {
    if (bgImage) {
      renderCanvas();
    }
  }, [bgImage]);

  

  useEffect(() => {
    const isWall = tool === "wall";
    renderCanvas(strokePts.length > 1 ? strokePts : null, isWall);
  }, [
    primaryWalls, primaryColumns,
    secondaryWalls, secondaryColumns,
    strokePts, view, mode, grid,
    showGrid, showTempGridHighlight, wallBreaks, tool,
    bgImage 

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
        const newScale = Math.max(0.0001, prev.scale * factor);


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
  <div className="flex flex-col items-start pt-0 px-4 pb-4 font-space-grotesk">


    <div className="flex items-start mb-4">
      <img
        src="/logo-02.svg"
        alt="Planground Logo"
        style={{ width: '82px', height: '82px' }}
      />
      <h1 id="planground-title" className="text-3xl font-bold ml-4 m-0">
        PLANGROUND
      </h1>
    </div>




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
    {["wall", "columns", "door", "eraser", "move", "hand"].map(id => (
      <Btn
        key={id}
        active={tool === id}
        disabled={id === "door" && !doorEnabled}
        onClick={() => id !== "door" || doorEnabled ? setTool(id) : null}
      >
        {id === "move" ? "Move object" : id.charAt(0).toUpperCase() + id.slice(1)}
      </Btn>
    ))}



      <Btn onClick={() => document.getElementById("bg-pdf-input").click()}>
        Import Background
      </Btn>
      <input
        type="file"
        id="bg-pdf-input"
        accept="application/pdf"
        style={{ display: "none" }}
        onChange={handlePdfUpload}
      />

      <Btn onClick={deleteBackground} disabled={!bgImage}>
        Delete Background
      </Btn>



    <Btn
      active={false}
      disabled={mode !== "primary" || !primaryWalls.length}
      onClick={idealizePrimaryWalls}
    >
      Idealize
    </Btn>

    <Btn
      onClick={() => findPrimaryGrid(gridCount)}
      disabled={mode !== "primary" || !primaryIdealized}
    >
      Find grid
    </Btn>

    



    <Btn onClick={optimizeDrawing} disabled={mode === "primary"}>
      Optimize
    </Btn>

    <Btn onClick={rhythmize} disabled={mode === "primary" || !hasOptimized}>
      Rhythmizer
    </Btn>

    <Btn disabled={!history.length} onClick={undo}>
      Undo
    </Btn>

    <Btn disabled={!future.length} onClick={redo}>
      Redo
    </Btn>

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

    {/* Grid Count Slider */}
    <div className="mt-2 flex items-center gap-2">
      <label className="whitespace-nowrap"># of Grids:</label>
      <input
        type="range"
        min="1"
        max="4"
        value={gridCount}
        onChange={e => setGridCount(+e.target.value)}
        className="accent-blue-600 w-48"
      />
      <span>{gridCount}</span>
    </div>





  </div>
);

  
  
 }