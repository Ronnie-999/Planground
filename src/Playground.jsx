import { useRef, useState, useEffect } from "react";

export default function Playground() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [simplificationFactor, setSimplificationFactor] = useState(5); // Degree of smoothing
  let points = [];

  // Draw a frame on the canvas
  const drawFrame = (ctx, canvas) => {
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);
  }, []);

  const startDrawing = (e) => {
    e.preventDefault();
    points = []; // Reset points on new drawing
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
    ctxRef.current = ctx;
    points.push({ x: offsetX, y: offsetY });
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);
    points.push({ x: offsetX, y: offsetY });
    const ctx = ctxRef.current;
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = (e) => {
    e.preventDefault();
    setIsDrawing(false);
    // Process the shape after drawing is finished
    setTimeout(() => processShape(points, simplificationFactor), 100);
  };

  const processShape = (points, tolerance) => {
    if (points.length < 2) return;
    if (isCircle(points)) {
      redrawAsCircle(points);
    } else if (isRectangle(points)) {
      redrawAsRectangle(points);
    } else {
      // For freehand drawing, apply Bezier smoothing using the tolerance
      redrawSmoothPath(points, tolerance);
    }
  };

  // Detects an approximate circle by comparing the bounding box dimensions.
  const isCircle = (points) => {
    const minX = Math.min(...points.map((p) => p.x));
    const maxX = Math.max(...points.map((p) => p.x));
    const minY = Math.min(...points.map((p) => p.y));
    const maxY = Math.max(...points.map((p) => p.y));
    const width = maxX - minX;
    const height = maxY - minY;
    return Math.abs(width - height) < 10; // If width ≈ height, consider it circular
  };

  // Detects an approximate rectangle by checking if most points lie near the bounding box edges.
  const isRectangle = (points) => {
    if (points.length < 4) return false;
    const minX = Math.min(...points.map((p) => p.x));
    const maxX = Math.max(...points.map((p) => p.x));
    const minY = Math.min(...points.map((p) => p.y));
    const maxY = Math.max(...points.map((p) => p.y));
    const threshold = 15;
    let nearEdgesCount = 0;
    points.forEach((p) => {
      const dLeft = Math.abs(p.x - minX);
      const dRight = Math.abs(maxX - p.x);
      const dTop = Math.abs(p.y - minY);
      const dBottom = Math.abs(maxY - p.y);
      if (dLeft < threshold || dRight < threshold || dTop < threshold || dBottom < threshold) {
        nearEdgesCount++;
      }
    });
    return (nearEdgesCount / points.length) > 0.7; // 70% or more points near an edge
  };

  const redrawAsCircle = (points) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    const centerX = (Math.min(...points.map(p => p.x)) + Math.max(...points.map(p => p.x))) / 2;
    const centerY = (Math.min(...points.map(p => p.y)) + Math.max(...points.map(p => p.y))) / 2;
    const radius = Math.max(...points.map(p => Math.hypot(p.x - centerX, p.y - centerY)));
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();
  };

  const redrawAsRectangle = (points) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    const minX = Math.min(...points.map(p => p.x));
    const maxX = Math.max(...points.map(p => p.x));
    const minY = Math.min(...points.map(p => p.y));
    const maxY = Math.max(...points.map(p => p.y));
    ctx.beginPath();
    ctx.rect(minX, minY, maxX - minX, maxY - minY);
    ctx.stroke();
  };

  // Redraw the freehand path using quadratic Bezier curves.
  // The 'tolerance' value is normalized into an alpha parameter that influences smoothing.
  const redrawSmoothPath = (points, tolerance) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
    if (points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    // Normalize tolerance to an alpha value (range roughly between 0.1 and 1)
    const alpha = Math.min(Math.max(tolerance / 20, 0.1), 1);
    for (let i = 1; i < points.length - 1; i++) {
      const p0 = points[i - 1];
      const p1 = points[i];
      const p2 = points[i + 1];
      // Compute midpoint between p1 and p2
      const midX = (p1.x + p2.x) / 2;
      const midY = (p1.y + p2.y) / 2;
      // Blend p2 with the midpoint based on alpha
      const endpointX = p2.x * (1 - alpha) + midX * alpha;
      const endpointY = p2.y * (1 - alpha) + midY * alpha;
      ctx.quadraticCurveTo(p1.x, p1.y, endpointX, endpointY);
    }
    ctx.stroke();
  };

  const getCoordinates = (e) => {
    if (e.nativeEvent instanceof TouchEvent) {
      const rect = canvasRef.current.getBoundingClientRect();
      const touch = e.nativeEvent.touches[0];
      return { offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top };
    } else {
      return { offsetX: e.nativeEvent.offsetX, offsetY: e.nativeEvent.offsetY };
    }
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
  };

  return (
    <div className="flex flex-col items-center p-4 font-space-grotesk">
      <div className="border-4 border-black rounded-lg p-2">
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
      <div className="mt-4 flex gap-4">
        <button onClick={clearCanvas} className="p-2 bg-red-500 text-white rounded">Clear</button>
      </div>
      <div className="mt-4">
        <label className="mr-2">Smoothness Level:</label>
        <input
          type="range"
          min="1"
          max="20"
          value={simplificationFactor}
          onChange={(e) => setSimplificationFactor(Number(e.target.value))}
        />
      </div>
    </div>
  );
}
