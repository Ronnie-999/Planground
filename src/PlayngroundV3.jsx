import { useRef, useState, useEffect } from "react";

/**
 * Planground – canvas with 3 tools:
 * 1. Wall   – continuous free-hand line (default)
 * 2. Columns – discrete dots placed on click/tap
 * 3. Eraser  – remove parts of drawing
 */
export default function Planground() {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState("wall"); // "wall" | "columns" | "eraser"

  // ===== initialise canvas & metadata =====
  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = window.innerWidth * 0.9;
    canvas.height = window.innerHeight * 0.8;
    const ctx = canvas.getContext("2d");
    drawFrame(ctx, canvas);

    // Set browser-tab title
    document.title = "Planground";

    // Remove any legacy <h1> that still says "Playground"
    document.querySelectorAll("h1").forEach((el) => {
      if (el.textContent.trim() === "Playground") {
        el.remove();
      }
    });
  }, []);

  // draw decorative frame
  const drawFrame = (ctx, canvas) => {
    ctx.lineWidth = 4;
    ctx.strokeStyle = "black";
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  // ===== pointer → coords helper =====
  const getCoordinates = (e) => {
    if (e.nativeEvent instanceof TouchEvent) {
      const rect = canvasRef.current.getBoundingClientRect();
      const touch = e.nativeEvent.touches[0];
      return {
        offsetX: touch.clientX - rect.left,
        offsetY: touch.clientY - rect.top,
      };
    }
    return {
      offsetX: e.nativeEvent.offsetX,
      offsetY: e.nativeEvent.offsetY,
    };
  };

  // ===== drawing handlers =====
  const startDrawing = (e) => {
    e.preventDefault();
    const { offsetX, offsetY } = getCoordinates(e);
    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    switch (tool) {
      case "eraser":
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineWidth = 20; // eraser size
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
        // single dot only
        setIsDrawing(false);
        break;
      default: // "wall"
        ctx.globalCompositeOperation = "source-over";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
        setIsDrawing(true);
        break;
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);
    const ctx = ctxRef.current;

    if (tool === "wall" || tool === "eraser") {
      ctx.lineTo(offsetX, offsetY);
      ctx.stroke();
    }
  };

  const stopDrawing = (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    setIsDrawing(false);

    if (tool === "eraser") {
      const ctx = ctxRef.current;
      ctx.globalCompositeOperation = "source-over";
      drawFrame(ctx, canvasRef.current); // ensure frame intact
    }
  };

  // ===== utilities =====
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawFrame(ctx, canvas);
  };

  // ===== JSX =====
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
      <div className="mt-4 flex gap-4">
        <button
          onClick={() => setTool("wall")}
          className={`p-2 rounded shadow 2xl:rounded-2xl ${
            tool === "wall" ? "bg-blue-600 text-white" : "bg-gray-200"
          }`}
        >
          Wall
        </button>
        <button
          onClick={() => setTool("columns")}
          className={`p-2 rounded shadow 2xl:rounded-2xl ${
            tool === "columns" ? "bg-blue-600 text-white" : "bg-gray-200"
          }`}
        >
          Columns
        </button>
        <button
          onClick={() => setTool("eraser")}
          className={`p-2 rounded shadow 2xl:rounded-2xl ${
            tool === "eraser" ? "bg-blue-600 text-white" : "bg-gray-200"
          }`}
        >
          Eraser
        </button>
        <button
          onClick={clearCanvas}
          className="p-2 bg-red-500 text-white rounded shadow 2xl:rounded-2xl"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
