```javascript
"use client";

import { useEffect, useRef, useState } from "react";
import CameraView from "../components/CameraView";
import Dashboard from "../components/Dashboard";
import ObjectList from "../components/ObjectList";
import SpatialMap from "../components/SpatialMap";

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);

  // Prevent multiple WebSockets
  const connectingRef = useRef(false);

  // Prevent multiple frames from being processed at once
  const processingRef = useRef(false);

  const [cameraStarted, setCameraStarted] = useState(false);
  const [connected, setConnected] = useState(false);

  const [data, setData] = useState({
    objects: [],
    counts: {
      danger: 0,
      warning: 0,
      safe: 0,
      total: 0,
    },
    fps: 0,
    inference_size: 640,
    map: [],
    lidar: {
      connected: false,
      points: [],
      point_count: 0,
      grid: [],
      elevation: [],
      traversability: [],
    },
  });

  // ============================================================
  // WEBSOCKET URL
  // ============================================================

  const getWebSocketURL = () => {
    const envURL = process.env.NEXT_PUBLIC_WS_URL;

    if (envURL) {
      return envURL;
    }

    // Local development fallback
    return "ws://127.0.0.1:8000/ws/detection";
  };

  // ============================================================
  // CONNECT WEBSOCKET
  // ============================================================

  function connectWebSocket() {
    // Already connected / connecting
    if (
      socketRef.current &&
      (
        socketRef.current.readyState === WebSocket.OPEN ||
        socketRef.current.readyState === WebSocket.CONNECTING
      )
    ) {
      console.log("WebSocket already connected/connecting");
      return;
    }

    // Extra protection
    if (connectingRef.current) {
      return;
    }

    connectingRef.current = true;

    const wsURL = getWebSocketURL();

    console.log("Connecting WebSocket:");
    console.log(wsURL);

    const socket = new WebSocket(wsURL);

    socketRef.current = socket;

    socket.onopen = () => {
      console.log("=================================");
      console.log("WebSocket connected");
      console.log("=================================");

      connectingRef.current = false;
      processingRef.current = false;

      setConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const result = JSON.parse(event.data);

        console.log("Detection result:", result);

        // Keep lidar state because backend does not currently send it
        setData((previous) => ({
          ...previous,

          ...result,

          lidar: result.lidar ?? previous.lidar,
        }));

        // Previous frame finished
        processingRef.current = false;
      } catch (error) {
        console.error("Invalid WebSocket response:", error);

        processingRef.current = false;
      }
    };

    socket.onclose = (event) => {
      console.log(
        "WebSocket disconnected:",
        event.code,
        event.reason
      );

      connectingRef.current = false;
      processingRef.current = false;

      setConnected(false);

      if (socketRef.current === socket) {
        socketRef.current = null;
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);

      connectingRef.current = false;
      processingRef.current = false;
    };
  }

  // ============================================================
  // START CAMERA
  // ============================================================

  async function startCamera() {
    try {
      // Prevent starting twice
      if (cameraStarted) {
        console.log("Camera already started");
        return;
      }

      console.log("Requesting camera...");

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: {
              ideal: 640,
            },
            height: {
              ideal: 360,
            },
            facingMode: "environment",
          },
          audio: false,
        });

      streamRef.current = stream;

      if (!videoRef.current) {
        console.error("Video element not available");

        stream.getTracks().forEach((track) => {
          track.stop();
        });

        return;
      }

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      console.log("Camera started");

      setCameraStarted(true);

      // Connect exactly one WebSocket
      connectWebSocket();

    } catch (error) {
      console.error("Camera error:", error);

      alert(
        "Could not access camera.\n\n" +
        "Please allow camera permission in your browser."
      );
    }
  }

  // ============================================================
  // SEND ONE FRAME
  // ============================================================

  function sendFrame() {
    const socket = socketRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Basic checks
    if (!cameraStarted) {
      return;
    }

    if (!video || !canvas) {
      return;
    }

    if (!socket) {
      return;
    }

    if (socket.readyState !== WebSocket.OPEN) {
      return;
    }

    // IMPORTANT:
    // Don't send another frame while YOLO is
    // still processing the previous frame.
    if (processingRef.current) {
      return;
    }

    // Make sure video actually contains a frame
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    const ctx = canvas.getContext("2d");

    if (!ctx) {
      console.error("Could not get canvas context");
      return;
    }

    canvas.width = 640;
    canvas.height = 360;

    // Draw current camera frame
    ctx.drawImage(
      video,
      0,
      0,
      640,
      360
    );

    // Convert to JPEG
    const image = canvas.toDataURL(
      "image/jpeg",
      0.65
    );

    try {
      // Mark as processing BEFORE sending
      processingRef.current = true;

      socket.send(image);

    } catch (error) {
      console.error(
        "Failed to send frame:",
        error
      );

      processingRef.current = false;
    }
  }

  // ============================================================
  // FRAME LOOP
  // ============================================================

  useEffect(() => {
    if (!cameraStarted) {
      return;
    }

    console.log("Starting frame transmission...");

    // Approximately 10 attempts/sec.
    // Actual transmission is limited by processingRef.
    const interval = setInterval(() => {
      sendFrame();
    }, 100);

    return () => {
      console.log("Stopping frame transmission...");
      clearInterval(interval);
    };

  }, [cameraStarted]);

  // ============================================================
  // CLEANUP
  // ============================================================

  useEffect(() => {
    return () => {
      console.log("Cleaning up Home component...");

      // Close WebSocket
      if (socketRef.current) {
        try {
          socketRef.current.close();
        } catch (error) {
          console.error(
            "WebSocket cleanup error:",
            error
          );
        }

        socketRef.current = null;
      }

      // Stop camera
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });

        streamRef.current = null;
      }

      // Also stop stream attached to video
      if (videoRef.current) {
        const stream =
          videoRef.current.srcObject;

        if (stream) {
          stream
            .getTracks()
            .forEach((track) => {
              track.stop();
            });
        }

        videoRef.current.srcObject = null;
      }

      processingRef.current = false;
      connectingRef.current = false;
    };
  }, []);

  // ============================================================
  // UI
  // ============================================================

  return (
    <main className="container">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">

        <div>
          <h1>
            Adaptive 2.5D LiDAR Mapping
          </h1>

          <p>
            Dynamic Environment Perception System
          </p>
        </div>

        <div
          className={
            connected
              ? "status online"
              : "status"
          }
        >
          {connected
            ? "● SYSTEM ONLINE"
            : "● DISCONNECTED"}
        </div>

      </header>


      {/* ======================================================
          CAMERA
      ====================================================== */}

      <section className="camera-section">

        <CameraView
          videoRef={videoRef}
          cameraStarted={cameraStarted}
          objects={data.objects}
          onStart={startCamera}
        />

        {/* Hidden canvas used to capture frames */}

        <canvas
          ref={canvasRef}
          style={{
            display: "none",
          }}
        />

      </section>


      {/* ======================================================
          DASHBOARD
      ====================================================== */}

      <Dashboard
        data={data}
      />


      {/* ======================================================
          2.5D MAP
      ====================================================== */}

      <SpatialMap
        objects={data.map || []}
        lidar={data.lidar}
      />


      {/* ======================================================
          OBJECT LIST
      ====================================================== */}

      <ObjectList
        objects={data.objects || []}
      />

    </main>
  );
}
```
