"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import CameraView from "../components/CameraView";
import Dashboard from "../components/Dashboard";
import ObjectList from "../components/ObjectList";
import SpatialMap from "../components/SpatialMap";

const EMPTY_DATA = {
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
  image: {
    width: 640,
    height: 360,
  },
};

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);

  // Prevent multiple WebSocket connections.
  const connectingRef = useRef(false);

  // Prevent sending a new frame while Render is still processing
  // the previous frame.
  const processingFrameRef = useRef(false);

  const intervalRef = useRef(null);

  const [cameraStarted, setCameraStarted] = useState(false);
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(EMPTY_DATA);

  // ------------------------------------------------------------
  // WEBSOCKET URL
  // ------------------------------------------------------------

  const getWebSocketUrl = useCallback(() => {
    const configuredUrl = process.env.NEXT_PUBLIC_WS_URL?.trim();

    if (configuredUrl) {
      return configuredUrl;
    }

    // Local development fallback.
    return "ws://127.0.0.1:8000/ws/detection";
  }, []);

  // ------------------------------------------------------------
  // CLOSE WEBSOCKET
  // ------------------------------------------------------------

  const closeWebSocket = useCallback(() => {
    const socket = socketRef.current;

    if (socket) {
      try {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      } catch (error) {
        console.warn("WebSocket close error:", error);
      }
    }

    socketRef.current = null;
    connectingRef.current = false;
    processingFrameRef.current = false;

    setConnected(false);
  }, []);

  // ------------------------------------------------------------
  // CONNECT WEBSOCKET
  // ------------------------------------------------------------

  const connectWebSocket = useCallback(() => {
    // Already connected.
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Connection is currently being created.
    if (connectingRef.current) {
      return;
    }

    connectingRef.current = true;

    const wsUrl = getWebSocketUrl();

    console.log("========================================");
    console.log("Connecting to detection backend...");
    console.log("WebSocket URL:", wsUrl);
    console.log("========================================");

    let socket;

    try {
      socket = new WebSocket(wsUrl);
    } catch (error) {
      console.error("Failed to create WebSocket:", error);

      connectingRef.current = false;
      setConnected(false);

      return;
    }

    socketRef.current = socket;

    // ----------------------------------------------------------
    // OPEN
    // ----------------------------------------------------------

    socket.onopen = () => {
      console.log("========================================");
      console.log("WebSocket connected successfully");
      console.log("Detection pipeline is ready");
      console.log("========================================");

      connectingRef.current = false;
      setConnected(true);
    };

    // ----------------------------------------------------------
    // MESSAGE
    // ----------------------------------------------------------

    socket.onmessage = (event) => {
      try {
        const result = JSON.parse(event.data);

        console.log("Detection response:", result);

        // Backend error.
        if (result.error) {
          console.error("Backend error:", result.error);

          processingFrameRef.current = false;
          return;
        }

        // Update dashboard data.
        setData({
          ...EMPTY_DATA,
          ...result,

          objects: Array.isArray(result.objects)
            ? result.objects
            : [],

          counts: {
            ...EMPTY_DATA.counts,
            ...(result.counts || {}),
          },

          map: Array.isArray(result.map)
            ? result.map
            : [],

          lidar: {
            ...EMPTY_DATA.lidar,
            ...(result.lidar || {}),
          },
        });

        // VERY IMPORTANT:
        // Allow the next frame only after the previous
        // detection response has arrived.
        processingFrameRef.current = false;
      } catch (error) {
        console.error(
          "Could not parse backend response:",
          error
        );

        processingFrameRef.current = false;
      }
    };

    // ----------------------------------------------------------
    // ERROR
    // ----------------------------------------------------------

    socket.onerror = (error) => {
      console.error("========================================");
      console.error("WebSocket ERROR");
      console.error(error);
      console.error("========================================");

      setConnected(false);
      processingFrameRef.current = false;
    };

    // ----------------------------------------------------------
    // CLOSE
    // ----------------------------------------------------------

    socket.onclose = (event) => {
      console.warn(
        "WebSocket disconnected.",
        "Code:",
        event.code,
        "Reason:",
        event.reason
      );

      connectingRef.current = false;
      processingFrameRef.current = false;

      setConnected(false);

      socketRef.current = null;
    };
  }, [getWebSocketUrl]);

  // ------------------------------------------------------------
  // SEND ONE FRAME
  // ------------------------------------------------------------

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const socket = socketRef.current;

    if (!video || !canvas) {
      return;
    }

    if (!socket) {
      return;
    }

    if (socket.readyState !== WebSocket.OPEN) {
      return;
    }

    if (!cameraStarted) {
      return;
    }

    // Don't send another frame until the backend has
    // finished processing the previous one.
    if (processingFrameRef.current) {
      return;
    }

    // Video must actually contain data.
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth <= 0 ||
      video.videoHeight <= 0
    ) {
      return;
    }

    try {
      const width = 640;
      const height = 360;

      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d", {
        alpha: false,
      });

      if (!context) {
        console.error("Could not get canvas context.");
        return;
      }

      // Draw current camera frame.
      context.drawImage(
        video,
        0,
        0,
        width,
        height
      );

      // Convert frame to JPEG.
      const imageData = canvas.toDataURL(
        "image/jpeg",
        0.65
      );

      if (!imageData || imageData.length < 100) {
        console.warn("Captured image appears to be empty.");
        return;
      }

      // Mark frame as being processed.
      processingFrameRef.current = true;

      console.log(
        "Sending frame:",
        Math.round(imageData.length / 1024),
        "KB"
      );

      socket.send(imageData);
    } catch (error) {
      console.error("Frame capture/send error:", error);

      processingFrameRef.current = false;
    }
  }, [cameraStarted]);

  // ------------------------------------------------------------
  // START CAMERA
  // ------------------------------------------------------------

  const startCamera = useCallback(async () => {
    try {
      console.log("Requesting browser camera...");

      // If an old camera stream exists, stop it first.
      const oldStream = videoRef.current?.srcObject;

      if (oldStream) {
        oldStream
          .getTracks()
          .forEach((track) => track.stop());

        videoRef.current.srcObject = null;
      }

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

      if (!videoRef.current) {
        stream
          .getTracks()
          .forEach((track) => track.stop());

        return;
      }

      videoRef.current.srcObject = stream;

      videoRef.current.muted = true;
      videoRef.current.playsInline = true;

      await videoRef.current.play();

      console.log("========================================");
      console.log("Camera started");
      console.log(
        "Camera resolution:",
        videoRef.current.videoWidth,
        "x",
        videoRef.current.videoHeight
      );
      console.log("========================================");

      setCameraStarted(true);

      // Connect AFTER camera is ready.
      connectWebSocket();
    } catch (error) {
      console.error("Camera error:", error);

      setCameraStarted(false);

      alert(
        "Could not access camera.\n\n" +
          "Please allow camera permission in your browser."
      );
    }
  }, [connectWebSocket]);

  // ------------------------------------------------------------
  // START FRAME LOOP
  // ------------------------------------------------------------

  useEffect(() => {
    if (!cameraStarted) {
      return;
    }

    console.log("Starting detection frame loop...");

    // 250 ms = maximum ~4 frames/sec sent.
    //
    // Because sendFrame() waits for the previous response,
    // Render CPU will not get flooded with frames.
    intervalRef.current = setInterval(() => {
      sendFrame();
    }, 250);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      processingFrameRef.current = false;

      console.log("Detection frame loop stopped.");
    };
  }, [cameraStarted, sendFrame]);

  // ------------------------------------------------------------
  // CLEANUP
  // ------------------------------------------------------------

  useEffect(() => {
    return () => {
      console.log("Cleaning up application...");

      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      closeWebSocket();

      const stream =
        videoRef.current?.srcObject;

      if (stream) {
        stream
          .getTracks()
          .forEach((track) => track.stop());
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [closeWebSocket]);

  // ------------------------------------------------------------
  // UI
  // ------------------------------------------------------------

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
          width={640}
          height={360}
          style={{
            display: "none",
          }}
        />

      </section>

      {/* ======================================================
          DASHBOARD
      ====================================================== */}

      <Dashboard data={data} />

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