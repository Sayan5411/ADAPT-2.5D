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
      traversability: []
    }
  });

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 360,
        },
        audio: false,
      });

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      setCameraStarted(true);

      connectWebSocket();
    } catch (error) {
      console.error("Camera error:", error);
      alert("Could not access camera.");
    }
  }

  function connectWebSocket() {
    const socket = new WebSocket(
      process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000/ws/detection"
    );

    socketRef.current = socket;

    socket.onopen = () => {
      console.log("WebSocket connected");
      setConnected(true);
    };

    socket.onmessage = (event) => {
      const result = JSON.parse(event.data);

      setData(result);
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected");
      setConnected(false);
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  }

  useEffect(() => {
    const interval = setInterval(() => {
      if (
        !cameraStarted ||
        !videoRef.current ||
        !canvasRef.current ||
        !socketRef.current
      ) {
        return;
      }

      if (socketRef.current.readyState !== WebSocket.OPEN) {
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = 640;
      canvas.height = 360;

      const ctx = canvas.getContext("2d");

      ctx.drawImage(
        video,
        0,
        0,
        640,
        360
      );

      const image = canvas.toDataURL(
        "image/jpeg",
        0.7
      );

      socketRef.current.send(image);
    }, 150);

    return () => clearInterval(interval);
  }, [cameraStarted]);

  useEffect(() => () => {
    socketRef.current?.close();
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach((track) => track.stop());
  }, []);

  return (
    <main className="container">

      <header className="header">
        <div>
          <h1>Adaptive 2.5D LiDAR Mapping</h1>
          <p>
            Dynamic Environment Perception System
          </p>
        </div>

        <div className={connected ? "status online" : "status"}>
          {connected ? "● SYSTEM ONLINE" : "● DISCONNECTED"}
        </div>
      </header>


      <section className="camera-section">

        <CameraView
          videoRef={videoRef}
          cameraStarted={cameraStarted}
          objects={data.objects}
          onStart={startCamera}
        />

        <canvas
          ref={canvasRef}
          style={{ display: "none" }}
        />

      </section>


      <Dashboard data={data} />
      <SpatialMap objects={data.map || []} lidar={data.lidar} />
      <ObjectList objects={data.objects} />

    </main>
  );
}