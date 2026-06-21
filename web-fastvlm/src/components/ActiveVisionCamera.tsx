"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";

interface ActiveVisionCameraProps {
  onAutoDetect: (base64Image: string, ocrText?: string) => void;
  onStop?: () => void;
  isProcessing: boolean;
  shouldEnable: boolean;
  cooldownActive: boolean;
  backendOnline: boolean;
}

export default function ActiveVisionCamera({ 
  onAutoDetect, onStop, isProcessing, shouldEnable, cooldownActive, backendOnline
}: ActiveVisionCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<"IDLE" | "MOTION_DETECTED" | "CHECKING" | "DETECTED">("IDLE");
  const [hasMounted, setHasMounted] = useState(false);

  // Internal refs for motion detection
  const stateRef = useRef({
    prevGrayData: null as Uint8ClampedArray | null,
    recentMotions: [] as number[],
    stableFrameCount: 0,
    lastCheckTime: 0,
    isChecking: false,
  });

  /**
   * Capture a high-quality frame for the VLM pipeline.
   * Applies contrast enhancement for better text readability.
   */
  const captureFullFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return null;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0);

    // Apply contrast enhancement for text readability
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    let minLum = 255, maxLum = 0;
    for (let i = 0; i < data.length; i += 4) {
      const lum = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
      if (lum < minLum) minLum = lum;
      if (lum > maxLum) maxLum = lum;
    }
    const range = maxLum - minLum;
    if (range > 30) {
      const scale = 255 / range;
      for (let i = 0; i < data.length; i += 4) {
        data[i]   = Math.min(255, Math.max(0, (data[i]   - minLum) * scale));
        data[i+1] = Math.min(255, Math.max(0, (data[i+1] - minLum) * scale));
        data[i+2] = Math.min(255, Math.max(0, (data[i+2] - minLum) * scale));
      }
      ctx.putImageData(imageData, 0, 0);
    }

    return canvas.toDataURL("image/jpeg", 0.95);
  }, []);

  /**
   * Capture a low-res frame for the quick_check endpoint.
   */
  const captureLowResFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return null;

    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, 320, 240);
    return canvas.toDataURL("image/jpeg", 0.6);
  }, []);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  useEffect(() => {
    if (!shouldEnable) return;

    let isMounted = true;
    let localStream: MediaStream | null = null;
    let animationFrameId: number;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 }
          } 
        });
        
        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        
        localStream = stream;
        const videoElement = videoRef.current;
        
        if (videoElement) {
          videoElement.srcObject = stream;
          videoElement.muted = true;
          videoElement.setAttribute('playsinline', 'true');
          
          try {
            await videoElement.play();
          } catch (playErr) {
            console.error("Auto-play failed:", playErr);
          }
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startCamera();

    // Main processing loop: motion detection + auto-trigger
    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current || !shouldEnable) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      if (!ctx || video.videoWidth === 0) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      // --- Motion Detection ---
      canvas.width = 64;
      canvas.height = 64;
      ctx.drawImage(video, 0, 0, 64, 64);
      const frameData = ctx.getImageData(0, 0, 64, 64).data;
      
      const currentGray = new Uint8ClampedArray(64 * 64);
      for (let i = 0, j = 0; i < frameData.length; i += 4, j++) {
        currentGray[j] = (frameData[i] + frameData[i + 1] + frameData[i + 2]) / 3;
      }

      const { prevGrayData, recentMotions } = stateRef.current;
      let motion = 0;
      if (prevGrayData) {
        let totalDiff = 0;
        for (let i = 0; i < currentGray.length; i++) {
          totalDiff += Math.abs(currentGray[i] - prevGrayData[i]);
        }
        motion = totalDiff / currentGray.length;
      }
      stateRef.current.prevGrayData = currentGray;
      recentMotions.push(motion);
      if (recentMotions.length > 10) recentMotions.shift();
      
      const avgMotion = recentMotions.reduce((a, b) => a + b, 0) / recentMotions.length;

      // --- Tier 2: Edge-density based text likelihood ---
      // Compute gradient magnitude on the 64x64 grayscale thumbnail.
      // Text strokes (even thin pencil) create sharp edges; faces/walls don't.
      let edgeSum = 0;
      const W = 64;
      for (let y = 1; y < 63; y++) {
        for (let x = 1; x < 63; x++) {
          const idx = y * W + x;
          const gx = Math.abs(currentGray[idx + 1] - currentGray[idx - 1]);
          const gy = Math.abs(currentGray[idx + W] - currentGray[idx - W]);
          if (gx + gy > 30) edgeSum++;   // count "edge pixels"
        }
      }
      const edgeRatio = edgeSum / (62 * 62);
      const hasTextLikeContent = edgeRatio > 0.04;  // ~4% of pixels are edges → likely text

      // --- Debug Log (visible in browser DevTools console) ---
      if (stateRef.current.stableFrameCount % 10 === 0) {
        // Log every 10th frame to avoid console spam
        console.log(`🔍 Gatekeeper | motion: ${avgMotion.toFixed(1)} | edges: ${(edgeRatio * 100).toFixed(1)}% | stable: ${stateRef.current.stableFrameCount} | text-like: ${hasTextLikeContent}`);
      }

      // --- Status + Auto-trigger Logic ---
      const now = Date.now();
      const canCheck = !isProcessing && !cooldownActive && backendOnline && !stateRef.current.isChecking;

      if (avgMotion < 7.0) {
        // Frame is stable (relaxed threshold — allows natural hand-shake with paper)
        stateRef.current.stableFrameCount++;

        if (hasTextLikeContent && stateRef.current.stableFrameCount > 8 && canCheck && (now - stateRef.current.lastCheckTime > 3000)) {
          // Stable + edges detected + not already checking → Tier 3: backend check
          stateRef.current.isChecking = true;
          stateRef.current.lastCheckTime = now;
          setStatus("CHECKING");

          const lowResFrame = captureLowResFrame();
          if (lowResFrame) {
            try {
              const res = await fetch("/api/backend/api/quick_check", {
                method: "POST",
                headers: { "Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true" },
                body: JSON.stringify({ image_base64: lowResFrame }),
                signal: AbortSignal.timeout(3000),
              });
              const data = await res.json();
              
              console.log(`🔍 Backend response:`, data);

              if (data.has_question && !isProcessing && !cooldownActive) {
                setStatus("DETECTED");
                console.log(`✅ QUESTION DETECTED — waking VLM!`);
                // Text confirmed! Capture full-res and send to VLM
                const fullFrame = captureFullFrame();
                if (fullFrame) {
                  onAutoDetect(fullFrame, data.ocr_preview);
                }
              } else {
                setStatus("IDLE");
              }
            } catch {
              setStatus("IDLE");
            }
          }
          stateRef.current.isChecking = false;
        } else {
          // Stable but no text edges (or still accumulating stable frames) — stay quiet
          setStatus("IDLE");
        }
      } else {
        // Too much motion — tell user to hold steady
        setStatus("MOTION_DETECTED");
        stateRef.current.stableFrameCount = 0;
      }

      setTimeout(() => {
        animationFrameId = requestAnimationFrame(processFrame);
      }, 100); // ~10fps
    };

    animationFrameId = requestAnimationFrame(processFrame);

    return () => {
      isMounted = false;
      cancelAnimationFrame(animationFrameId);
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [shouldEnable, isProcessing, cooldownActive, backendOnline, captureFullFrame, captureLowResFrame, onAutoDetect]);

  const statusDisplay = {
    IDLE: { text: "Scanning...", dot: "idle" },
    MOTION_DETECTED: { text: "Hold steady...", dot: "motion_detected" },
    CHECKING: { text: "Checking...", dot: "checking" },
    DETECTED: { text: "Question found!", dot: "detected" },
  };

  const current = statusDisplay[status];

  return (
    <div className="camera-container">
      {hasMounted ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="live-video"
          suppressHydrationWarning
        />
      ) : (
        <div className="live-video-placeholder" />
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />
      
      <div className="hud-overlay">
        <div className={`hud-metric ${current.dot}`}>
          <span className={`hud-dot ${current.dot}`}></span>
          {current.text}
        </div>
        {isProcessing && <div className="hud-status status-analyzing">AI IS WORKING...</div>}
        {cooldownActive && !isProcessing && <div className="hud-status status-cooldown">Next scan in a moment...</div>}
      </div>

      {/* Stop button only appears while processing */}
      {isProcessing && shouldEnable && (
        <div className="capture-controls active">
          <button 
            className="shutter-button stop-mode"
            onClick={onStop}
            aria-label="Stop AI"
            id="stop-btn"
          >
            <div className="shutter-inner"></div>
          </button>
        </div>
      )}
    </div>
  );
}
