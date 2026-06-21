"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import ActiveVisionCamera from "@/components/ActiveVisionCamera";

// Use Next.js proxy rewrite to avoid Mixed Content / CORS errors on mobile
const BACKEND_URL = "/api/backend";

type SystemStatus = "IDLE" | "CONNECTING" | "SCANNING" | "THINKING" | "ANSWERING" | "COMPLETE" | "ERROR";

export default function FastVLMCore() {
  const [status, setStatus] = useState<SystemStatus>("CONNECTING");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [backendDevice, setBackendDevice] = useState<string | null>(null);
  const [isCloudMode, setIsCloudMode] = useState(false);
  const [manualPrompt, setManualPrompt] = useState("");
  const [showTextInput, setShowTextInput] = useState(false);
  const [cooldownActive, setCooldownActive] = useState(false);

  const clearTimerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const cooldownTimerRef = useRef<NodeJS.Timeout | null>(null);

  const isProcessing = status === "THINKING" || status === "ANSWERING" || status === "SCANNING";

  // ─── Hardware Check for Recruiters ───
  useEffect(() => {
    const checkHardware = async () => {
      // Allow bypass via ?force=true in the URL
      if (typeof window !== "undefined" && window.location.search.includes('force=true')) return;

      const nav = navigator as any;
      const hasGPU = !!nav.gpu; // Use WebGPU support as a proxy for modern hardware
      const memory = nav.deviceMemory || 8; // If unsupported, assume 8GB
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

      if (!hasGPU || memory < 8 || isMobile) {
        alert("Sorry, your device does not have a dedicated GPU or enough memory to run the live AI inference locally. Redirecting you to a recorded demonstration of the working project...");
        window.location.href = "YOUR_GOOGLE_DRIVE_LINK_HERE";
      }
    };
    
    // Slight delay to ensure UI renders first before alerting
    setTimeout(checkHardware, 500);
  }, []);

  // ─── Backend Health Check ───
  const checkBackendHealth = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, { 
        headers: {
          "Bypass-Tunnel-Reminder": "true"
        },
        signal: AbortSignal.timeout(3000) 
      });
      if (res.ok) {
        const data = await res.json();
        const isLocal = data.model_loaded === true;
        const isCloudFallback = data.mode === "cloud_fallback";
        
        setBackendOnline(isLocal || isCloudFallback);
        setIsCloudMode(isCloudFallback);
        setBackendDevice(
          isCloudFallback 
            ? "Cloud Instance" 
            : (data.hardware?.device_name || data.device || null)
        );
        
        if (isLocal || isCloudFallback) {
          setStatus((prev) => (prev === "CONNECTING" || prev === "ERROR" ? "IDLE" : prev));
        } else {
          setError("Backend running but model not loaded. Check server logs.");
          setStatus("ERROR");
        }
        return isLocal || isCloudFallback;
      } else {
        throw new Error(`Health check failed: ${res.status}`);
      }
    } catch {
      setBackendOnline(false);
      setBackendDevice(null);
      if (!isProcessing) {
        setStatus("ERROR");
        setError("Cannot connect to Vision Backend. Start the server:\n\ncd ml-fastvlm && python server.py");
      }
    }
    return false;
  }, [isProcessing, status]);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 10000);
    return () => clearInterval(interval);
  }, [checkBackendHealth]);

  // ─── Auto-Detect Handler (replaces manual capture) ───
  const handleAutoDetect = useCallback(async (base64Image: string, ocrText?: string) => {
    if (!backendOnline || isProcessing || cooldownActive) return;

    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    setResult(null);
    setError(null);
    setStatus("THINKING");

    const textInput = manualPrompt.trim() || ocrText?.trim() || "";

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true"
        },
        body: JSON.stringify({
          image_base64: base64Image,
          prompt: textInput || "",
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split("\n");
        
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          
          try {
            const payload = JSON.parse(line.slice(6));
            
            if (payload.status === "thinking") {
              setStatus("THINKING");
            } else if (payload.status === "answering") {
              setStatus("ANSWERING");
              fullText = payload.full_text || "";
              setResult(fullText);
            } else if (payload.status === "complete") {
              setStatus("COMPLETE");
              setResult(payload.full_text || fullText);
            } else if (payload.status === "error") {
              throw new Error(payload.message);
            }
          } catch (parseErr: any) {
            if (parseErr.message && !parseErr.message.includes("JSON")) {
              throw parseErr;
            }
          }
        }
      }

      setManualPrompt("");

      // Start cooldown — prevent re-triggering for 5 seconds
      setCooldownActive(true);
      cooldownTimerRef.current = setTimeout(() => {
        setCooldownActive(false);
      }, 5000);

      // Auto-clear result after 2 minutes
      clearTimerRef.current = setTimeout(() => {
        setResult(null);
        setStatus("IDLE");
      }, 120000);

    } catch (err: any) {
      if (err.name === "AbortError") return; 
      
      console.error("Inference error:", err);
      setError(`Inference failed: ${err.message}`);
      setStatus("ERROR");
      
      setTimeout(() => {
        setStatus((prev) => (prev === "ERROR" ? "IDLE" : prev));
        setError(null);
      }, 5000);
    }
  }, [backendOnline, isProcessing, cooldownActive, manualPrompt]);

  const handleStop = () => {
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    if (abortControllerRef.current) abortControllerRef.current.abort();
    
    setStatus("IDLE");
    setResult(null);
    setError(null);
    setCooldownActive(false);
  };

  const statusLabel = {
    IDLE: "Scanning for Questions",
    CONNECTING: "Connecting...",
    SCANNING: "Position your question",
    THINKING: "Analyzing image...",
    ANSWERING: "Generating answer...",
    COMPLETE: "Done",
    ERROR: "Error",
  };

  const statusStep = {
    SCANNING: "1/3",
    THINKING: "2/3",
    ANSWERING: "2/3",
    COMPLETE: "3/3",
  };

  return (
    <main className="main-container">
      {/* Backend Status Indicator */}
      <div className={`backend-status ${backendOnline ? "online" : "offline"}`} id="backend-status">
        <span className="backend-dot"></span>
        <span className="backend-label">
          {backendOnline 
            ? `Engine: ${backendDevice || "Ready"}`
            : "Backend Offline"
          }
        </span>
      </div>

      {/* Cloud Fallback Notice */}
      {isCloudMode && (
        <div className="cloud-notice" id="cloud-notice">
          <p>
            The model failed to build locally on this device, so the system is running 
            the same model and code on a cloud instance. No external API is being called — 
            it is still your FastVLM model, just hosted remotely.
          </p>
        </div>
      )}

      <ActiveVisionCamera
        onAutoDetect={handleAutoDetect}
        onStop={handleStop}
        isProcessing={isProcessing} 
        shouldEnable={true}
        cooldownActive={cooldownActive}
        backendOnline={backendOnline}
      />

      {/* Text Input Toggle Button */}
      <button 
        className="text-input-toggle"
        onClick={() => setShowTextInput(!showTextInput)}
        aria-label="Toggle text input"
        id="text-input-toggle"
      >
        ⌨️
      </button>

      {/* Manual Text Input */}
      {showTextInput && (
        <div className="text-input-panel" id="text-input-panel">
          <input
            type="text"
            className="text-input-field"
            placeholder="Type your question here..."
            value={manualPrompt}
            onChange={(e) => setManualPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && manualPrompt.trim()) {
                setShowTextInput(false);
              }
            }}
            autoFocus
            id="manual-prompt-input"
          />
          <div className="text-input-hint">
            Press Enter to set, then show a paper to auto-trigger. Or just point camera at text.
          </div>
        </div>
      )}

      {/* Instruction Box — shown when idle and ready */}
      {status === "IDLE" && !result && !error && backendOnline && (
        <div className="instruction-box">
          <div className="instruction-icon">🧠</div>
          <h2>Always-On AI</h2>
          <p>
            Hold a handwritten question in front of the camera. It will be detected and answered automatically.
          </p>
          {manualPrompt && (
            <div className="queued-prompt">
              💬 Queued: &ldquo;{manualPrompt}&rdquo;
            </div>
          )}
        </div>
      )}

      {/* Offline Instructions */}
      {!backendOnline && status !== "CONNECTING" && (
        <div className="instruction-box offline-box">
          <div className="instruction-icon">⚡</div>
          <h2>Start Vision Backend</h2>
          <p>Run this command to start the AI engine:</p>
          <code className="command-block">cd ml-fastvlm && python server.py</code>
          <p className="retry-hint" onClick={checkBackendHealth}>Click to retry connection</p>
        </div>
      )}

      {/* Result Bottom Sheet */}
      <div className={`result-drawer ${(result || error || (status !== "IDLE" && status !== "CONNECTING" && status !== "ERROR")) ? "open" : ""}`}>
        <div className="drawer-handle"></div>
        
        <div className="status-banner">
          <span className={`status-dot ${status.toLowerCase()}`}></span>
          <span>
            {statusLabel[status]}
            {statusStep[status as keyof typeof statusStep] && (
              <span className="step-counter"> ({statusStep[status as keyof typeof statusStep]})</span>
            )}
          </span>
        </div>

        {error ? (
          <div className="error-message">⚠️ {error}</div>
        ) : result ? (
          <div className="result-content">
            <pre>{result}</pre>
          </div>
        ) : (
          <div className="placeholder-text">
            {status === "SCANNING" 
              ? "Position your question in frame..." 
              : status === "THINKING"
              ? "Vision engine processing..."
              : "AI is working..."
            }
          </div>
        )}
      </div>
    </main>
  );
}
