"use client";

import { useState, useEffect, useCallback, useRef } from "react";

/**
 * Hook for browser-native speech-to-text via the Web Speech API.
 * 
 * Fixes from the original:
 * 1. Accumulates FINAL transcripts (not just interim) to avoid losing words
 * 2. Exposes `isSupported` so the UI can hide the mic button when unavailable
 * 3. Uses a ref for the accumulated transcript so it's always current
 * 4. Auto-restarts on non-fatal errors (e.g. no-speech timeout)
 * 
 * NOTE: Chrome's Web Speech API sends audio to Google servers (requires internet).
 * This is the only non-local component. A text input fallback is provided in the UI.
 */
export function useSpeechToText() {
  const [transcript, setTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const accumulatedRef = useRef("");
  const shouldRestartRef = useRef(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.warn("Speech Recognition API not available in this browser.");
      setIsSupported(false);
      return;
    }
    
    setIsSupported(true);
    
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      let finalTranscript = accumulatedRef.current;
      let interimTranscript = "";
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          // Accumulate final results — these won't change
          finalTranscript += result[0].transcript + " ";
          accumulatedRef.current = finalTranscript;
        } else {
          // Show interim results for live feedback
          interimTranscript += result[0].transcript;
        }
      }
      
      // Display: accumulated finals + current interim
      const displayText = (finalTranscript + interimTranscript).trim();
      setTranscript(displayText);
    };

    recognition.onerror = (event: any) => {
      if (event.error === "network") {
        console.warn("Speech recognition requires internet (Chrome sends audio to Google).");
        setIsListening(false);
        shouldRestartRef.current = false;
      } else if (event.error === "no-speech") {
        // Silence timeout — restart if we should still be listening
        if (shouldRestartRef.current) {
          try { recognition.start(); } catch (_) {}
        }
      } else if (event.error === "aborted") {
        // Normal stop, do nothing
      } else {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      // Auto-restart if the recognition ended but we still want to listen
      // (the Web Speech API often stops after ~60s of continuous listening)
      if (shouldRestartRef.current) {
        try { recognition.start(); } catch (_) {}
      } else {
        setIsListening(false);
      }
    };

    recognitionRef.current = recognition;
    
    return () => {
      shouldRestartRef.current = false;
      try { recognition.stop(); } catch (_) {}
    };
  }, []);

  const startListening = useCallback(() => {
    if (!recognitionRef.current) return;
    
    // Reset accumulated transcript
    accumulatedRef.current = "";
    setTranscript("");
    setIsListening(true);
    shouldRestartRef.current = true;
    
    try {
      recognitionRef.current.start();
    } catch (e) {
      // May throw if already started
      console.warn("Speech start error:", e);
    }
  }, []);

  const stopListening = useCallback(() => {
    shouldRestartRef.current = false;
    
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (_) {}
    }
    setIsListening(false);
  }, []);

  /** Get the current accumulated transcript (ref-based, always fresh). */
  const getTranscript = useCallback(() => {
    return accumulatedRef.current.trim() || transcript.trim();
  }, [transcript]);

  return { transcript, isListening, isSupported, startListening, stopListening, getTranscript };
}
