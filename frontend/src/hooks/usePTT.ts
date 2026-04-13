/**
 * usePTT — Push-to-Talk audio capture hook
 * ==========================================
 * Captures audio from the browser microphone, streams PCM chunks
 * over WebSocket, and manages PTT lifecycle state.
 *
 * References:
 *   - Spec §9 (Push-to-Talk Specification)
 *   - Spec §9.1 (Hold-to-talk, 10s max, 500ms min)
 *   - Spec §9.2 (PCM 16kHz 16-bit mono, ~1s chunks)
 *   - Spec §9.3 (Visual states: idle, recording, processing, responding)
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export type PTTState = 'idle' | 'recording' | 'processing' | 'responding';

export interface UsePTTReturn {
  pttState: PTTState;
  startRecording: () => void;
  stopRecording: () => void;
  isAvailable: boolean;
  error: string | null;
}

interface UsePTTOptions {
  /** WebSocket instance for streaming audio */
  ws: WebSocket | null;
  /** Called when PTT state changes */
  onStateChange?: (state: PTTState) => void;
}

// Spec §9.1: Configuration
const MAX_RECORDING_MS = 10_000;  // Hard cap
const MIN_RECORDING_MS = 500;     // Shorter = ignored
const DEBOUNCE_MS = 500;          // Cooldown between releases

// Spec §9.2: Audio format
const SAMPLE_RATE = 16_000;       // 16kHz
const CHUNK_DURATION_MS = 1_000;  // ~1-second chunks

export function usePTT({ ws, onStateChange }: UsePTTOptions): UsePTTReturn {
  const [pttState, setPttState] = useState<PTTState>('idle');
  const [isAvailable, setIsAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const recordingStartRef = useRef<number>(0);
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastReleaseRef = useRef<number>(0);
  const pttStateRef = useRef<PTTState>('idle');

  // Check mic availability on mount
  useEffect(() => {
    if (typeof navigator !== 'undefined' && navigator.mediaDevices) {
      setIsAvailable(true);
    }
  }, []);

  // Propagate state changes
  useEffect(() => {
    onStateChange?.(pttState);
  }, [pttState, onStateChange]);

  const updateState = useCallback((state: PTTState) => {
    setPttState(state);
    pttStateRef.current = state;
  }, []);

  const startRecording = useCallback(async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return;
    }

    // Spec §9.1: Debounce — 500ms cooldown
    const now = Date.now();
    if (now - lastReleaseRef.current < DEBOUNCE_MS) {
      return;
    }

    if (pttState !== 'idle') return;

    setError(null);

    try {
      // Request mic access — PCM at 16kHz mono (Spec §9.2)
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;

      // Create AudioContext for PCM extraction
      const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);

      // ScriptProcessorNode for PCM chunk extraction
      // Buffer size ~1s at 16kHz = 16384 samples
      const processor = audioCtx.createScriptProcessor(16384, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (pttStateRef.current !== 'recording') return;

        const float32 = e.inputBuffer.getChannelData(0);

        // Convert float32 → int16 PCM (Spec §9.2: 16-bit depth)
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Stream as binary WebSocket frame
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(int16.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      // Send ptt_start to server
      ws.send(JSON.stringify({ type: 'ptt_start' }));

      recordingStartRef.current = Date.now();
      updateState('recording');

      // Spec §9.1: 10-second hard cap — auto-release
      maxTimerRef.current = setTimeout(() => {
        stopRecording();
      }, MAX_RECORDING_MS);

      // Haptic feedback for mobile (Spec §13.2)
      if (navigator.vibrate) {
        navigator.vibrate(50);
      }

    } catch (err) {
      // Spec §9.4: Microphone permission denied
      if (err instanceof DOMException && err.name === 'NotAllowedError') {
        setError('Microphone access required');
      } else {
        setError('Failed to start recording');
      }
      console.error('PTT start failed:', err);
    }
  }, [ws, pttState, updateState]);

  const stopRecording = useCallback(() => {
    if (pttState !== 'recording') return;

    // Clear max-duration timer
    if (maxTimerRef.current) {
      clearTimeout(maxTimerRef.current);
      maxTimerRef.current = null;
    }

    // Spec §9.1: Check minimum duration
    const duration = Date.now() - recordingStartRef.current;
    lastReleaseRef.current = Date.now();

    // Stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    if (duration < MIN_RECORDING_MS) {
      // Spec §9.4: Recording too short → ignore
      updateState('idle');
      return;
    }

    // Send ptt_end to server
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ptt_end' }));
    }

    // Transition to processing (waiting for Whisper result)
    updateState('processing');

    // Processing state will be updated by the WebSocket message handler
    // when the server responds with status/duck_start/error
  }, [ws, pttState, updateState]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (maxTimerRef.current) clearTimeout(maxTimerRef.current);
      if (processorRef.current) processorRef.current.disconnect();
      if (audioContextRef.current) audioContextRef.current.close();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    pttState,
    startRecording,
    stopRecording,
    isAvailable,
    error,
  };
}
