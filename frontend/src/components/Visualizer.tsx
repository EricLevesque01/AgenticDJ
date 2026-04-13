/**
 * Visualizer — DJ Voice Audio Visualizer
 * =======================================
 * 64-band frequency bar visualizer connected to the DJ audio element
 * via Web Audio API AnalyserNode. Active only during DJ breaks.
 *
 * NOTE: Spotify audio cannot be routed through Web Audio API via the SDK.
 * The visualizer ONLY responds to DJ voice audio (Spec §13.2).
 *
 * References:
 *   - Spec §13.2 (Visualizer — 64 bands, 30fps, hidden when music plays)
 *   - Spec §8.1 (Audio Routing — DJ voice via <audio> element)
 */

'use client';

import { useEffect, useRef, useCallback } from 'react';

interface VisualizerProps {
  /** Whether the DJ is currently speaking (controls visibility) */
  isDJActive: boolean;
  /** ID of the <audio> element to attach to */
  audioElementId?: string;
}

// Spec §13.2: 64 bands, 30fps
const NUM_BANDS = 64;
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

export default function Visualizer({
  isDJActive,
  audioElementId = 'dj-audio',
}: VisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);

  // Connect AnalyserNode to the DJ <audio> element
  const connectAnalyser = useCallback(() => {
    const audioEl = document.getElementById(audioElementId) as HTMLAudioElement | null;
    if (!audioEl || audioCtxRef.current) return;

    try {
      const ctx = new AudioContext();
      const analyser = ctx.createAnalyser();

      // Spec §13.2: 64 bands → fftSize must be 2× bands
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.8;

      const source = ctx.createMediaElementSource(audioEl);
      source.connect(analyser);
      analyser.connect(ctx.destination);

      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
      sourceRef.current = source;
    } catch (err) {
      console.warn('[Visualizer] Web Audio API unavailable:', err);
    }
  }, [audioElementId]);

  // Draw loop at 30fps
  const draw = useCallback((timestamp: number) => {
    if (timestamp - lastFrameTimeRef.current < FRAME_INTERVAL) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }
    lastFrameTimeRef.current = timestamp;

    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) {
      animFrameRef.current = requestAnimationFrame(draw);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);

    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const barWidth = (width / NUM_BANDS) - 1;
    const halfH = height / 2;

    for (let i = 0; i < NUM_BANDS; i++) {
      const value = data[i] / 255; // 0–1
      const barH = Math.max(2, value * height * 0.9);

      // Gradient: indigo at base → purple at top
      const gradient = ctx.createLinearGradient(0, halfH, 0, halfH - barH);
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.9)');
      gradient.addColorStop(1, 'rgba(167, 139, 250, 0.6)');

      ctx.fillStyle = gradient;

      const x = i * (barWidth + 1);
      // Draw from center outward (mirror top+bottom)
      ctx.fillRect(x, halfH - barH / 2, barWidth, barH);
    }

    animFrameRef.current = requestAnimationFrame(draw);
  }, []);

  // Start/stop animation when DJ becomes active
  useEffect(() => {
    if (isDJActive) {
      connectAnalyser();

      // Resume AudioContext if suspended (browser policy)
      if (audioCtxRef.current?.state === 'suspended') {
        audioCtxRef.current.resume();
      }

      animFrameRef.current = requestAnimationFrame(draw);
    } else {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }

      // Clear canvas when inactive
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    return () => {
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isDJActive, connectAnalyser, draw]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animFrameRef.current !== null) cancelAnimationFrame(animFrameRef.current);
      audioCtxRef.current?.close();
    };
  }, []);

  return (
    <div
      className="visualizer"
      id="visualizer"
      style={{
        opacity: isDJActive ? 1 : 0,
        transition: 'opacity 400ms ease',
      }}
    >
      <canvas
        ref={canvasRef}
        width={400}
        height={60}
        style={{ width: '100%', height: '100%', display: 'block' }}
        aria-hidden="true"
      />
    </div>
  );
}
