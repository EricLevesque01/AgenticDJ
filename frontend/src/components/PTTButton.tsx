/**
 * PTTButton — Push-to-Talk UI Component
 * =======================================
 * Hold-to-talk button with visual state ring per Spec §9.3.
 *
 * References:
 *   - Spec §9.3 (Visual States: idle, recording, processing, responding)
 *   - Spec §13.2 (80px diameter, center-bottom, Space key)
 */

'use client';

import { useCallback, useEffect } from 'react';
import { usePTT, type PTTState } from '@/hooks/usePTT';

interface PTTButtonProps {
  ws: WebSocket | null;
  disabled?: boolean;
  onStateChange?: (state: PTTState) => void;
}

// Spec §9.3: Icons for each state
const STATE_ICONS: Record<PTTState, string> = {
  idle: '🎙️',
  recording: '🔴',
  processing: '⏳',
  responding: '✅',
};

const STATE_LABELS: Record<PTTState, string> = {
  idle: 'Hold to talk',
  recording: 'Listening...',
  processing: 'Processing...',
  responding: 'DJ responding',
};

export default function PTTButton({ ws, disabled, onStateChange }: PTTButtonProps) {
  const { pttState, startRecording, stopRecording, isAvailable, error } = usePTT({
    ws,
    onStateChange,
  });

  // Spec §13.2: Space key shortcut (hold to talk)
  useEffect(() => {
    if (disabled || !isAvailable) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && pttState === 'idle') {
        e.preventDefault();
        startRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && pttState === 'recording') {
        e.preventDefault();
        stopRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [disabled, isAvailable, pttState, startRecording, stopRecording]);

  const handlePointerDown = useCallback(() => {
    if (!disabled && isAvailable && pttState === 'idle') {
      startRecording();
    }
  }, [disabled, isAvailable, pttState, startRecording]);

  const handlePointerUp = useCallback(() => {
    if (pttState === 'recording') {
      stopRecording();
    }
  }, [pttState, stopRecording]);

  // CSS class for current state (Spec §9.3)
  const stateClass = `ptt-button ptt-button--${pttState}`;
  const isDisabled = disabled || !isAvailable || !ws;

  return (
    <div className="ptt-area">
      <button
        id="ptt-button"
        className={`${stateClass}${isDisabled ? ' ptt-button--disabled' : ''}`}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        disabled={isDisabled}
        aria-label={STATE_LABELS[pttState]}
        title={`${STATE_LABELS[pttState]} (Space)`}
      >
        <span style={{ fontSize: '1.8rem', lineHeight: 1 }}>
          {STATE_ICONS[pttState]}
        </span>
      </button>
      {error && (
        <div className="error-banner" style={{ bottom: '100px' }}>
          {error}
        </div>
      )}
    </div>
  );
}
