/**
 * StatusHUD — Agent Pipeline Status Display
 * ==========================================
 * Shows the current node + message during agent work.
 * Fades out after 2s of inactivity per Spec §13.2.
 *
 * References:
 *   - Spec §13.2 (Status HUD)
 *   - Spec §7.3 ({ type: "status", node: string, message: string })
 */

'use client';

import { useEffect, useRef, useState } from 'react';

export interface StatusMessage {
  node: string;
  message: string;
}

interface StatusHUDProps {
  status: StatusMessage | null;
  connectionState: 'connecting' | 'connected' | 'reconnecting' | 'disconnected';
}

const NODE_LABELS: Record<string, string> = {
  observer:       'Observer',
  historian:      'Historian',
  discoverer:     'Discoverer',
  curator:        'Curator',
  scriptwriter:   'Scriptwriter',
  vocalizer:      'Vocalizer',
  broadcast:      'Broadcast',
  memory_manager: 'Memory',
};

const NODE_COLORS: Record<string, string> = {
  observer:       '#4a90d9',
  historian:      '#e67e22',
  discoverer:     '#27ae60',
  curator:        '#8e44ad',
  scriptwriter:   '#f39c12',
  vocalizer:      '#1abc9c',
  broadcast:      '#e74c3c',
  memory_manager: '#7f8c8d',
};

export default function StatusHUD({ status, connectionState }: StatusHUDProps) {
  const [visible, setVisible] = useState(false);
  const [displayedStatus, setDisplayedStatus] = useState<StatusMessage | null>(null);
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Spec §13.2: Show on new status, fade out after 2s of inactivity
  useEffect(() => {
    if (!status) return;

    setDisplayedStatus(status);
    setVisible(true);

    if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
    fadeTimerRef.current = setTimeout(() => {
      setVisible(false);
    }, 2000);

    return () => {
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
    };
  }, [status]);

  const nodeColor = displayedStatus
    ? NODE_COLORS[displayedStatus.node] ?? '#6366f1'
    : '#6366f1';

  const nodeLabel = displayedStatus
    ? NODE_LABELS[displayedStatus.node] ?? displayedStatus.node
    : '';

  return (
    <div className="status-hud" id="status-hud">
      <div className="status-hud__content">
        {/* Connection dot */}
        <span
          className={`status-hud__dot ${
            connectionState === 'connected'
              ? 'status-hud__dot--connected'
              : 'status-hud__dot--disconnected'
          }`}
        />

        {/* Agent status — fades in/out */}
        <span
          className="status-hud__text"
          style={{
            opacity: visible ? 1 : 0,
            transition: 'opacity 400ms ease',
          }}
        >
          {displayedStatus && (
            <>
              <span
                style={{
                  color: nodeColor,
                  fontWeight: 600,
                  marginRight: 6,
                }}
              >
                {nodeLabel}
              </span>
              <span>{displayedStatus.message}</span>
            </>
          )}
          {!displayedStatus && connectionState}
        </span>
      </div>
    </div>
  );
}
