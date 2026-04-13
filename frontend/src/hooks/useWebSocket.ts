/**
 * useWebSocket — WebSocket connection manager
 *
 * Manages the persistent WebSocket connection to the FastAPI backend.
 * Handles reconnection with exponential backoff, heartbeat, and
 * message type routing.
 *
 * References:
 *   - Spec §7.1 (Transport — ws://127.0.0.1:8000/ws)
 *   - Spec §7.2 (Connection Lifecycle — heartbeat, reconnect)
 *   - Spec §7.3 (Message Types)
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ClientMessage, ConnectionState, ServerMessage } from '@/lib/types';

const WS_URL = 'ws://127.0.0.1:8000/ws';
// Spec §7.2: Heartbeat every 30s
const HEARTBEAT_INTERVAL_MS = 30_000;
// Spec §7.2: Exponential backoff — 1s, 2s, 4s, max 30s
const INITIAL_RECONNECT_DELAY_MS = 1_000;
const MAX_RECONNECT_DELAY_MS = 30_000;

export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  sessionId: string | null;
  sendMessage: (message: ClientMessage) => void;
  sendBinary: (data: ArrayBuffer) => void;
  lastMessage: ServerMessage | null;
  onMessage: (handler: (msg: ServerMessage) => void) => void;
  onBinary: (handler: (data: ArrayBuffer) => void) => void;
  socket: WebSocket | null;
}

export function useWebSocket(accessToken: string | null): UseWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const messageHandlerRef = useRef<((msg: ServerMessage) => void) | null>(null);
  const binaryHandlerRef = useRef<((data: ArrayBuffer) => void) | null>(null);
  // Use a ref for the token so the effect doesn't re-run when token changes
  const accessTokenRef = useRef(accessToken);
  // Track whether this hook is still mounted (prevents reconnect after unmount)
  const mountedRef = useRef(true);

  // Keep the ref in sync
  useEffect(() => {
    accessTokenRef.current = accessToken;
  }, [accessToken]);

  // Single stable connection effect — only runs on mount/unmount
  useEffect(() => {
    mountedRef.current = true;
    let ws: WebSocket | null = null;

    function cleanupHeartbeat() {
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
        heartbeatRef.current = null;
      }
    }

    function clearReconnectTimer() {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    }

    function doConnect() {
      const token = accessTokenRef.current;
      if (!token || !mountedRef.current) return;

      // Close any existing connection cleanly
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect from cleanup
        wsRef.current.close();
        wsRef.current = null;
      }
      cleanupHeartbeat();

      setConnectionState('connecting');

      ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(token)}`);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        console.log('[WS] Connected');
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS;

        // Start heartbeat (Spec §7.2: ping every 30s)
        heartbeatRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        }, HEARTBEAT_INTERVAL_MS);
      };

      ws.onmessage = (event: MessageEvent) => {
        if (event.data instanceof ArrayBuffer) {
          binaryHandlerRef.current?.(event.data);
          return;
        }

        try {
          const msg: ServerMessage = JSON.parse(event.data as string);
          setLastMessage(msg);

          if (msg.type === 'connected') {
            setSessionId(msg.session_id);
            setConnectionState('connected');
            console.log('[WS] Session:', msg.session_id);
          }

          messageHandlerRef.current?.(msg);
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      };

      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        console.log('[WS] Disconnected:', event.code, event.reason);
        setConnectionState('reconnecting');
        cleanupHeartbeat();

        // Spec §7.2: Don't reconnect on auth failure
        if (event.code === 4001) {
          setConnectionState('disconnected');
          return;
        }

        // Exponential backoff reconnect
        const delay = reconnectDelayRef.current;
        console.log(`[WS] Reconnecting in ${delay}ms`);
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS);
        reconnectTimerRef.current = setTimeout(doConnect, delay);
      };

      ws.onerror = (error) => {
        console.error('[WS] Error:', error);
      };
    }

    // Only connect if we have a token
    if (accessToken) {
      doConnect();
    }

    // Cleanup on unmount only
    return () => {
      mountedRef.current = false;
      clearReconnectTimer();
      cleanupHeartbeat();
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect from cleanup
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const sendMessage = useCallback((message: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WS] Cannot send — not connected');
    }
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    } else {
      console.warn('[WS] Cannot send binary — not connected');
    }
  }, []);

  const onMessage = useCallback((handler: (msg: ServerMessage) => void) => {
    messageHandlerRef.current = handler;
  }, []);

  const onBinary = useCallback((handler: (data: ArrayBuffer) => void) => {
    binaryHandlerRef.current = handler;
  }, []);

  return {
    connectionState,
    sessionId,
    sendMessage,
    sendBinary,
    lastMessage,
    onMessage,
    onBinary,
    socket: wsRef.current,
  };
}
