"use client";

import { createContext, useContext, useEffect, useRef, useState, useCallback, type ReactNode } from "react";
import type { WSMessage } from "@/types";

interface WSContextValue {
  connected: boolean;
  backendOnline: boolean;
  lastMessage: WSMessage | null;
  metrics: WSMessage[];
  samples: WSMessage[];
  subscribe: (type: string, cb: (msg: WSMessage) => void) => () => void;
}

const WSContext = createContext<WSContextValue>({
  connected: false,
  backendOnline: false,
  lastMessage: null,
  metrics: [],
  samples: [],
  subscribe: () => () => {},
});

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [metrics, setMetrics] = useState<WSMessage[]>([]);
  const [samples, setSamples] = useState<WSMessage[]>([]);
  const subsRef = useRef<Map<string, Set<(msg: WSMessage) => void>>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);

  // Health check polling -- detects if backend is reachable even without WS
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    async function check() {
      try {
        const res = await fetch("/health", { signal: AbortSignal.timeout(3000) });
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    }
    check();
    timer = setInterval(check, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${protocol}://${window.location.hostname}:8001/ws`);
      wsRef.current = ws;

      ws.onopen = () => { setConnected(true); setBackendOnline(true); };
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        try {
          const msg: WSMessage = JSON.parse(event.data);
          setLastMessage(msg);
          if (msg.type === "metric") setMetrics((prev) => [...prev.slice(-500), msg]);
          if (msg.type === "sample") setSamples((prev) => [...prev.slice(-50), msg]);
          const subs = subsRef.current.get(msg.type);
          if (subs) subs.forEach((cb) => cb(msg));
          const allSubs = subsRef.current.get("*");
          if (allSubs) allSubs.forEach((cb) => cb(msg));
        } catch {}
      };
    }

    connect();
    return () => { clearTimeout(reconnectTimer); ws?.close(); };
  }, []);

  const subscribe = useCallback((type: string, cb: (msg: WSMessage) => void) => {
    if (!subsRef.current.has(type)) subsRef.current.set(type, new Set());
    subsRef.current.get(type)!.add(cb);
    return () => { subsRef.current.get(type)?.delete(cb); };
  }, []);

  return (
    <WSContext.Provider value={{ connected, backendOnline, lastMessage, metrics, samples, subscribe }}>
      {children}
    </WSContext.Provider>
  );
}

export function useWebSocket() {
  return useContext(WSContext);
}
