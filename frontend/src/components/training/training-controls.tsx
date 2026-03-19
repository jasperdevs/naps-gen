"use client";

import { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { Play, Square, Loader2 } from "lucide-react";
import type { TrainingStatus } from "@/types";

interface Props {
  status: TrainingStatus | null;
  onStatusChange: (s: TrainingStatus) => void;
}

export function TrainingControls({ status, onStatusChange }: Props) {
  const [starting, setStarting] = useState(false);
  const isRunning = status?.is_running ?? false;
  const runStatus = status?.run_status ?? (isRunning ? "running" : "idle");
  const progress = status?.total_steps ? (status.current_step / status.total_steps) * 100 : 0;

  const eta = useMemo(() => {
    if (!status || !isRunning || status.current_step <= 0 || status.total_steps <= 0 || status.elapsed_seconds <= 0) {
      return null;
    }
    const stepsRemaining = status.total_steps - status.current_step;
    if (stepsRemaining <= 0) return "<1m";
    const secondsPerStep = status.elapsed_seconds / status.current_step;
    const etaSeconds = Math.max(0, Math.round(stepsRemaining * secondsPerStep));
    const hours = Math.floor(etaSeconds / 3600);
    const minutes = Math.floor((etaSeconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m`;
    return `<1m`;
  }, [isRunning, status]);

  async function handleStart() {
    setStarting(true);
    try {
      await api.startTraining({});
      const s = await api.getTrainingStatus();
      onStatusChange(s as TrainingStatus);
    } catch (e) { console.error(e); }
    finally { setStarting(false); }
  }

  async function handleStop() {
    await api.stopTraining();
    const s = await api.getTrainingStatus();
    onStatusChange(s as TrainingStatus);
  }

  return (
    <Card size="sm">
      <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <Badge variant={isRunning ? "default" : runStatus === "failed" ? "destructive" : "secondary"}>
            {isRunning ? "Running" : runStatus === "failed" ? "Failed" : runStatus === "completed" ? "Completed" : runStatus === "stopped" ? "Stopped" : "Idle"}
          </Badge>
          {status && (
            <div className="flex gap-4 text-xs text-muted-foreground">
              {status.total_steps > 0 ? <span>Step {status.current_step?.toLocaleString()}/{status.total_steps?.toLocaleString()}</span> : null}
              {status.total_epochs > 0 ? <span>Epoch {status.current_epoch}/{status.total_epochs}</span> : null}
              {status.current_loss != null ? <span>Loss {status.current_loss?.toFixed(4)}</span> : null}
              {eta ? <span>ETA {eta}</span> : null}
              {!isRunning && status.error ? <span className="text-destructive/90">{status.error}</span> : null}
            </div>
          )}
        </div>
        {isRunning ? (
          <Button onClick={handleStop} variant="destructive" size="sm" className="shrink-0">
            <Square className="mr-1.5 h-3.5 w-3.5" /> Stop
          </Button>
        ) : (
          <Button onClick={handleStart} disabled={starting} size="sm" className="shrink-0">
            {starting ? <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" /> : <Play className="mr-1.5 h-3.5 w-3.5" />}
            {starting ? "Starting" : "Start Training"}
          </Button>
        )}
      </CardContent>
      {isRunning && <div className="px-3 pb-3"><Progress value={progress} /></div>}
    </Card>
  );
}
