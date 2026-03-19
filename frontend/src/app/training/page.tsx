"use client";

import { useEffect, useMemo, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { useWebSocket } from "@/hooks/use-websocket";
import { api } from "@/lib/api";
import { TrainingControls } from "@/components/training/training-controls";
import { TrainingLossChart } from "@/components/training/training-loss-chart";
import { TrainingSamples } from "@/components/training/training-samples";
import type { TrainingMetric, TrainingSampleHistoryItem, TrainingStatus, WSMessage } from "@/types";

export default function TrainingPage() {
  const { metrics, samples } = useWebSocket();
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [savedMetrics, setSavedMetrics] = useState<TrainingMetric[]>([]);
  const [savedSamples, setSavedSamples] = useState<TrainingSampleHistoryItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const trainingStatus = await api.getTrainingStatus() as TrainingStatus;
        setStatus(trainingStatus);

        if (trainingStatus.run_id) {
          const [metricRes, sampleRes] = await Promise.all([
            api.getRunMetrics(trainingStatus.run_id) as Promise<{ run_id: number; metrics: TrainingMetric[] }>,
            api.getRunSamples(trainingStatus.run_id) as Promise<{ run_id: number; samples: TrainingSampleHistoryItem[] }>,
          ]);
          setSavedMetrics(metricRes.metrics.map((metric) => ({ ...metric, type: "metric", run_id: trainingStatus.run_id })));
          setSavedSamples(sampleRes.samples);
        }
      } catch {}
      finally { setLoaded(true); }
    }

    load();
  }, []);

  useEffect(() => {
    if (metrics.length === 0) return;
    const last = metrics[metrics.length - 1];
    if (last.type === "metric") {
      setStatus((prev) => prev ? {
        ...prev,
        is_running: true,
        run_status: "running",
        error: null,
        run_id: last.run_id,
        current_step: last.step,
        current_epoch: last.epoch,
        current_loss: last.loss,
        total_steps: last.total_steps,
        learning_rate: last.lr,
      } : prev);
    }
  }, [metrics]);

  const combinedMetrics = useMemo(() => {
    const liveMetrics = metrics.filter((m): m is Extract<WSMessage, { type: "metric" }> => m.type === "metric");
    const byStep = new Map<number, TrainingMetric | Extract<WSMessage, { type: "metric" }>>();
    for (const metric of savedMetrics) byStep.set(metric.step, metric);
    for (const metric of liveMetrics) byStep.set(metric.step, metric);
    return Array.from(byStep.values()).sort((a, b) => a.step - b.step);
  }, [metrics, savedMetrics]);

  const combinedSamples = useMemo(() => {
    const liveSamples = samples.filter((s): s is Extract<WSMessage, { type: "sample" }> => s.type === "sample");
    const historySamples: WSMessage[] = savedSamples.map((sample) => ({
      type: "sample",
      run_id: status?.run_id ?? 0,
      step: sample.step,
      images: sample.images,
    }));
    const byStep = new Map<number, WSMessage>();
    for (const sample of historySamples) byStep.set(sample.step, sample);
    for (const sample of liveSamples) byStep.set(sample.step, sample);
    return Array.from(byStep.values()).sort((a, b) => b.step - a.step);
  }, [samples, savedSamples, status?.run_id]);

  if (!loaded) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-14 rounded-xl" />
        <Skeleton className="h-[300px] rounded-xl" />
        <Skeleton className="h-[200px] rounded-xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Training</h1>
      <TrainingControls status={status} onStatusChange={setStatus} />
      <TrainingLossChart metrics={combinedMetrics as WSMessage[]} />
      <TrainingSamples samples={combinedSamples} />
    </div>
  );
}
