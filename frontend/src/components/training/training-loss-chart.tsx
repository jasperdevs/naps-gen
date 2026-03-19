"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Line, LineChart, XAxis, YAxis, CartesianGrid } from "recharts";
import type { WSMessage } from "@/types";

const chartConfig = {
  loss: { label: "Loss", color: "var(--chart-3)" },
  smoothed: { label: "Smoothed", color: "var(--chart-1)" },
} satisfies ChartConfig;

export function TrainingLossChart({ metrics }: { metrics: WSMessage[] }) {
  const chartData = useMemo(() => {
    const data = metrics
      .filter((m): m is Extract<WSMessage, { type: "metric" }> => m.type === "metric")
      .map((m) => ({ step: m.step, loss: m.loss }));

    if (data.length === 0) return [];

    // Downsample to max 200 points so recharts doesn't choke
    const maxPoints = 200;
    const stride = Math.max(1, Math.floor(data.length / maxPoints));
    const sampled: typeof data = [];
    for (let i = 0; i < data.length; i += stride) {
      sampled.push(data[i]);
    }
    if (sampled[sampled.length - 1] !== data[data.length - 1]) {
      sampled.push(data[data.length - 1]);
    }

    let ema = sampled[0].loss;
    return sampled.map((d) => {
      ema = 0.9 * ema + 0.1 * d.loss;
      return { step: d.step, loss: +d.loss.toFixed(5), smoothed: +ema.toFixed(5) };
    });
  }, [metrics]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Loss</CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length === 0 ? (
          <Skeleton className="h-[300px] w-full rounded-lg" />
        ) : (
          <ChartContainer config={chartConfig} className="h-[300px] w-full">
            <LineChart accessibilityLayer data={chartData}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="step"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={40}
                tickFormatter={(v) => `${v}`}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={45}
                domain={["auto", "auto"]}
              />
              <ChartTooltip
                content={<ChartTooltipContent />}
                cursor={false}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="smoothed"
                stroke="var(--color-smoothed)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
