"use client";

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useWebSocket } from "@/hooks/use-websocket";
import type { WSMessage } from "@/types";

/**
 * Global hook that listens to ALL WS events and shows persistent
 * toasts for background tasks (scraping, training, generation).
 * Mount once in layout -- works across page navigation.
 */
export function useBackgroundToasts() {
  const { subscribe } = useWebSocket();
  const scraperToastId = useRef<string | number | undefined>(undefined);
  const trainingToastId = useRef<string | number | undefined>(undefined);

  useEffect(() => {
    const unsubs: (() => void)[] = [];

    // ── Scraper toasts ──
    unsubs.push(subscribe("scraper_started", (msg) => {
      if (msg.type === "scraper_started") {
        scraperToastId.current = toast.loading(
          `Scraping Roblox images (0/${msg.target})...`,
          { duration: Infinity, id: "scraper" }
        );
      }
    }));

    unsubs.push(subscribe("scraper_progress", (msg) => {
      if (msg.type === "scraper_progress") {
        toast.loading(
          `Scraping: ${msg.downloaded}/${msg.target} downloaded`,
          { id: "scraper", duration: Infinity }
        );
      }
    }));

    unsubs.push(subscribe("scraper_complete", (msg) => {
      if (msg.type === "scraper_complete") {
        toast.success(
          `Scraping done! ${msg.downloaded} images downloaded${msg.dedup_removed ? `, ${msg.dedup_removed} dupes removed` : ""}`,
          { id: "scraper", duration: 5000 }
        );
      }
    }));

    unsubs.push(subscribe("scraper_stopped", (msg) => {
      if (msg.type === "scraper_stopped") {
        toast.info(`Scraper stopped. ${msg.downloaded} images saved.`, { id: "scraper", duration: 4000 });
      }
    }));

    unsubs.push(subscribe("scraper_error", (msg) => {
      if (msg.type === "scraper_error") {
        toast.error(`Scraper error: ${msg.error}`, { id: "scraper", duration: 6000 });
      }
    }));

    // ── Training toasts ──
    unsubs.push(subscribe("training_started", (msg) => {
      if (msg.type === "training_started") {
        trainingToastId.current = toast.loading(
          `Training started: ${msg.n_images} images, ${msg.param_count.toLocaleString()} params`,
          { duration: Infinity, id: "training" }
        );
      }
    }));

    unsubs.push(subscribe("metric", (msg) => {
      if (msg.type === "metric" && msg.step % 25 === 0) {
        toast.loading(
          `Training: step ${msg.step}/${msg.total_steps}, loss ${msg.loss.toFixed(4)}`,
          { id: "training", duration: Infinity }
        );
      }
    }));

    unsubs.push(subscribe("training_stopped", () => {
      toast.info("Training stopped.", { id: "training", duration: 4000 });
    }));

    unsubs.push(subscribe("training_error", (msg) => {
      if (msg.type === "training_error") {
        toast.error(`Training error: ${msg.error}`, { id: "training", duration: 8000 });
      }
    }));

    unsubs.push(subscribe("sample", (msg) => {
      if (msg.type === "sample") {
        toast.success(`New samples at step ${msg.step}`, { duration: 3000 });
      }
    }));

    // ── Generation toasts ──
    unsubs.push(subscribe("generation_complete", (msg) => {
      if (msg.type === "generation_complete") {
        toast.success(
          `Generated ${msg.batch_size} image${msg.batch_size > 1 ? "s" : ""} in ${msg.elapsed}s`,
          { duration: 3000 }
        );
      }
    }));

    // ── Dataset toasts ──
    unsubs.push(subscribe("dataset_update", (msg) => {
      if (msg.type === "dataset_update" && msg.action === "dedup_complete") {
        toast.success("Deduplication complete.", { duration: 3000 });
      }
    }));

    return () => unsubs.forEach((u) => u());
  }, [subscribe]);
}
