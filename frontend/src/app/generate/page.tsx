"use client";

import { useEffect, useState } from "react";
import { Sparkles, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { GenerateBar } from "@/components/generate/generate-bar";
import { ImageWithLightbox } from "@/components/ui/image-lightbox";
import { api } from "@/lib/api";
import type { GenerationResult } from "@/types";

interface HistoryImage {
  path: string;
  filename: string;
}

function downloadImage(url: string, filename: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
}

export default function GeneratePage() {
  const [history, setHistory] = useState<GenerationResult[]>([]);
  const [savedImages, setSavedImages] = useState<HistoryImage[]>([]);
  const [loading, setLoading] = useState(false);
  const [pendingCount, setPendingCount] = useState(4);

  // Load saved generation history on mount
  useEffect(() => {
    api.getGenerationHistory().then((d) => {
      const data = d as { images: HistoryImage[]; total: number };
      setSavedImages(data.images);
    }).catch(() => {});
  }, []);

  async function handleGenerate(params: Record<string, unknown>) {
    setLoading(true);
    setPendingCount((params.batch_size as number) || 4);
    try {
      const r = await api.generate(params);
      setHistory((prev) => [r as GenerationResult, ...prev]);
      // Refresh saved history so new images appear even after refresh
      api.getGenerationHistory().then((d) => {
        const data = d as { images: HistoryImage[]; total: number };
        setSavedImages(data.images);
      }).catch(() => {});
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }

  // Combine live session images (base64) + saved disk images (paths)
  const liveImages = history.flatMap((r) =>
    r.images.map((src, i) => ({ src, seed: r.seed, sampler: r.sampler, steps: r.steps, index: i, type: "live" as const }))
  );

  // Saved images that aren't duplicated by live session
  const liveFilenames = new Set(
    history.flatMap(() => []) // can't easily match, so just show saved below live
  );

  return (
    <div className="relative -mx-4 -mt-5 sm:-mx-6 min-h-screen">
      <div className="px-4 pt-5 pb-44 sm:px-6">
        {/* Loading skeletons */}
        {loading && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 mb-4">
            {Array.from({ length: pendingCount }).map((_, i) => (
              <div key={i} className="aspect-video rounded-xl bg-secondary animate-pulse" />
            ))}
          </div>
        )}

        {/* Empty state */}
        {liveImages.length === 0 && savedImages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center min-h-[70vh]">
            <Sparkles className="h-12 w-12 text-muted-foreground/20 mb-3" />
            <p className="text-sm text-muted-foreground">Generate your first images</p>
          </div>
        )}

        {/* Live session images (base64, newest first) */}
        {liveImages.length > 0 && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 mb-4">
            {liveImages.map((img, i) => (
              <ImageCard
                key={`live-${img.seed}-${img.index}-${i}`}
                src={img.src}
                label={`seed ${img.seed} / ${img.sampler} / ${img.steps}s`}
                onDownload={() => downloadImage(img.src, `napsgen_${img.seed}_${img.index}.png`)}
              />
            ))}
          </div>
        )}

        {/* Saved history images (from disk, persist across refresh) */}
        {savedImages.length > 0 && liveImages.length === 0 && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {savedImages.map((img) => (
              <ImageCard
                key={img.filename}
                src={img.path}
                label={img.filename.replace(".png", "")}
                onDownload={() => downloadImage(img.path, img.filename)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Floating generate bar */}
      <div className="pointer-events-none fixed bottom-16 left-0 right-0 h-40 md:bottom-0 md:left-52 bg-gradient-to-t from-background via-background/80 to-transparent z-30" />
      <div className="fixed bottom-16 left-0 right-0 md:bottom-0 md:left-52 z-40 px-4 pb-5 pt-2">
        <div className="mx-auto max-w-2xl">
          <GenerateBar onGenerate={handleGenerate} loading={loading} />
        </div>
      </div>
    </div>
  );
}

function ImageCard({ src, label, onDownload }: { src: string; label: string; onDownload: () => void }) {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-border bg-secondary">
      <ImageWithLightbox
        src={src}
        alt={label}
        className="block w-full"
        imgClassName="w-full h-auto"
      />
      <div className="absolute inset-x-0 bottom-0 translate-y-full opacity-0 transition-all duration-200 ease-out group-hover:translate-y-0 group-hover:opacity-100 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3 pt-10">
        <div className="flex items-end justify-between">
          <span className="text-[11px] text-white/70 truncate mr-2">{label}</span>
          <Button
            size="xs"
            variant="secondary"
            className="opacity-80 hover:opacity-100 shrink-0"
            onClick={(e) => { e.stopPropagation(); onDownload(); }}
          >
            <Download className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
}
