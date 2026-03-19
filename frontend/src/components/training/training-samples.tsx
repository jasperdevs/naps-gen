"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ImageWithLightbox } from "@/components/ui/image-lightbox";
import { ImageIcon } from "lucide-react";
import type { WSMessage } from "@/types";

export function TrainingSamples({ samples }: { samples: WSMessage[] }) {
  const msgs = samples
    .filter((s): s is Extract<WSMessage, { type: "sample" }> => s.type === "sample")
    .slice(-8);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">Samples</CardTitle>
      </CardHeader>
      <CardContent>
        {msgs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <ImageIcon className="h-8 w-8 mb-2 opacity-30" />
            <p className="text-xs">Samples appear during training</p>
          </div>
        ) : (
          <div className="space-y-4">
            {msgs.map((msg, idx) => (
              <div key={idx} className="space-y-1.5">
                <p className="text-xs text-muted-foreground">Step {msg.step}</p>
                <div className="grid grid-cols-4 gap-2">
                  {msg.images.map((src, i) => (
                    <div key={i} className="overflow-hidden rounded-lg border border-border bg-secondary">
                      <ImageWithLightbox
                        src={src}
                        alt={`Step ${msg.step} #${i}`}
                        className="block w-full"
                        imgClassName="w-full h-auto"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
