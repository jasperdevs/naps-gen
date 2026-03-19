"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface Props {
  src: string;
  alt?: string;
  className?: string;
  imgClassName?: string;
  onLoad?: () => void;
  loading?: "lazy" | "eager";
  decoding?: "async" | "auto" | "sync";
  fetchPriority?: "auto" | "high" | "low";
}

export function ImageWithLightbox({
  src,
  alt = "Image",
  className = "",
  imgClassName = "",
  onLoad,
  loading = "lazy",
  decoding = "async",
  fetchPriority = "auto",
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button type="button" onClick={() => setOpen(true)} className={cn("cursor-zoom-in", className)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          className={cn("h-auto w-full", imgClassName)}
          onLoad={onLoad}
          loading={loading}
          decoding={decoding}
          fetchPriority={fetchPriority}
          draggable={false}
        />
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-[90vw] max-h-[90vh] p-2 bg-black/95 border-border">
          <DialogTitle className="sr-only">{alt}</DialogTitle>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt={alt} className="max-h-[85vh] max-w-full mx-auto object-contain rounded" />
        </DialogContent>
      </Dialog>
    </>
  );
}
