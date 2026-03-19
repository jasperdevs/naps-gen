"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps {
  className?: string
  value?: number[]
  defaultValue?: number[]
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  onValueChange?: (value: number[]) => void
}

function Slider({
  className,
  value,
  defaultValue,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  onValueChange,
}: SliderProps) {
  const current = value?.[0] ?? defaultValue?.[0] ?? min
  const pct = ((current - min) / (max - min)) * 100

  return (
    <div
      data-slot="slider"
      className={cn("relative flex w-full touch-none items-center select-none", disabled && "opacity-50", className)}
    >
      <div
        data-slot="slider-track"
        className="relative h-1 w-full rounded-full bg-muted cursor-pointer"
        onClick={(e) => {
          if (disabled) return
          const rect = e.currentTarget.getBoundingClientRect()
          const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
          const raw = min + x * (max - min)
          const stepped = Math.round(raw / step) * step
          const clamped = Math.max(min, Math.min(max, stepped))
          onValueChange?.([clamped])
        }}
      >
        <div
          data-slot="slider-range"
          className="absolute h-full rounded-full bg-primary"
          style={{ width: `${pct}%` }}
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={current}
        disabled={disabled}
        onChange={(e) => onValueChange?.([Number(e.target.value)])}
        className="absolute inset-0 w-full opacity-0 cursor-pointer"
      />
      <div
        data-slot="slider-thumb"
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 block size-3.5 shrink-0 rounded-full border border-ring bg-white shadow-sm pointer-events-none"
        style={{ left: `${pct}%` }}
      />
    </div>
  )
}

export { Slider }
