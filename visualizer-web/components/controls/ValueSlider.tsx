"use client";

/**
 * A slider whose thumb shows its own value, as Neuronpedia's do. Worth the
 * custom component: the panel has up to eight of these and a separate value
 * readout per row would double its height.
 */

import { Slider } from "radix-ui";

import { cn } from "@/lib/utils";

interface ValueSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (value: number) => string;
  onChange: (value: number) => void;
  className?: string;
}

export function ValueSlider({
  label,
  value,
  min,
  max,
  step = 1,
  format,
  onChange,
  className,
}: ValueSliderProps) {
  const clamped = Math.min(Math.max(value, min), max);
  return (
    <div className={cn("flex items-center gap-x-2.5", className)}>
      <div className="w-[92px] shrink-0 text-right text-[10px] leading-tight font-medium tracking-wide text-slate-400 uppercase">
        {label}
      </div>
      <Slider.Root
        className="relative flex h-6 flex-1 touch-none items-center select-none"
        value={[clamped]}
        min={min}
        max={max}
        step={step}
        onValueChange={([next]) => onChange(next)}
      >
        <Slider.Track className="relative h-[6px] grow rounded-full border border-slate-300 bg-white">
          <Slider.Range className="absolute h-full rounded-full bg-sky-600" />
        </Slider.Track>
        {/* The thumb is the element with role="slider", so the name goes here. */}
        <Slider.Thumb
          aria-label={label}
          className="flex h-[18px] w-11 cursor-grab items-center justify-center rounded-full border border-sky-600 bg-white text-[9px] font-semibold tabular-nums text-sky-700 shadow transition-colors hover:bg-sky-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/50 active:cursor-grabbing"
        >
          {format ? format(clamped) : clamped}
        </Slider.Thumb>
      </Slider.Root>
    </div>
  );
}
