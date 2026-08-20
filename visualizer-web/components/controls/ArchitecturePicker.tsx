"use client";

import { ChevronDown } from "lucide-react";
import { Select } from "radix-ui";

import { ArchitectureInfo } from "@/components/controls/ArchitectureInfo";
import { ARCHITECTURES, displayModel } from "@/data/architectures";
import { cn } from "@/lib/utils";

/** Radix rejects "" as an item value, so the empty choice needs a stand-in. */
const NONE = "__none__";

interface Props {
  /**
   * The accessible name. Also the caption, unless one is given — the caption
   * says when the selected family shipped, which is no use as a label for the
   * control that selects it.
   */
  label: string;
  /** Small-caps caption under the current value. */
  caption?: string;
  value: string | null;
  /** Shown, in slate rather than sky, when nothing is selected. */
  placeholder: string;
  onChange: (id: string | null) => void;
  /** Offers an explicit "none" item at the top of the list. */
  clearable?: boolean;
  /** Label for the none item, when clearable. */
  clearLabel?: string;
  /** The other picker's choice, left out here: a model does not differ from itself. */
  exclude?: string | null;
  /** Extra trigger classes, for the copies that sit on top of a diagram. */
  className?: string;
}

/**
 * Architectures rather than checkpoints: two Llama models differ in size, not
 * in where their hook points are, and size is what the sliders are for.
 */
export function ArchitecturePicker({
  label,
  caption,
  value,
  placeholder,
  onChange,
  clearable = false,
  clearLabel = "None",
  exclude = null,
  className,
}: Props) {
  const current =
    value === null ? null : ARCHITECTURES.find((a) => a.id === value);
  const options = ARCHITECTURES.filter((arch) => arch.id !== exclude);

  return (
    <Select.Root
      value={value ?? NONE}
      onValueChange={(next) => onChange(next === NONE ? null : next)}
    >
      <Select.Trigger
        aria-label={label}
        // Shorter and narrower below `sm`, where the caption is not drawn: with
        // one line in it the trigger does not need two lines of height, and the
        // width it gives back is what lets the timeline sit beside it there.
        className={cn(
          "flex h-9 min-w-[132px] cursor-pointer items-center gap-x-2 rounded-md border border-slate-300 bg-white pr-2 pl-2.5 text-left transition-colors hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 sm:h-11 sm:min-w-[208px]",
          className,
        )}
      >
        {/* No circled i in here. Nested inside the trigger it was unreachable by
            touch: the tap landed on the trigger and opened the list. It sits
            beside the picker instead, where it is its own target. */}
        <span className="flex min-w-0 flex-1 flex-col">
          <span
            className={`truncate font-mono text-xs font-medium ${
              current ? "text-sky-700" : "text-slate-400"
            }`}
          >
            {current?.label ?? placeholder}
          </span>
          {/* Dropped below `sm`. `RELEASED APR 2025` is a second line of text on
              the control that has the least room for one, and the date is on the
              timeline's own thumb right beside it. */}
          <span className="mt-0.5 hidden text-[8px] leading-none font-medium tracking-wide text-slate-400 uppercase sm:block">
            {caption ?? label}
          </span>
        </span>
        <Select.Icon>
          <ChevronDown className="h-4 w-4 text-slate-400" />
        </Select.Icon>
      </Select.Trigger>

      <Select.Portal>
        <Select.Content
          position="popper"
          align="start"
          sideOffset={4}
          className="z-50 max-h-[min(520px,70dvh)] overflow-hidden rounded-md border border-slate-200 bg-white shadow-[0px_10px_38px_-10px_rgba(22,23,24,0.35),0px_10px_20px_-15px_rgba(22,23,24,0.2)]"
        >
          <Select.Viewport className="thin-scrollbar max-h-[min(520px,70dvh)] overflow-y-auto">
            {clearable && (
              <Select.Item
                value={NONE}
                className="flex cursor-pointer flex-col items-start gap-y-0.5 border-b border-slate-100 px-3 py-2 focus:outline-none data-highlighted:bg-slate-100"
              >
                <Select.ItemText>
                  <span className="font-mono text-xs font-medium text-slate-500">
                    {clearLabel}
                  </span>
                </Select.ItemText>
              </Select.Item>
            )}
            {options.map((arch) => (
              <Select.Item
                key={arch.id}
                value={arch.id}
                className={`flex cursor-pointer items-center gap-x-3 border-b border-slate-100 py-2 pr-2.5 pl-3 last:border-b-0 focus:outline-none data-highlighted:bg-slate-100 ${
                  arch.id === value ? "bg-sky-50" : ""
                }`}
              >
                <span className="flex min-w-0 flex-1 flex-col items-start gap-y-0.5">
                  <Select.ItemText>
                    <span className="font-mono text-xs font-medium text-sky-700">
                      {arch.label}
                    </span>
                  </Select.ItemText>
                  <span className="text-[10px] leading-tight text-slate-400">
                    {arch.exampleModels
                      .slice(0, 2)
                      .map(displayModel)
                      .join(", ")}
                  </span>
                </span>
                {/* On the row's right edge rather than its left: the card
                    anchors on the icon, so from the left it opens on top of
                    the list it is meant to help you read. */}
                <ArchitectureInfo arch={arch} side="right" nested />
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
