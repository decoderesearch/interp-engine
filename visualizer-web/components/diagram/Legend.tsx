"use client";

import { cn } from "@/lib/utils";

const ROLES = [
  { label: "Residual", className: "bg-role-resid" },
  { label: "Attention", className: "bg-role-attn" },
  { label: "MLP", className: "bg-role-mlp" },
  { label: "Routing", className: "bg-role-route" },
  { label: "Global", className: "bg-role-global" },
];

interface Props {
  /** Adds the red ring, which only means anything with two diagrams up. */
  comparing?: boolean;
  className?: string;
}

export function Legend({ comparing = false, className }: Props) {
  return (
    // Two columns, three from `lg`, which is exactly where the sidebar stops
    // being a sheet over the diagram and becomes a 320px column beside it.
    // Five short labels stacked one per line spent the sidebar's scarcest
    // resource -- vertical space, shared with the controls under it -- on a
    // list whose longest entry is nine characters.
    <div
      className={cn(
        "grid grid-cols-2 gap-x-2 gap-y-1 lg:grid-cols-3",
        className,
      )}
    >
      {ROLES.map((role) => (
        <Row key={role.label} label={role.label}>
          <span className={cn("h-2 w-2 rounded-full", role.className)} />
        </Row>
      ))}

      {/* Spans the row it lands on: it is twice the length of any other label,
          and left in one cell it wraps to two lines while its neighbours sit
          on one. */}
      {comparing && (
        <Row label="Engine difference" className="col-span-2 lg:col-span-3">
          <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full border border-red-500 bg-red-100">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
          </span>
        </Row>
      )}
    </div>
  );
}

/** A fixed icon slot, so a ring and a dot start their labels at the same x. */
function Row({
  label,
  children,
  className,
}: {
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "flex items-center gap-x-1.5 text-[10px] text-slate-500",
        className,
      )}
    >
      <span className="flex w-3.5 shrink-0 justify-center">{children}</span>
      {label}
    </span>
  );
}
