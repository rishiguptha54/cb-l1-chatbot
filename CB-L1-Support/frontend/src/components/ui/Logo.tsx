import { APP_NAME } from "@/config/app";
import { cn } from "@/utils/cn";

interface LogoProps {
  className?: string;
  /** Icon box size in pixels. */
  size?: number;
  withWordmark?: boolean;
  subtitle?: string;
}

/** Scattered defect "nodes" on a dark tile — the CB L1 Support Chatbot brand mark. */
const DOTS: { cx: number; cy: number; fill: string }[] = [
  { cx: 7.5, cy: 7, fill: "#10b981" },
  { cx: 13.5, cy: 6, fill: "#3b82f6" },
  { cx: 17.5, cy: 10, fill: "#f59e0b" },
  { cx: 6.5, cy: 13.5, fill: "#3b82f6" },
  { cx: 12, cy: 13, fill: "#10b981" },
  { cx: 9, cy: 18, fill: "#f59e0b" },
];

export function Logo({ className, size = 36, withWordmark = false, subtitle }: LogoProps) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <span
        className="relative grid shrink-0 place-items-center rounded-xl bg-[#0b1220] shadow-glow ring-1 ring-white/10"
        style={{ width: size, height: size }}
        aria-hidden="true"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          style={{ width: size * 0.72, height: size * 0.72 }}
        >
          {DOTS.map((d, i) => (
            <circle key={i} cx={d.cx} cy={d.cy} r={1.8} fill={d.fill} />
          ))}
        </svg>
      </span>
      {withWordmark && (
        <span className="flex flex-col leading-none">
          <span className="text-[0.95rem] font-bold tracking-tight text-foreground">
            {APP_NAME}
          </span>
          {subtitle && (
            <span className="mt-0.5 text-[0.7rem] font-medium text-muted-foreground">
              {subtitle}
            </span>
          )}
        </span>
      )}
    </span>
  );
}
