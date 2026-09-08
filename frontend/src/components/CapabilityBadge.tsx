import type { CapabilityStatus } from "../types/workbench";

const LABELS: Record<CapabilityStatus, string> = {
  implemented: "IMPLEMENTED",
  available_not_connected: "AVAILABLE · NOT CONNECTED",
  architecture_defined: "ARCHITECTURE DEFINED",
  not_implemented: "NOT IMPLEMENTED",
};

type CapabilityBadgeProps = {
  status: CapabilityStatus;
  compact?: boolean;
};

export function CapabilityBadge({
  status,
  compact = false,
}: CapabilityBadgeProps) {
  return (
    <span className={`capability-badge ${status}${compact ? " compact" : ""}`}>
      {LABELS[status]}
    </span>
  );
}
