import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/services/api";
import type { HealthStatus } from "@/types";

/** Poll backend readiness so the header can show a live connection badge. */
export function useHealth() {
  return useQuery<HealthStatus | null>({
    queryKey: ["health"],
    queryFn: ({ signal }) => fetchHealth(signal),
    refetchInterval: 30_000,
    staleTime: 20_000,
  });
}
