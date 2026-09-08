import { afterEach, describe, expect, it, vi } from "vitest";

import { POLLED_HEALTH_ENDPOINTS, runQuantumDiagnostics } from "./health";

describe("quantum infrastructure status API", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("keeps explicit diagnostics out of periodic health polling", () => {
    expect(Object.values(POLLED_HEALTH_ENDPOINTS)).toEqual([
      "/api/health",
      "/api/quantum/health",
    ]);
    expect(Object.values(POLLED_HEALTH_ENDPOINTS)).not.toContain(
      "/api/quantum/diagnostics",
    );
  });

  it("runs diagnostics only through an explicit POST request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await runQuantumDiagnostics();

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/quantum/diagnostics",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
