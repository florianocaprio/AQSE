import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { ExperimentRecord } from "../types/workbench";
import {
  ExperimentCard,
  experimentDisclosurePolicy,
} from "./ExperimentsWorksheet";

const RECORD: ExperimentRecord = {
  id: "experiment-1",
  created_at: "2026-09-08T00:00:00Z",
  kind: "scalar_simulation",
  status: "completed",
  title: "Completed scalar simulation",
  request_snapshot: {
    domain: "scalar_simulation",
    configuration: {
      sensor_id: "LATENT-SNAPSHOT-MARKER",
      duration: 1,
      sampling_rate: 100,
      background_field: 4.5e-5,
      amplitude: 1e-6,
      frequency: 5,
      phase: 0,
      drift_rate: 0,
      noise_std: 1e-8,
      temperature: 293.15,
      anomaly_enabled: true,
      anomaly_time: 0.5,
      anomaly_amplitude: 2e-6,
      random_seed: 17,
    },
  },
  warnings: [],
  timings_ms: {},
};

describe("experiment blind-mode disclosure boundary", () => {
  it("blocks ledger export and snapshot disclosure while blind mode is enabled", () => {
    const policy = experimentDisclosurePolicy(true, 1);
    const markup = renderToStaticMarkup(
      <ExperimentCard
        record={RECORD}
        exposeRequestSnapshot={policy.exposeRequestSnapshots}
      />,
    );

    expect(policy.allowLedgerExport).toBe(false);
    expect(markup).not.toContain("Requested configuration snapshot");
    expect(markup).not.toContain("LATENT-SNAPSHOT-MARKER");
  });

  it("allows the user to disclose and export a non-empty ledger outside blind mode", () => {
    const policy = experimentDisclosurePolicy(false, 1);
    const markup = renderToStaticMarkup(
      <ExperimentCard
        record={RECORD}
        exposeRequestSnapshot={policy.exposeRequestSnapshots}
      />,
    );

    expect(policy.allowLedgerExport).toBe(true);
    expect(markup).toContain("Requested configuration snapshot");
    expect(markup).toContain("LATENT-SNAPSHOT-MARKER");
    expect(experimentDisclosurePolicy(false, 0).allowLedgerExport).toBe(false);
  });
});
