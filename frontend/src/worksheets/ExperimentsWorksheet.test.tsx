import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { ExperimentRecord } from "../types/workbench";
import type { DemoBundleSummary } from "../types/demo";
import {
  DemoBundleCard,
  ExperimentCard,
  experimentDisclosurePolicy,
  selectionFreezeForPair,
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

const NETWORK_BUNDLE: DemoBundleSummary = {
  bundle_id: "bundle-network",
  task_id: "aqse.network-pattern.v1",
  profile_id: "aqse.network-state8.v1",
  theta_candidate_name: "protected_qng",
  theta_id: "theta-1",
  accepted_qng_updates: 3,
  afse_method_id: "aqse.afse.nystrom-ridge32.v1",
  afse_reference_size: 32,
  afse_ridge_lambda: 1e-6,
  classifier_model_id: "classifier-1",
  class_order: ["NORMAL", "DEVICE_COMPATIBLE"],
  validation: {
    task_id: "aqse.network-pattern.v1",
    profile_id: "aqse.network-state8.v1",
    partition: "validation",
    balanced_accuracy: 0.75,
    macro_f1: 0.7,
    coverage: 0.9,
    sample_count: 32,
    eligible_count: 30,
    class_support: { NORMAL: 16, DEVICE_COMPATIBLE: 16 },
    class_recall: { NORMAL: 0.75, DEVICE_COMPATIBLE: 0.75 },
    uncertain_count: 2,
    heuristic_ood_count: 1,
    by_node_count: [],
    replay: null,
  },
  raw_baseline_validation: {
    task_id: "aqse.network-pattern.v1",
    profile_id: "aqse.network-state8.v1",
    partition: "validation",
    balanced_accuracy: 0.7,
    macro_f1: 0.65,
    coverage: 0.9,
    sample_count: 32,
    eligible_count: 30,
    class_support: { NORMAL: 16, DEVICE_COMPATIBLE: 16 },
    class_recall: { NORMAL: 0.7, DEVICE_COMPATIBLE: 0.6 },
    uncertain_count: 2,
    heuristic_ood_count: 0,
    by_node_count: [],
    replay: null,
  },
  validation_comparison: {
    partition: "validation",
    task_id: "aqse.network-pattern.v1",
    sample_count: 32,
    comparison: "quantum-afse-vs-same-architecture-raw-state8",
    tie_tolerance: 1e-12,
    afse_balanced_accuracy: 0.75,
    raw_balanced_accuracy: 0.7,
    balanced_accuracy_delta: 0.05,
    balanced_accuracy_delta_interval: {
      confidence_level: 0.95,
      lower: 0.01,
      upper: 0.09,
      replicates: 1000,
      seed: 2001006,
      resampling_unit: "independent_episode",
      degenerate: false,
    },
    afse_macro_f1: 0.7,
    raw_macro_f1: 0.65,
    macro_f1_delta: 0.05,
    macro_f1_delta_interval: {
      confidence_level: 0.95,
      lower: 0.01,
      upper: 0.09,
      replicates: 1000,
      seed: 2001006,
      resampling_unit: "independent_episode",
      degenerate: false,
    },
    outcome: "HELPED",
  },
  active: true,
};

describe("experiment blind-mode disclosure boundary", () => {
  it("resolves only bundle pairs published by the same persisted freeze", () => {
    const freeze = {
      freeze_id: "freeze-knowledge",
      study_artifact_id: "knowledge-study",
      study_content_digest: "digest",
      local_bundle_id: "local-saved",
      network_bundle_id: "network-saved",
    };

    expect(selectionFreezeForPair([freeze], "local-saved", "network-saved"))
      .toBe(freeze);
    expect(selectionFreezeForPair([freeze], "local-saved", "network-other"))
      .toBeNull();
  });

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

  it("renders saved bundle identity and measured validation metrics read-only", () => {
    const markup = renderToStaticMarkup(<DemoBundleCard bundle={NETWORK_BUNDLE} />);

    expect(markup).toContain("bundle-network");
    expect(markup).toContain("protected_qng");
    expect(markup).toContain("75.0%");
    expect(markup).toContain("HELPED");
    expect(markup).toContain("ACTIVE");
  });
});
