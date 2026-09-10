import { renderToStaticMarkup } from "react-dom/server";
import type { JSX } from "react";
import { describe, expect, it } from "vitest";

import { DemoProvider } from "../state/demo";
import { WorkbenchProvider } from "../state/workbench";
import { AfseWorksheet } from "./AfseWorksheet";
import { ExperimentsWorksheet } from "./ExperimentsWorksheet";
import { FeaturesWorksheet } from "./FeaturesWorksheet";
import { NeuralWorksheet, formatObservableRuleStatus } from "./NeuralWorksheet";
import { QngWorksheet } from "./QngWorksheet";
import { QuantumWorksheet } from "./QuantumWorksheet";

function renderWorksheet(Worksheet: () => JSX.Element): string {
  return renderToStaticMarkup(
    <WorkbenchProvider>
      <DemoProvider>
        <Worksheet />
      </DemoProvider>
    </WorkbenchProvider>,
  );
}

describe("end-to-end demo worksheets", () => {
  it("keeps real-data panels explicit when artifacts and analysis are unavailable", () => {
    expect(renderWorksheet(FeaturesWorksheet)).toContain("No State8 values are fabricated");
    expect(renderWorksheet(QuantumWorksheet)).toContain("No compatible bundle applied");
    expect(renderWorksheet(AfseWorksheet)).toContain("No AFSE coordinates are shown");
    expect(renderWorksheet(NeuralWorksheet)).toContain("No real score vector is available");
    expect(renderWorksheet(NeuralWorksheet)).toContain("NON-CAUSAL");
    expect(formatObservableRuleStatus("COMMON_CHANGE_AMBIGUOUS")).toBe(
      "COMMON CHANGE AMBIGUOUS",
    );
  });

  it("exposes bounded training and read-only registry controls without auto-running work", () => {
    const qng = renderWorksheet(QngWorksheet);
    const experiments = renderWorksheet(ExperimentsWorksheet);
    expect(qng).toContain("Train bounded candidates");
    expect(qng).toContain("Apply compatible bundle pair");
    expect(qng).toContain("disabled");
    expect(experiments).toContain("Opening this worksheet never re-runs TEST");
    expect(experiments).toContain("Apply selected compatible pair");
    expect(experiments).toContain("Observation → human review → TRAIN approval");
    expect(experiments).toContain("NO AUTO-TRAINING");
    expect(experiments).toContain("Start bounded retraining from approved collections");
  });
});
