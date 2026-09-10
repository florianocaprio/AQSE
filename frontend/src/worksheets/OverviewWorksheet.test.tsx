import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DemoProvider } from "../state/demo";
import { WorkbenchProvider } from "../state/workbench";
import { OverviewWorksheet } from "./OverviewWorksheet";

describe("OverviewWorksheet quantum diagnostics", () => {
  it("exposes diagnostics as an explicit disabled-until-ready action", () => {
    const markup = renderToStaticMarkup(
      <WorkbenchProvider>
        <DemoProvider>
          <OverviewWorksheet />
        </DemoProvider>
      </WorkbenchProvider>,
    );

    expect(markup).toContain("Quantum adapter");
    expect(markup).toContain("RUN QUANTUM DIAGNOSTICS");
    expect(markup).toContain("disabled");
    expect(markup).toContain("CONTINUOUS ANALYSIS");
    expect(markup).toContain("No bundle applied");
  });
});
