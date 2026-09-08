import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { WorkbenchProvider } from "../state/workbench";
import { OverviewWorksheet } from "./OverviewWorksheet";

describe("OverviewWorksheet quantum diagnostics", () => {
  it("exposes diagnostics as an explicit disabled-until-ready action", () => {
    const markup = renderToStaticMarkup(
      <WorkbenchProvider>
        <OverviewWorksheet />
      </WorkbenchProvider>,
    );

    expect(markup).toContain("Quantum adapter");
    expect(markup).toContain("RUN QUANTUM DIAGNOSTICS");
    expect(markup).toContain("disabled");
  });
});
