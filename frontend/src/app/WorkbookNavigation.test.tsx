import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { WorkbookNavigation } from "./WorkbookNavigation";

describe("WorkbookNavigation", () => {
  it("renders every worksheet and identifies the active one", () => {
    const markup = renderToStaticMarkup(
      <WorkbookNavigation active="features" onNavigate={() => undefined} />,
    );
    expect(markup).toContain("Overview");
    expect(markup).toContain("Sensors");
    expect(markup).toContain("Local Embedding / AFSE");
    expect(markup).toContain("Experiments");
    expect(markup).toContain('aria-current="page"');
  });
});
