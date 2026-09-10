import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TrainingLoopDiagram } from "./TrainingLoopDiagram";

describe("TrainingLoopDiagram", () => {
  it("shows labels, gradient and FS geometry as distinct declared inputs", () => {
    const markup = renderToStaticMarkup(<TrainingLoopDiagram />);
    expect(markup).toContain("Observed TRAIN features");
    expect(markup).toContain("Loss(Kθ, y TRAIN)");
    expect(markup).toContain("Gradient");
    expect(markup).toContain("Fubini–Study metric");
    expect(markup).toContain("Labels feed loss, not inference");
  });
});
