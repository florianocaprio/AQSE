import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TrainingLoopDiagram } from "./TrainingLoopDiagram";

describe("TrainingLoopDiagram", () => {
  it("shows labels, gradient and FS geometry as distinct declared inputs", () => {
    const markup = renderToStaticMarkup(<TrainingLoopDiagram />);
    expect(markup).toContain("X train");
    expect(markup).toContain("y train");
    expect(markup).toContain("Gradient ∇θL");
    expect(markup).toContain("Empirical FS metric ḡ");
    expect(markup).toContain("not connected");
  });
});
