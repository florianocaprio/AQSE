import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { DemoProvider } from "./state/demo";
import { WorkbenchProvider } from "./state/workbench";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <WorkbenchProvider>
      <DemoProvider>
        <App />
      </DemoProvider>
    </WorkbenchProvider>
  </StrictMode>,
);
