import { useEffect, type JSX } from "react";

import { WorkbookNavigation } from "./app/WorkbookNavigation";
import { WORKSHEETS } from "./app/worksheetRegistry";
import { useWorkbench } from "./state/workbench";
import {
  useNetworkStream,
  useWorkbenchBootstrap,
} from "./state/useWorkbenchController";
import type { WorksheetId } from "./types/workbench";
import { AfseWorksheet } from "./worksheets/AfseWorksheet";
import { ExperimentsWorksheet } from "./worksheets/ExperimentsWorksheet";
import { FeaturesWorksheet } from "./worksheets/FeaturesWorksheet";
import { NeuralWorksheet } from "./worksheets/NeuralWorksheet";
import { OverviewWorksheet } from "./worksheets/OverviewWorksheet";
import { QngWorksheet } from "./worksheets/QngWorksheet";
import { QuantumWorksheet } from "./worksheets/QuantumWorksheet";
import { SensorsWorksheet } from "./worksheets/SensorsWorksheet";

const WORKSHEET_COMPONENTS: Record<WorksheetId, () => JSX.Element> = {
  overview: OverviewWorksheet,
  sensors: SensorsWorksheet,
  features: FeaturesWorksheet,
  quantum: QuantumWorksheet,
  qng: QngWorksheet,
  afse: AfseWorksheet,
  neural: NeuralWorksheet,
  experiments: ExperimentsWorksheet,
};

function App() {
  const { state, dispatch } = useWorkbench();
  useWorkbenchBootstrap();
  useNetworkStream();
  const ActiveWorksheet = WORKSHEET_COMPONENTS[state.active_worksheet];

  useEffect(() => {
    const onHashChange = () => {
      const candidate = window.location.hash.replace(/^#\/?/, "") as WorksheetId;
      if (WORKSHEETS.some(({ id }) => id === candidate)) {
        dispatch({ type: "NAVIGATE", worksheet: candidate });
      }
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [dispatch]);

  const navigate = (worksheet: WorksheetId) => {
    window.location.hash = `/${worksheet}`;
    dispatch({ type: "NAVIGATE", worksheet });
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  return (
    <div className="workbook-shell">
      <a className="skip-link" href="#worksheet-content">Skip to worksheet</a>
      <WorkbookNavigation active={state.active_worksheet} onNavigate={navigate} />
      <div className="workbook-main">
        <header className="workbook-topbar">
          <div>
            <span className="workbook-project">AQSE</span>
            <span>Adaptive Quantum Sensor Engine</span>
          </div>
          <div className="topbar-status" aria-label="Current environment status">
            <span className={`status-dot ${state.backend_health.status}`} />
            <span>Backend {state.backend_health.status}</span>
            <i aria-hidden="true" />
            <span>Milestone 1C</span>
          </div>
        </header>
        <main id="worksheet-content" tabIndex={-1}>
          <ActiveWorksheet />
        </main>
        <footer className="workbook-footer">
          <span>AQSE local research workbench · Milestone 1C</span>
          <span>No physical QPU · no AFSE math · no neural output</span>
        </footer>
      </div>
    </div>
  );
}

export default App;
