import { WORKSHEETS } from "./worksheetRegistry";
import type { WorksheetId } from "../types/workbench";

type WorkbookNavigationProps = {
  active: WorksheetId;
  onNavigate: (worksheet: WorksheetId) => void;
};

export function WorkbookNavigation({
  active,
  onNavigate,
}: WorkbookNavigationProps) {
  return (
    <nav className="workbook-navigation" aria-label="AQSE scientific worksheets">
      <div className="workbook-mark" aria-hidden="true">
        <span>AQ</span>
        <strong>SE</strong>
      </div>
      <div className="worksheet-links">
        {WORKSHEETS.map((worksheet) => (
          <button
            key={worksheet.id}
            type="button"
            className={active === worksheet.id ? "worksheet-link active" : "worksheet-link"}
            aria-current={active === worksheet.id ? "page" : undefined}
            onClick={() => onNavigate(worksheet.id)}
          >
            <span>{worksheet.index}</span>
            <strong>{worksheet.label}</strong>
          </button>
        ))}
      </div>
      <div className="workbook-mode">
        <span>MODE</span>
        <strong>LOCAL</strong>
      </div>
    </nav>
  );
}
