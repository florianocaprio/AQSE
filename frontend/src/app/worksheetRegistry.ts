import type { WorksheetId } from "../types/workbench";

export type WorksheetDefinition = {
  id: WorksheetId;
  index: string;
  label: string;
  shortLabel: string;
};

export const WORKSHEETS: WorksheetDefinition[] = [
  { id: "overview", index: "01", label: "Overview", shortLabel: "Overview" },
  { id: "sensors", index: "02", label: "Sensors", shortLabel: "Sensors" },
  { id: "features", index: "03", label: "Features", shortLabel: "Features" },
  {
    id: "quantum",
    index: "04",
    label: "Quantum Engine",
    shortLabel: "Quantum",
  },
  {
    id: "qng",
    index: "05",
    label: "QNG Training",
    shortLabel: "QNG",
  },
  {
    id: "afse",
    index: "06",
    label: "Local Embedding / AFSE",
    shortLabel: "AFSE",
  },
  {
    id: "neural",
    index: "07",
    label: "Neural Model",
    shortLabel: "Neural",
  },
  {
    id: "experiments",
    index: "08",
    label: "Experiments",
    shortLabel: "Experiments",
  },
];
