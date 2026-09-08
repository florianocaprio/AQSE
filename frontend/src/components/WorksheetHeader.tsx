import type { ReactNode } from "react";

type WorksheetHeaderProps = {
  titleId: string;
  index: string;
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
};

export function WorksheetHeader({
  titleId,
  index,
  eyebrow,
  title,
  description,
  actions,
}: WorksheetHeaderProps) {
  return (
    <header className="worksheet-header">
      <div className="worksheet-number" aria-hidden="true">{index}</div>
      <div className="worksheet-heading-copy">
        <p className="section-index">{eyebrow}</p>
        <h1 id={titleId}>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="worksheet-actions">{actions}</div>}
    </header>
  );
}
