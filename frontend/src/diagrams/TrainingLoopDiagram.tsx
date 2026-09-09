export function TrainingLoopDiagram() {
  return (
    <div className="training-loop" aria-label="AQSE bounded QNG training dependency flow">
      <div className="training-flow">
        <span className="training-stage">Observed TRAIN features</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Frozen encoder</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">VQC(θ)</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">TQK Kθ</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Loss(Kθ, y TRAIN)</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Gradient</span>
      </div>
      <div className="training-flow" aria-label="Distinct gradient and geometry inputs">
        <span className="training-stage">VQC states + derivatives</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Fubini–Study metric</span>
        <span aria-hidden="true">＋</span>
        <span className="training-stage">Gradient</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Damped QNG</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Accepted θ update</span>
      </div>
      <div className="theta-return" aria-label="QNG updates theta and returns it to the VQC">
        <span>θ update returns only to VQC(θ)</span>
        <span>Labels feed loss, not inference. The metric is not produced by the loss.</span>
      </div>
    </div>
  );
}
