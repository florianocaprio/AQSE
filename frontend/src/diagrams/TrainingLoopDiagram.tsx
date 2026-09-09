export function TrainingLoopDiagram() {
  return (
    <div className="training-loop" aria-label="Declared AQSE QNG training data flow">
      <div className="training-flow">
        <span className="training-stage">X train</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Approved encoding · not connected</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">VQC(θ)</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">TQK fidelity kernel Kθ</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Centered alignment loss L(Kθ, y train)</span>
      </div>
      <div className="training-flow" aria-label="Distinct gradient and geometry inputs">
        <span className="training-stage">Loss L</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Gradient ∇θL</span>
        <span className="training-stage">VQC states + derivatives</span>
        <span aria-hidden="true">→</span>
        <span className="training-stage">Empirical FS metric ḡ</span>
        <span aria-hidden="true">↘</span>
        <span className="training-stage">Damped QNG update</span>
      </div>
      <div className="theta-return" aria-label="QNG updates theta and returns it to the VQC">
        <span>θ update</span>
        <span aria-hidden="true">↳ returns to VQC(θ), never to the sensor simulator</span>
      </div>
    </div>
  );
}
