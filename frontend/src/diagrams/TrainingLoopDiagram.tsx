export function TrainingLoopDiagram() {
  const stages = [
    "Feature dataset",
    "VQC(θ)",
    "TQK",
    "Kernel",
    "Alignment loss",
    "Gradient + FS metric",
    "QNG",
  ];

  return (
    <div className="training-loop" aria-label="AQSE QNG training loop">
      <div className="training-flow">
        {stages.map((stage, index) => (
          <div className="training-stage-wrap" key={stage}>
            <span className="training-stage">{stage}</span>
            {index < stages.length - 1 && <span aria-hidden="true">→</span>}
          </div>
        ))}
      </div>
      <div className="theta-return" aria-label="QNG updates theta and returns it to the VQC">
        <span>θ update</span>
        <span aria-hidden="true">↳ returns to VQC(θ), not to the sensor</span>
      </div>
    </div>
  );
}
