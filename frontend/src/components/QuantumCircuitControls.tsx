import { useId, useState } from "react";

const PARAMETER_COUNT = 16;
const MIN_ANGLE = -Math.PI;
const MAX_ANGLE = Math.PI;
const ANGLE_STEP = Math.PI / 1000;

type ParameterGroup = {
  title: string;
  description: string;
  start: number;
  end: number;
};

const PARAMETER_GROUPS: ParameterGroup[] = [
  {
    title: "Phase layer α · RZ",
    description: "θ0–θ7, one phase rotation per qubit",
    start: 0,
    end: 8,
  },
  {
    title: "Rotation layer β · RY",
    description: "θ8–θ15, one trainable rotation per qubit",
    start: 8,
    end: 16,
  },
];

export function QuantumCircuitControls() {
  const [parameters, setParameters] = useState<number[]>(() =>
    Array.from({ length: PARAMETER_COUNT }, () => 0),
  );

  const updateParameter = (index: number, value: number) => {
    if (!Number.isFinite(value)) return;
    const boundedValue = clamp(value, MIN_ANGLE, MAX_ANGLE);
    setParameters((current) =>
      current.map((parameter, currentIndex) =>
        currentIndex === index ? boundedValue : parameter,
      ),
    );
  };

  const resetParameters = () => {
    setParameters(Array.from({ length: PARAMETER_COUNT }, () => 0));
  };

  return (
    <section className="workspace-section" aria-labelledby="quantum-controls-heading">
      <div className="section-heading">
        <div>
          <p className="section-index">02 / QUANTUM CONFIGURATION</p>
          <h2 id="quantum-controls-heading">Quantum Circuit Parameters</h2>
          <p>
            Prepare a local 16-angle parameter draft for the validated TQK8
            circuit. This panel does not execute, train, or change the circuit.
          </p>
        </div>
        <div className="local-state-badge">
          <span>Parameter state</span>
          <strong>LOCAL DRAFT</strong>
        </div>
      </div>

      <div className="quantum-control-panel">
        <div className="quantum-notice" role="note">
          <div>
            <span className="notice-marker" aria-hidden="true">∅</span>
            <p>
              <strong>Not connected to execution</strong>
              Values are held in browser memory only and reset on page reload.
            </p>
          </div>
          <button type="button" className="text-button" onClick={resetParameters}>
            Reset all to zero
          </button>
        </div>

        <div className="parameter-groups">
          {PARAMETER_GROUPS.map((group) => (
            <fieldset className="parameter-group" key={group.title}>
              <legend>{group.title}</legend>
              <p className="group-description">{group.description}</p>
              <div className="parameter-grid">
                {parameters.slice(group.start, group.end).map((value, offset) => {
                  const index = group.start + offset;
                  return (
                    <ParameterControl
                      key={index}
                      index={index}
                      qubit={offset}
                      value={value}
                      onChange={(nextValue) => updateParameter(index, nextValue)}
                    />
                  );
                })}
              </div>
            </fieldset>
          ))}
        </div>

        <div className="parameter-vector">
          <span>θ parameter vector · radians</span>
          <output aria-label="Current local quantum circuit parameter vector">
            [{parameters.map((value) => formatParameter(value)).join(", ")}]
          </output>
        </div>
      </div>
    </section>
  );
}

type ParameterControlProps = {
  index: number;
  qubit: number;
  value: number;
  onChange: (value: number) => void;
};

function ParameterControl({
  index,
  qubit,
  value,
  onChange,
}: ParameterControlProps) {
  const rangeId = useId();
  const numberId = `${rangeId}-number`;
  const descriptionId = `${rangeId}-description`;

  return (
    <div className="parameter-control">
      <div className="parameter-label">
        <label htmlFor={rangeId}>θ{index}</label>
        <span>q{qubit}</span>
      </div>
      <input
        id={rangeId}
        className="parameter-range"
        type="range"
        min={MIN_ANGLE}
        max={MAX_ANGLE}
        step={ANGLE_STEP}
        value={clamp(value, MIN_ANGLE, MAX_ANGLE)}
        onChange={(event) => onChange(event.target.valueAsNumber)}
        aria-describedby={descriptionId}
      />
      <div className="parameter-number-row">
        <input
          id={numberId}
          type="number"
          min={MIN_ANGLE}
          max={MAX_ANGLE}
          step={ANGLE_STEP}
          value={value}
          onChange={(event) => onChange(event.target.valueAsNumber)}
          aria-label={`Theta ${index} numeric value in radians`}
          aria-describedby={descriptionId}
        />
        <span>rad</span>
      </div>
      <small id={descriptionId}>Range −π to +π</small>
    </div>
  );
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatParameter(value: number): string {
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  return normalized.toFixed(3);
}
