/**
 * Parameter definitions, a direct port of the Python app's `midi.settings`
 * (web/midi.py). Values are grouped for the control panel UI.
 *
 * All parameters remain loadable via `get()` so the rendering pipeline is
 * unchanged; entries marked `hidden: true` are simply not shown as sliders.
 */
export const SETTINGS = {
  "Pulse React":        { group: "Pulse",     value: 0.01, min: 0,    max: 0.1,    step: 0.001, dec: 2 },
  "Brightness React":   { group: "Pulse",     value: 0.04, min: 0,    max: 0.5,    step: 0.001, dec: 2 },
  "Motion React":       { group: "Motion",    value: 0.01, min: 0,    max: 0.5,    step: 0.001, dec: 2 },
  "Pulse Smooth":       { group: "Pulse",     value: 0.88,  min: 0,    max: 1,    step: 0.01, dec: 2, hidden: true },
  "Hue Shift":          { group: "Post",      value: 0.45, min: 0,    max: 5.0, step: 0.01, dec: 2 },
  "Motion Power":       { group: "Motion",    value: 1.39, min: 1.0,  max: 2.0, step: 0.01, dec: 2, hidden: true },
  "Smoothing Factor":   { group: "Timing",    value: 1.0,  min: 0,    max: 1,    step: 0.01, dec: 2, hidden: true },
  "Noise Weight":       { group: "Input",     value: 1.48, min: 0,    max: 10,   step: 0.01, dec: 2, hidden: true },
  "Audio Weight":       { group: "Input",     value: 0.1,  min: 0,    max: 2.0,  step: 0.01, dec: 2, hidden: true },
  "Audio Randomization":{ group: "Input",     value: 0.0,  min: 0,    max: 1,    step: 0.01, dec: 2, hidden: true },
  "Lowpass Sensivity":  { group: "Audio",     value: 0.47, min: 0,    max: 5.0,  step: 0.01, dec: 2, hidden: true },
  "Lowpass Cutoff":     { group: "Audio",     value: 128,  min: 0,    max: 256,  step: 1,    dec: 0, hidden: true },
  "Cutoff":             { group: "Audio",     value: 0,    min: 0,    max: 100,  step: 1,    dec: 0, hidden: true },
  "Truncation":         { group: "Pulse",     value: 1.0,  min: 0,    max: 2,    step: 0.01, dec: 2, hidden: true },
  "Pulse Mode":         { group: "Pulse",     value: 3.0,  min: 0,    max: 4,    step: 1,    dec: 0, hidden: true },
  "Pulse Power":        { group: "Pulse",     value: 1.0,  min: 0.5,  max: 3,    step: 0.01, dec: 2, hidden: true },
  "Motion Randomness":  { group: "Motion",    value: 0.50, min: 0,    max: 1,    step: 0.01, dec: 2, hidden: true },
  "Motion Smooth":      { group: "Motion",    value: 0.75, min: 0,    max: 1,    step: 0.01, dec: 2, hidden: true },
  "Tanh Output":        { group: "Post",      value: 0,    min: 0,    max: 1,    step: 1,    dec: 0, hidden: true },
};

export const GROUP_ORDER = ["Input", "Timing", "Audio", "Pulse", "Motion", "Post"];

export function get(name) {
  return SETTINGS[name].value;
}
