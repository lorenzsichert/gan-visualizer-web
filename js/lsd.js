/**
 * Port of the Python `lsd.py` LSDLatent class (lucid-sonic-dreams style
 * latent-space modulation) to plain JS using Float32Array.
 *
 * Mirrors the Python implementation exactly:
 *  - base latent is a truncated normal in [-2, 2] scaled by `truncation`
 *  - pulse pushes the latent along the audio/brightness direction (smoothed EMA)
 *  - motion is a per-dimension signed random walk (smoothed EMA, accumulated)
 *  - motion signs bounce at +/- 2*truncation
 *  - randomness factors re-roll every 4 seconds (time-based)
 *  - all smoothing EMAs and the motion step are normalized to a 60 fps
 *    reference so every parameter is frame-rate independent
 *
 * Pulse modes:
 *  0 = Classic  (uniform push), 1 = Audio, 2 = Brightness,
 *  3 = Brightness + Classic,    4 = Brightness + Audio
 */

export function truncatedNormal(lo = -2, hi = 2) {
  // Box-Muller standard normal with rejection to the [lo, hi] range.
  for (;;) {
    let u = Math.random() * 2 - 1;
    let v = Math.random() * 2 - 1;
    const s = u * u + v * v;
    if (s >= 1 || s === 0) continue;
    const z = u * Math.sqrt((-2 * Math.log(s)) / s);
    if (z >= lo && z <= hi) return z;
  }
}

export function randn(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = truncatedNormal(-8, 8);
  return a;
}

export class LSDLatent {
  constructor(dim, truncation = 1.0, motionRandomness = 0.5) {
    this.dim = dim;
    this.reset(truncation, motionRandomness);
  }

  reset(truncation = 1.0, motionRandomness = 0.5) {
    const d = this.dim;
    this.baseUnit = new Float32Array(d);
    for (let i = 0; i < d; i++) this.baseUnit[i] = truncatedNormal(-2, 2);
    this.noise = new Float32Array(this.baseUnit);
    this.pulseNoise = new Float32Array(d);
    this.audioNoise = new Float32Array(d);
    this.brightnessNoise = new Float32Array(d);
    this.motionNoise = new Float32Array(d);
    this.motionCum = new Float32Array(d);
    this.motionSigns = new Float32Array(d);
    for (let i = 0; i < d; i++) this.motionSigns[i] = Math.random() < 0.5 ? 1 : -1;
    this.randFactors = new Float32Array(d);
    for (let i = 0; i < d; i++) this.randFactors[i] = Math.random() < 0.5 ? 1.0 : 1.0 - motionRandomness;
    this.randPush = new Float32Array(d);
    for (let i = 0; i < d; i++) this.randPush[i] = Math.random();
    this.lastMotionRandomness = motionRandomness;
    this.frame = 0;
    this.time = 0;
  }

  step({
    pulseAmp, motionAmp, music = null,
    pulseMode = 0, pulseReact = 0.5, pulsePower = 1.0,
    brightnessReact = 0.5, brightnessDir = null,
    motionReact = 0.5, motionRandomness = 0.5,
    truncation = 1.0, fps = 60.0,
    pulseSmooth = 0.75, motionSmooth = 0.75,
  }) {
    const d = this.dim;
    const first = this.frame === 0;
    this.frame += 1;
    const dt = 1 / Math.max(fps, 1e-6);
    this.time += dt;

    // Re-roll randomness factors immediately when the slider changes,
    // otherwise changes only land on the periodic 4s re-roll.
    if (motionRandomness !== this.lastMotionRandomness) {
      this.lastMotionRandomness = motionRandomness;
      for (let i = 0; i < d; i++) this.randFactors[i] = Math.random() < 0.5 ? 1.0 : 1.0 - motionRandomness;
    }

    // Time-based 4s re-roll (was frame-count based, i.e. FPS-dependent).
    if (!first && this.time >= 4.0) {
      this.time = 0;
      for (let i = 0; i < d; i++) this.randFactors[i] = Math.random() < 0.5 ? 1.0 : 1.0 - motionRandomness;
    }

    const m = motionReact * 20.0 / fps;
    let p = pulseAmp;
    if (pulsePower !== 1.0) p = Math.pow(p, pulsePower) * 0.1;

    const pulseAdd = new Float32Array(d);
    for (let i = 0; i < d; i++) pulseAdd[i] = this.randPush[i] * pulseReact * p;

    let audioAdd = new Float32Array(d);
    if (music != null) audioAdd = normalizeAs(music, d).map(x => x * pulseReact * p);

    let brightAdd = new Float32Array(d);
    if (brightnessDir != null) brightAdd = normalizeAs(brightnessDir, d).map(x => x * brightnessReact * p);

    const motionAdd = new Float32Array(d);
    for (let i = 0; i < d; i++) motionAdd[i] = m * motionAmp * this.motionSigns[i] * this.randFactors[i];

    if (!first) {
      // EMA smoothing normalized to a 60 fps reference so the real-time
      // response is identical at any frame rate: per-frame factor s^(60/fps),
      // giving total decay s^(60*t) regardless of fps.
      const ps = Math.pow(pulseSmooth, 60 / fps);
      const ms = Math.pow(motionSmooth, 60 / fps);
      for (let i = 0; i < d; i++) {
        pulseAdd[i] = this.pulseNoise[i] * ps + pulseAdd[i] * (1 - ps);
        audioAdd[i] = this.audioNoise[i] * ps + audioAdd[i] * (1 - ps);
        brightAdd[i] = this.brightnessNoise[i] * ps + brightAdd[i] * (1 - ps);
        motionAdd[i] = this.motionNoise[i] * ms + motionAdd[i] * (1 - ms);
      }
    }

    this.pulseNoise.set(pulseAdd);
    this.audioNoise.set(audioAdd);
    this.brightnessNoise.set(brightAdd);
    this.motionNoise.set(motionAdd);
    for (let i = 0; i < d; i++) this.motionCum[i] += motionAdd[i];

    let push;
    const mode = Math.round(pulseMode);
    if (mode === 1) push = audioAdd;
    else if (mode === 2) push = brightAdd;
    else if (mode === 3) push = sum(pulseAdd, brightAdd);
    else if (mode === 4) push = sum(audioAdd, brightAdd);
    else push = pulseAdd;

    // Truncation scales the base latent every frame so the slider is live.
    for (let i = 0; i < d; i++) this.noise[i] = truncation * this.baseUnit[i] + push[i] + this.motionCum[i];

    const lo = -2.0 * truncation, hi = 2.0 * truncation;
    for (let i = 0; i < d; i++) {
      if (this.noise[i] - m < lo) this.motionSigns[i] = 1.0;
      else if (this.noise[i] + m >= hi) this.motionSigns[i] = -1.0;
    }

    return this.noise;
  }
}

function sum(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function normalizeAs(x, dim) {
  // Unit-normalize x (Float32Array or Array) and resize to `dim`.
  const v = new Float32Array(dim);
  const src = (x instanceof Float32Array || x instanceof Array) ? x : x;
  const n = Math.min(src.length, dim);
  for (let i = 0; i < n; i++) v[i] = src[i];
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}
