/**
 * AudioWorkletProcessor that captures microphone PCM and computes a linear
 * magnitude spectrum, mirroring the Python `recording.get_sample` pipeline:
 *
 *   samples      = 1024 floats analysed (zero-padded to 8192 for the FFT)
 *   windowed     = (samples - mean) * Hann window   (DC removed)
 *   spectrum     = |rfft(windowed)|  ->  4097 bins
 *
 * The transform length is 8192 (4097 bins, ~5.9 Hz grid at 48 kHz), but only
 * the newest 1024 samples are analysed. The 1024-sample Hann window has a
 * group delay of ~10.7 ms at 48 kHz and a true frequency resolution of ~47 Hz,
 * a deliberate compromise: short enough that z reacts immediately, long enough
 * to separate the low bands. (A full 8192-sample window would resolve ~5.9 Hz
 * but delay z by ~85 ms.) Zero-padding keeps the full 8192-point frequency
 * grid. Magnitudes are rescaled by MAG_REF / window so the absolute level
 * stays calibrated to the original 1024-point FFT — the spectrum display and
 * the audio -> GAN modulation gains are unaffected by the resolution change.
 *
 * The window mean is subtracted before windowing so any DC offset in the input
 * does not leak into the lowest bins — with a coarse FFT that leakage
 * otherwise pins the whole low-frequency end of the display at a constant
 * height even in silence.
 *
 * The sliding window ending at the current sample is transformed after every
 * HOP samples (~2.7 ms at 48 kHz), i.e. on every audio render quantum, so z is
 * built from the freshest possible spectrum.
 *
 * Each message posts the spectrum plus the audio-clock frame index of its
 * newest sample, so the main thread can measure the true audio -> GAN latency.
 */
const MAG_REF = 1024; // FFT size the magnitude calibration is referenced to
const FFT_SIZE = 8192; // transform size/bins
const WINDOW = 1024; // samples analysed per transform (time response)
const HOP = 128; // compute on every render quantum (~2.7 ms @ 48 kHz)

class AudioSpectrumProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = WINDOW;
    this.fftSize = FFT_SIZE;
    this.bins = FFT_SIZE / 2 + 1;
    this.hop = HOP;
    this.sinceCompute = 0;
    // Ring buffer holding the last `block` samples (sliding window).
    this.ring = new Float32Array(this.block);
    this.pos = 0;
    this.filled = 0;

    this.win = new Float32Array(this.block);
    for (let i = 0; i < this.block; i++) {
      this.win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this.block - 1)));
    }

    this.re = new Float32Array(this.fftSize);
    this.im = new Float32Array(this.fftSize);
    // Reused every block: postMessage structured-clones synchronously, so the
    // same buffer can be re-posted without allocating hundreds of buffers/sec.
    this.spectrum = new Float32Array(this.bins);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.ring[this.pos] = ch[i];
      this.pos = (this.pos + 1) % this.block;
      this.filled = Math.min(this.filled + 1, this.block);
    }
    // Skip until the ring holds a full window (first ~21 ms of capture).
    if (this.filled < this.block) return true;
    this.sinceCompute += ch.length;
    if (this.sinceCompute < this.hop) return true;
    this.sinceCompute = 0;
    this.compute(ch.length);
    return true;
  }

  compute(chLen) {
    const n = this.block;
    const N = this.fftSize;
    const re = this.re, im = this.im, win = this.win, ring = this.ring, pos = this.pos;
    // Mean of the window (DC), subtracted before windowing to keep a DC offset
    // out of the low bins.
    let mean = 0;
    for (let k = 0; k < n; k++) mean += ring[(pos + k) % n];
    mean /= n;
    // Re-window the last `n` samples in capture order, then zero-pad the rest
    // of the 8192-point transform input.
    for (let k = 0; k < n; k++) {
      const idx = (pos + k) % n;
      re[k] = (ring[idx] - mean) * win[k];
      im[k] = 0;
    }
    for (let k = n; k < N; k++) {
      re[k] = 0;
      im[k] = 0;
    }
    fft(re, im, N);
    const out = this.spectrum;
    // Rescale to the 1024-point reference so the resolution change doesn't
    // shift the absolute level seen by the display and the GAN modulation.
    const scale = MAG_REF / n;
    for (let i = 0; i < this.bins; i++) {
      out[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]) * scale;
    }
    // `currentFrame` is the frame index of this process block's first sample;
    // the newest sample just consumed is currentFrame + chLen - 1.
    this.port.postMessage({
      spectrum: out,
      sampleRate,
      frame: this.currentFrame + chLen - 1,
    });
  }
}

// Per-size FFT tables: bit-reversal permutation plus one full period of
// twiddles (cos/sin). Rebuilt once per FFT size, then every transform is plain
// table lookups — no per-butterfly sin/cos or twiddle recurrence.
const fftTables = new Map();
function tablesFor(n) {
  let t = fftTables.get(n);
  if (t) return t;
  let bits = 0;
  while (1 << bits < n) bits++;
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let x = i;
    let r = 0;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    rev[i] = r;
  }
  const half = n >> 1;
  const cos = new Float32Array(half);
  const sin = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    const a = (-2 * Math.PI * k) / n;
    cos[k] = Math.cos(a);
    sin[k] = Math.sin(a);
  }
  t = { rev, cos, sin };
  fftTables.set(n, t);
  return t;
}

/** Iterative radix-2 complex FFT in place. n must be a power of two. */
function fft(re, im, n) {
  const { rev, cos, sin } = tablesFor(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const tw = k * step;
        const cRe = cos[tw], cIm = sin[tw];
        const a = i + k, b = i + k + half;
        const uRe = re[a], uIm = im[a];
        const vRe = re[b] * cRe - im[b] * cIm;
        const vIm = re[b] * cIm + im[b] * cRe;
        re[a] = uRe + vRe; im[a] = uIm + vIm;
        re[b] = uRe - vRe; im[b] = uIm - vIm;
      }
    }
  }
}

registerProcessor('audio-spectrum', AudioSpectrumProcessor);
