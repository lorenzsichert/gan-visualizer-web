/**
 * AudioWorkletProcessor that captures microphone PCM and computes a linear
 * magnitude spectrum, mirroring the Python `recording.get_sample` pipeline:
 *
 *   samples      = 512 floats (FFT size, matches the Python `blocksize`)
 *   windowed     = samples * Hann window
 *   spectrum     = |rfft(windowed)|  ->  257 bins
 *
 * Unlike the Python app (which blocks until a fresh 512-sample chunk arrives,
 * ~11.6 ms of latency), a *sliding* 512-sample window ending at the current
 * sample is transformed after every 128-frame render quantum (~2.9 ms). This
 * cuts the capture-side latency from up to 11.6 ms to ~2.9 ms with no change
 * to the spectrum content the render loop consumes.
 *
 * Each message posts the spectrum plus the audio-clock frame index of its
 * newest sample, so the main thread can measure the true audio -> GAN latency.
 */
class AudioSpectrumProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = 512;
    this.bins = this.block / 2 + 1;
    // Ring buffer holding the last `block` samples (sliding window).
    this.ring = new Float32Array(this.block);
    this.pos = 0;
    this.filled = 0;

    this.win = new Float32Array(this.block);
    for (let i = 0; i < this.block; i++) {
      this.win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this.block - 1)));
    }

    this.re = new Float32Array(this.block);
    this.im = new Float32Array(this.block);
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
    // Fresh spectrum after every render quantum. Skip until the ring holds a
    // full 512-sample window (first ~11.6 ms of capture).
    if (this.filled < this.block) return true;
    this.compute(ch.length);
    return true;
  }

  compute(chLen) {
    const n = this.block;
    const re = this.re, im = this.im, win = this.win, ring = this.ring, pos = this.pos;
    // Re-window the last `n` samples in capture order.
    for (let k = 0; k < n; k++) {
      const idx = (pos + k) % n;
      re[k] = ring[idx] * win[k];
      im[k] = 0;
    }
    fft(re, im, n);
    const out = this.spectrum;
    for (let i = 0; i < this.bins; i++) {
      out[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
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

/** Iterative radix-2 complex FFT in place. n must be a power of two. */
function fft(re, im, n) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cRe = 1, cIm = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = i + k + half;
        const uRe = re[a], uIm = im[a];
        const vRe = re[b] * cRe - im[b] * cIm;
        const vIm = re[b] * cIm + im[b] * cRe;
        re[a] = uRe + vRe; im[a] = uIm + vIm;
        re[b] = uRe - vRe; im[b] = uIm - vIm;
        const nRe = cRe * wRe - cIm * wIm;
        cIm = cRe * wIm + cIm * wRe;
        cRe = nRe;
      }
    }
  }
}

registerProcessor('audio-spectrum', AudioSpectrumProcessor);
