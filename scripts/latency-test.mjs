#!/usr/bin/env node
/**
 * End-to-end latency test for the audio -> GAN-latent DSP chain.
 *
 * Re-implements the exact signal path from the web app and measures how long a
 * musical event takes to reach the GAN latent input:
 *
 *   mic -> AudioWorklet (512-sample block + Hann FFT)   [audio-worklet.js]
 *        -> postMessage -> main thread spectrum copy    [audio.js]
 *        -> 60 fps render loop read                     [main.js loop()]
 *        -> computeLatent EMA smoothing                 [main.js]
 *        -> lsd.step pulse/motion EMA                   [lsd.js]
 *        -> worker.postMessage({ type: 'z', ... })      [main.js]
 *
 * A step in the audio level is injected at a known time and the response of the
 * final latent input (the pulse-EMA output that becomes the GAN z) is measured:
 * how many ms until it reaches 50% / 63% / 90% of its steady-state value. Also
 * reports the pure capture-side staleness (block + emission + RAF sampling),
 * which is the minimum possible latency even with zero smoothing.
 *
 * Run:  node scripts/latency-test.mjs
 */

const SR = 44100;
const BLOCK = 512;
const QUANTUM = 128;
const BINS = BLOCK / 2 + 1;
const RAF_HZ = 60;

// ---- exact FFT from worklets/audio-worklet.js -----------------------------
function fft(re, im, n) {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
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

function hannWindow(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return a;
}

// ---- worklet (current: posts every 512-frame block) -----------------------
function makeWorkletBlock() {
  const w = { block: BLOCK, fill: 0, buf: new Float32Array(BLOCK), win: hannWindow(BLOCK), re: new Float32Array(BLOCK), im: new Float32Array(BLOCK), emitted: [], frame: 0 };
  w.process = (ch) => {
    for (let i = 0; i < ch.length; i++) {
      w.buf[w.fill++] = ch[i];
      if (w.fill >= w.block) {
        const newestMs = ((w.frame - 1) / SR) * 1000;
        for (let k = 0; k < BLOCK; k++) { w.re[k] = w.buf[k] * w.win[k]; w.im[k] = 0; }
        fft(w.re, w.im, BLOCK);
        const spec = new Float32Array(BINS);
        for (let k = 0; k < BINS; k++) spec[k] = Math.sqrt(w.re[k] * w.re[k] + w.im[k] * w.im[k]);
        w.emitted.push({ tMs: newestMs, spectrum: spec });
        w.fill = 0;
      }
      w.frame++;
    }
  };
  return w;
}

// ---- worklet (low-latency: sliding 512 window posted every 128 frames) -----
function makeWorkletSliding() {
  const w = { ring: new Float32Array(BLOCK), pos: 0, filled: 0, win: hannWindow(BLOCK), emitted: [], frame: 0 };
  w.process = (ch) => {
    for (let i = 0; i < ch.length; i++) {
      w.ring[w.pos] = ch[i];
      w.pos = (w.pos + 1) % BLOCK;
      w.filled = Math.min(w.filled + 1, BLOCK);
      w.frame++;
      if (w.frame % QUANTUM === 0 && w.filled === BLOCK) {
        const re = new Float32Array(BLOCK), im = new Float32Array(BLOCK);
        for (let k = 0; k < BLOCK; k++) re[k] = w.ring[(w.pos + k) % BLOCK] * w.win[k];
        fft(re, im, BLOCK);
        const spec = new Float32Array(BINS);
        for (let k = 0; k < BINS; k++) spec[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        w.emitted.push({ tMs: ((w.frame - 1) / SR) * 1000, spectrum: spec });
      }
    }
  };
  return w;
}

// ---- exact smoothing from main.js -----------------------------------------
function makeSmoother(factor) {
  const s = new Float32Array(BINS);
  return {
    step(spectrum, dtMs) {
      const dt = dtMs / 1000;
      const smoothing = 1 - Math.exp((-dt * 10) / Math.max(factor, 1e-6));
      for (let i = 0; i < BINS; i++) s[i] = smoothing * s[i] + (1 - smoothing) * spectrum[i];
      return s;
    },
  };
}

// ---- lsd.step pulse EMA from lsd.js ---------------------------------------
function makePulseEma(pulseSmooth, fps) {
  const ps = Math.pow(pulseSmooth, 60 / fps);
  let prev = new Float32Array(BINS);
  return {
    step(env) {
      const next = new Float32Array(BINS);
      for (let i = 0; i < BINS; i++) next[i] = prev[i] * ps + env[i] * (1 - ps);
      prev = next;
      return next;
    },
  };
}

function simulate({ smoothFactor = 0.16, pulseSmooth = 0.75, sliding = false, stepAtMs = 5000 }) {
  const w = sliding ? makeWorkletSliding() : makeWorkletBlock();
  const smoother = makeSmoother(smoothFactor);
  const pulse = makePulseEma(pulseSmooth, RAF_HZ);

  const totalFrames = Math.floor(12 * SR);
  const stepFrame = Math.round((stepAtMs / 1000) * SR);

  let stalenessSum = 0, stalenessN = 0, stalenessMax = 0;

  // Response tracker for the final latent input (pulse EMA output).
  let preStep = null;        // baseline (last output before the step entered)
  let started = false;       // step has entered the pipeline
  const samples = [];        // { tMs, l1 } response magnitude vs baseline

  let rafTimer = 0;
  const rafMs = 1000 / RAF_HZ;

  for (let frame = 0; frame < totalFrames; frame += QUANTUM) {
    const quantumNowMs = ((frame + QUANTUM - 1) / SR) * 1000;
    const ch = new Float32Array(QUANTUM);
    for (let i = 0; i < QUANTUM; i++) {
      const f = frame + i;
      if (f >= stepFrame) ch[i] = 1.0; // step (constant loud signal)
    }

    w.process(ch);

    rafTimer += (QUANTUM / SR) * 1000; // 2.9 ms real-time increment per quantum
    while (rafTimer >= rafMs) {
      rafTimer -= rafMs;
      const last = w.emitted[w.emitted.length - 1];
      if (!last) break;

      const readNowMs = quantumNowMs;
      stalenessSum += Math.max(0, readNowMs - last.tMs);
      stalenessN++;
      stalenessMax = Math.max(stalenessMax, Math.max(0, readNowMs - last.tMs));

      const smoothed = smoother.step(last.spectrum, rafMs);
      const out = pulse.step(smoothed);

      // Baseline = output while the newest sample of the FFT window is still
      // before the step, i.e. the window is entirely pre-step audio.
      if (last.tMs < stepAtMs) {
        preStep = new Float32Array(out);
        continue;
      }
      if (!started) {
        started = true;
        samples.length = 0;
      }
      let l1 = 0;
      for (let i = 0; i < BINS; i++) l1 += (out[i] - preStep[i]) ** 2;
      samples.push({ tMs: readNowMs, l1: Math.sqrt(l1) });
    }
  }

  // Steady state = mean of the last 30 response samples (>= ~0.5 s).
  const tail = samples.slice(-30);
  const finalL1 = tail.reduce((a, s) => a + s.l1, 0) / tail.length;

  const fracMs = (f) => {
    const target = finalL1 * f;
    for (const s of samples) if (s.l1 >= target) return s.tMs - stepAtMs;
    return null;
  };
  const blockMs = (sliding ? QUANTUM : BLOCK) / SR * 1000;

  return {
    blockMaxMs: blockMs,
    rafAvgMs: stalenessN ? stalenessSum / stalenessN : 0,
    rafMaxMs: stalenessMax,
    t50: fracMs(0.5),
    t63: fracMs(0.63),
    t90: fracMs(0.9),
  };
}

function fmt(x) { return x === null || x === undefined || !isFinite(x) ? 'n/a' : `${x.toFixed(0)} ms`; }

function report(label, r) {
  console.log(`--- ${label} ---`);
  console.log(`  capture-side staleness (block+emit+RAF): avg ${r.rafAvgMs.toFixed(1)} ms, max ${r.rafMaxMs.toFixed(1)} ms (hard floor ~${r.blockMaxMs.toFixed(1)} ms block)`);
  console.log(`  step -> GAN latent input reaches  50%   : ${fmt(r.t50)}`);
  console.log(`  step -> GAN latent input reaches  63%   : ${fmt(r.t63)}`);
  console.log(`  step -> GAN latent input reaches  90%   : ${fmt(r.t90)}`);
  console.log('');
}

console.log('Simulating 44.1 kHz, 512-bin Hann FFT, 60 fps render loop, step at t=5s.\n');
report('BEFORE  (block=512, Smoothing=0.16, Pulse Smooth=0.75)', simulate({}));
report('AFTER   (sliding window, Smoothing=0.5, Pulse Smooth=0.5)', simulate({ smoothFactor: 0.5, pulseSmooth: 0.5, sliding: true }));
report('MAX LAT  (sliding window, Smoothing=1.0, Pulse Smooth=0.3)', simulate({ smoothFactor: 1.0, pulseSmooth: 0.3, sliding: true }));
