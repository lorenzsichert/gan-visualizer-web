/**
 * Main thread: UI, audio state, latent modulation (port of main.py
 * `update_frame` + the StyleGAN branch), rendering, and FPS tracking.
 *
 * Heavy work is split for maximum FPS:
 *   - Audio DSP  ......... AudioWorkletProcessor (off-thread FFT)
 *   - ONNX inference ..... inference worker (off-thread WASM, multi-threaded)
 *   - Rendering .......... this thread (single GPU blit)
 */
import { SETTINGS, GROUP_ORDER, get } from './settings.js';
import { LSDLatent, randn } from './lsd.js';
import { AudioPipeline } from './audio.js';
import { SpectrumView, aWeightDb } from './spectrum.js';

const W = 512;    // model output width
const H = 512;    // model output height
const BINS = 257; // spectrum bins (fftSize/2 + 1)

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');

const elStatus = document.getElementById('status');
const elFps = document.getElementById('fps');
const elInfer = document.getElementById('infer');
const elThreads = document.getElementById('threads');
const elProvider = document.getElementById('provider');
const elAudio = document.getElementById('audio');
const elToast = document.getElementById('bench-toast');
const elToastRows = document.getElementById('bench-rows');
const elToastNote = document.getElementById('bench-note');

// ---------------------------------------------------------------------------
// Persistent latent state (mirrors GANVisualizer.__init__ / _resize_latent_state)
// ---------------------------------------------------------------------------
let DIM = 512;
let lookup = new Uint16Array(DIM);
let a = randn(DIM);
let smoothed = new Float32Array(BINS);
let prevSpectrum = new Float32Array(BINS);
let spectrum = new Float32Array(BINS);
let audioNoise = new Float32Array(DIM);
let z = new Float32Array(DIM);
let lsd = null;
let brightnessDir = null;
let lastHueFlux = 0;
let audioRandTimer = 0;

// Per-bin A-weighting gain for the latent/LSD path, cached per sample rate
// (mirrors the panel display's perceptual weighting). The raw `spectrum` fed
// to SpectrumView is untouched so it isn't weighted twice.
let aWeightGain = null;
let aWeightSR = 0;

// Rebuild per-dimension state (called on model ready).
function initLatentState(dim) {
  DIM = dim;
  lookup = new Uint16Array(dim);
  for (let i = 0; i < dim; i++) lookup[i] = i % BINS;
  a = randn(dim);
  smoothed.fill(0);
  prevSpectrum.fill(0);
  lsd = new LSDLatent(dim, get('Truncation'), get('Motion Randomness'));
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------
const audio = new AudioPipeline();
const spectrumView = new SpectrumView(document.getElementById('spectrum'));
let demo = false;
// 0 = off, 1 = mirror (sharp tiles), 2 = mirror + blurred side tiles.
let mirrorMode = 2;

// A source's display filter for the spectrum view: its center frequency, band
// width, and react amount (plus the react range so the bell can be normalized).
function filterInfo(freqKey, widthKey, reactKey) {
  return {
    freq: get(freqKey),
    width: get(widthKey),
    react: get(reactKey),
    reactMin: SETTINGS[reactKey].min,
    reactMax: SETTINGS[reactKey].max,
  };
}

function genDemoSpectrum(t, out) {
  const beat1 = 0.5 + 0.5 * Math.sin(t * 2.1);
  const beat2 = 0.5 + 0.5 * Math.sin(t * 3.7);
  out.fill(0);
  const peaks = [
    { f: 8, amp: 0.9 * beat1, w: 3 },
    { f: 20, amp: 0.7 * beat1, w: 4 },
    { f: 45, amp: 0.6 * beat2, w: 6 },
    { f: 90, amp: 0.5 * beat2, w: 8 },
    { f: 150, amp: 0.35, w: 10 },
    { f: 220, amp: 0.3 * (0.5 + 0.5 * Math.sin(t * 5)), w: 12 },
  ];
  for (const p of peaks) {
    for (let i = 0; i < BINS; i++) {
      const d = (i - p.f) / p.w;
      out[i] += p.amp * Math.exp(-d * d * 0.5);
    }
  }
  for (let i = 0; i < BINS; i++) out[i] += 0.002 + Math.random() * 0.01;
}

// ---------------------------------------------------------------------------
// Inference worker
// ---------------------------------------------------------------------------
let worker = null;
let ready = false;

// The auto-calibrated compute config is cached so the provider/thread benchmark
// only runs once per machine/model; `?bench` in the URL forces a fresh one.
const CONFIG_KEY = 'compute-config';
const OVERRIDE_KEY = 'threads-override';
const PROVIDER_KEY = 'provider-override';

// null = auto (benchmarked); a number = the user's manual WASM thread override,
// which skips the benchmark on (re)start because the thread pool is baked into
// the WASM module at init.
let threadsOverride = loadThreadsOverride();
// null = auto (benchmarked); a string = the user's manual execution-provider
// override (e.g. "webgpu").
let providerOverride = loadProviderOverride();

// Which execution providers this browser exposes. Detection runs on the window
// (not inside the worker) so WebGPU/WebNN presence is reliable; the worker still
// verifies each one by actually running it and times out anything that fails.
function detectProviders() {
  const list = [];
  if (typeof navigator !== 'undefined' && navigator.gpu) list.push('webgpu');
  if (typeof navigator !== 'undefined' && navigator.ml) list.push('webnn');
  list.push('wasm');
  return list;
}

function threadOptions() {
  if (typeof SharedArrayBuffer === 'undefined') return [1];
  const hc = navigator.hardwareConcurrency || 4;
  const set = new Set([1, hc]);
  for (let t = 2; t <= hc; t *= 2) set.add(t);
  return [...set].sort((a, b) => a - b);
}

function loadThreadsOverride() {
  try {
    const n = Number(localStorage.getItem(OVERRIDE_KEY));
    if (Number.isInteger(n) && n >= 1) return n;
  } catch (err) {
    /* Corrupt or unavailable storage — fall back to auto. */
  }
  return null;
}

function loadProviderOverride() {
  try {
    const p = localStorage.getItem(PROVIDER_KEY);
    if (p && detectProviders().includes(p)) return p;
  } catch (err) {
    /* Corrupt or unavailable storage — fall back to auto. */
  }
  return null;
}

function cachedConfig() {
  try {
    const data = JSON.parse(localStorage.getItem(CONFIG_KEY) || 'null');
    if (
      data &&
      detectProviders().includes(data.provider) &&
      Number.isInteger(data.threads) &&
      data.threads >= 1 &&
      data.hw === (navigator.hardwareConcurrency || 0)
    ) {
      return { provider: data.provider, threads: data.threads };
    }
  } catch (err) {
    /* Corrupt or unavailable storage — the worker will just re-benchmark. */
  }
  return null;
}

function initWorker() {
  worker = new Worker('/js/inference-worker.js', { type: 'module' });
  worker.onmessage = (e) => handleWorker(e.data);
  worker.onerror = (e) => {
    elStatus.textContent = 'worker error: ' + e.message;
    console.error(e);
  };
  worker.postMessage({
    type: 'init',
    providers: detectProviders(),
    threads: threadsOverride,
    providerOverride,
    cachedConfig:
      threadsOverride == null && providerOverride == null ? cachedConfig() : null,
    forceBench: new URLSearchParams(location.search).has('bench'),
  });
}

// Manual thread-count selection: switch by spinning up a fresh worker (a new
// WASM module) because onnxruntime-web bakes numThreads in at first init.
function applyThreads(value) {
  if (value === 'auto') {
    threadsOverride = null;
    try {
      localStorage.removeItem(OVERRIDE_KEY);
    } catch (err) {
      /* non-fatal */
    }
  } else {
    threadsOverride = Number(value);
    // A thread count only means something for WASM, so it wins over any
    // provider override.
    providerOverride = null;
    try {
      localStorage.setItem(OVERRIDE_KEY, String(threadsOverride));
      localStorage.removeItem(PROVIDER_KEY);
    } catch (err) {
      /* non-fatal */
    }
  }
  restartWorker();
}

// Manual provider selection: restart with the chosen execution provider. Picking
// a GPU provider drops any WASM thread override (which only applies to `wasm`).
function applyProvider(value) {
  if (value === 'auto') {
    providerOverride = null;
    try {
      localStorage.removeItem(PROVIDER_KEY);
    } catch (err) {
      /* non-fatal */
    }
  } else {
    providerOverride = value;
    if (value !== 'wasm') threadsOverride = null;
    try {
      localStorage.setItem(PROVIDER_KEY, value);
      if (value !== 'wasm') localStorage.removeItem(OVERRIDE_KEY);
    } catch (err) {
      /* non-fatal */
    }
  }
  restartWorker();
}

function restartWorker() {
  ready = false;
  window.__dbg.ready = false;
  if (worker) worker.terminate();
  worker = null;
  elThreads.classList.remove('on');
  elThreads.disabled = true;
  elProvider.classList.remove('on');
  elProvider.disabled = true;
  elStatus.textContent = 'restarting&hellip;';
  initWorker();
}

// ---------------------------------------------------------------------------
// Compute-benchmark notification
// ---------------------------------------------------------------------------
let toastTimer = 0;

function providerLabel(provider, threads) {
  return provider === 'wasm' ? `wasm · ${threads}t` : provider;
}

function showBenchToast() {
  elToast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideBenchToast, 10000);
}

function hideBenchToast() {
  clearTimeout(toastTimer);
  elToast.hidden = true;
}

// Add/update the live timing row for one benchmarked candidate.
function addBenchRow(provider, threads, ms) {
  const key = `${provider}:${threads}`;
  let row = elToastRows.querySelector(`[data-key="${key}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'bench-row';
    row.dataset.key = key;
    const name = document.createElement('span');
    name.className = 'bench-name';
    name.textContent = providerLabel(provider, threads);
    const time = document.createElement('span');
    time.className = 'bench-ms';
    row.append(name, time);
    elToastRows.appendChild(row);
  }
  row.querySelector('.bench-ms').textContent = Number.isFinite(ms)
    ? `${ms.toFixed(1)} ms`
    : 'unavailable';
  row.classList.toggle('failed', !Number.isFinite(ms));
  showBenchToast();
}

// Highlight the winning row and show the summary once the worker is ready.
function finishBenchToast(msg) {
  const bench = msg.bench;
  if (!bench || !bench.results) return;
  const chosenKey = `${bench.provider}:${bench.threads}`;
  for (const row of elToastRows.children) {
    row.classList.toggle('selected', row.dataset.key === chosenKey);
  }
  elToastNote.textContent =
    msg.provider === 'wasm'
      ? `Selected wasm · ${bench.threads} thread${bench.threads === 1 ? '' : 's'}`
      : `Selected ${msg.provider}`;
  showBenchToast();
}

function handleWorker(msg) {
  switch (msg.type) {
    case 'ready': {
      ready = true;
      window.__dbg.ready = true;
      initLatentState(msg.dim || DIM);
      elStatus.textContent = 'model ready';
      window.__dbg.bench = msg.bench;
      // The threads dropdown only applies to the WASM provider; show the
      // calibrated provider in its own dropdown.
      const isWasm = msg.provider === 'wasm';
      elThreads.classList.toggle('on', isWasm);
      elThreads.disabled = !isWasm;
      if (isWasm && msg.threads >= 1) elThreads.value = String(msg.threads);
      if (msg.provider) {
        elProvider.value = msg.provider;
        elProvider.classList.add('on');
        elProvider.disabled = false;
      }
      // Persist the auto-calibrated config (never a manual override) so the
      // next load can skip the benchmark entirely.
      if (msg.bench && !msg.bench.manual && msg.bench.results) {
        try {
          localStorage.setItem(
            CONFIG_KEY,
            JSON.stringify({
              provider: msg.bench.provider,
              threads: msg.bench.threads,
              hw: navigator.hardwareConcurrency || 0,
            })
          );
        } catch (err) {
          /* Non-fatal: we just re-benchmark next load. */
        }
      }
      finishBenchToast(msg);
      // Background brightness-direction discovery (doesn't block rendering).
      worker.postMessage({ type: 'brightness', samples: 48 });
      break;
    }
    case 'status':
      elStatus.innerHTML = msg.text;
      break;
    case 'bench-result':
      addBenchRow(msg.provider, msg.threads, msg.ms);
      break;
    case 'result':
      renderResult(msg.bytes, msg.ms);
      // Ack the frame so the worker starts the next inference only after this
      // one was actually displayed (bounds the worker->main result queue).
      worker.postMessage({ type: 'render-done' });
      break;
    case 'brightness':
      brightnessDir = msg.dir;
      elStatus.textContent = `brightness ready (${msg.samples} samples)`;
      break;
  }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
const off = document.createElement('canvas');
off.width = W;
off.height = H;
const offCtx = off.getContext('2d');
const img = offCtx.createImageData(W, H);

// Low-res blurred copy of the frame, pre-rendered once per frame when mirror
// mode 2 is active. Blurring with ctx.filter is expensive on the main thread
// (especially Safari), so it is computed once at a fraction of the resolution;
// every tile just blits this small canvas scaled up, which looks identical for
// a soft background and is a fraction of the cost.
const blurTile = document.createElement('canvas');
const blurTileCtx = blurTile.getContext('2d');

function renderResult(bytes, ms) {
  img.data.set(bytes);
  offCtx.putImageData(img, 0, 0);

  const cw = canvas.width;
  const ch = canvas.height;
  const scale = Math.min(cw / W, ch / H);
  const dw = W * scale;
  const dh = H * scale;
  const y = (ch - dh) / 2;
  const x = (cw - dw) / 2;

  ctx.fillStyle = '#0a0b0d';
  ctx.fillRect(0, 0, cw, ch);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, x, y, dw, dh);

  // Mirror mode: tile reflected copies of the image out past the fitted frame
  // so a window that is wider (landscape) or taller (portrait) than the image
  // is fully covered instead of showing letterbox bars. Tiles alternate
  // orientation so the seams mirror seamlessly. In mode 2 the tiled panels are
  // additionally blurred. (With `scale = min(...)` only one axis ever has
  // spare room, so no corners need filling.)
  if (mirrorMode > 0) {
    const blurred = mirrorMode === 2;
    let tileSource = off;
    if (blurred) {
      const blurPx = Math.max(6, Math.round(dw / 40));
      const bw = Math.max(64, Math.round(dw / 4));
      if (blurTile.width !== bw) {
        blurTile.width = bw;
        blurTile.height = bw;
      }
      blurTileCtx.clearRect(0, 0, bw, bw);
      blurTileCtx.filter = `blur(${Math.max(2, Math.round((blurPx * bw) / dw))}px)`;
      blurTileCtx.drawImage(off, 0, 0, bw, bw);
      blurTileCtx.filter = 'none';
      tileSource = blurTile;
    }
    const drawTile = (tx, ty, flipX, flipY) => {
      ctx.save();
      ctx.translate(tx + (flipX ? dw : 0), ty + (flipY ? dh : 0));
      if (flipX) ctx.scale(-1, 1);
      if (flipY) ctx.scale(1, -1);
      ctx.imageSmoothingQuality = 'low';
      ctx.drawImage(tileSource, 0, 0, dw, dh);
      ctx.fillStyle = blurred ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.0)';
      ctx.fillRect(0, 0, dw, dh);
      ctx.restore();
    };
    if (cw > dw) {
      let mirrored = true;
      for (let tx = x - dw; tx + dw > 0; tx -= dw) {
        drawTile(tx, y, mirrored, false);
        mirrored = !mirrored;
      }
      mirrored = true;
      for (let tx = x + dw; tx < cw; tx += dw) {
        drawTile(tx, y, mirrored, false);
        mirrored = !mirrored;
      }
    }
    if (ch > dh) {
      let mirrored = true;
      for (let ty = y - dh; ty + dh > 0; ty -= dh) {
        drawTile(x, ty, false, mirrored);
        mirrored = !mirrored;
      }
      mirrored = true;
      for (let ty = y + dh; ty < ch; ty += dh) {
        drawTile(x, ty, false, mirrored);
        mirrored = !mirrored;
      }
    }
  }

  // FPS + inference latency meters.
  fpsFrames++;
  inferSum += ms;
  inferCount++;
  window.__dbg.results++;
}

// ---------------------------------------------------------------------------
// Per-frame latent computation (port of main.py `update_frame`, StyleGAN path)
// ---------------------------------------------------------------------------
function computeLatent(dt, nowSec) {
  // Raw spectrum: mic, demo, or silence.
  if (audio.active) {
    spectrum.set(audio.spectrum);
  } else if (demo) {
    genDemoSpectrum(nowSec, spectrum);
  } else {
    spectrum.fill(0);
  }

  // --- A-weighting (perceptual loudness, same curve as the panel display) ---
  // The raw magnitude spectrum is the panel; the latent/LSD path uses this
  // A-weighted version so levels follow human loudness perception. Gain is
  // cached per sample rate; bin i's frequency is i * sr / 512 (fftSize).
  const sr = audio.lastSampleRate || 48000;
  if (!aWeightGain || aWeightSR !== sr) {
    if (!aWeightGain) aWeightGain = new Float32Array(BINS);
    for (let i = 0; i < BINS; i++) {
      aWeightGain[i] = Math.pow(10, aWeightDb((i * sr) / 512) / 20);
    }
    aWeightSR = sr;
  }

  // --- Smoothing ---
  const smoothingFactor = get('Smoothing Factor');
  const smoothing = 1 - Math.exp((-dt * 10) / Math.max(smoothingFactor, 1e-6));
  for (let i = 0; i < BINS; i++) {
    smoothed[i] = smoothing * smoothed[i] + (1 - smoothing) * spectrum[i] * aWeightGain[i];
  }

  // --- Randomize Latent Vector (swap two lookup entries periodically) ---
  // Time-based so the cadence is identical at any frame rate (previously this
  // counted frames, which made it FPS-dependent).
  const ar = get('Audio Randomization');
  if (ar !== 0) {
    audioRandTimer += dt;
    if (audioRandTimer >= 0.5 / ar) {
      audioRandTimer = 0;
      const c = (Math.random() * DIM) | 0;
      const d = (Math.random() * DIM) | 0;
      const t = lookup[c];
      lookup[c] = lookup[d];
      lookup[d] = t;
    }
  }

  // --- Flux / low-pass measures ---
  // Brightness, pulse and motion each respond through a Gaussian band filter in
  // log-frequency space (draggable on the spectrum for brightness): they react
  // only around their center frequency, decaying smoothly to both sides.
  const bfFreq = Math.max(get('Brightness Freq'), 1);
  const bfWidth = Math.max(get('Brightness Width'), 0.05);
  const bfNorm = 1 / (2 * bfWidth * bfWidth);
  const pfFreq = Math.max(get('Pulse Freq'), 1);
  const pfWidth = Math.max(get('Pulse Width'), 0.05);
  const pfNorm = 1 / (2 * pfWidth * pfWidth);
  const mfFreq = Math.max(get('Motion Freq'), 1);
  const mfWidth = Math.max(get('Motion Width'), 0.05);
  const mfNorm = 1 / (2 * mfWidth * mfWidth);

  let pulseAmp = 0;
  let lowPassBright = 0;
  for (let i = 1; i < BINS; i++) {
    const d = Math.log2((i * sr) / (512 * bfFreq));
    const v = smoothed[i] * Math.exp(-d * d * bfNorm);
    if (v > lowPassBright) lowPassBright = v;
    const dp = Math.log2((i * sr) / (512 * pfFreq));
    const vp = smoothed[i] * Math.exp(-dp * dp * pfNorm);
    if (vp > pulseAmp) pulseAmp = vp;
  }

  const invDt = 1 / Math.max(dt, 1e-6);
  let lowPassDrift = 0;
  let motionAmp = 0;
  for (let i = 0; i < BINS; i++) {
    const f = (spectrum[i] - prevSpectrum[i]) * invDt;
    if (f > lowPassDrift) lowPassDrift = f;
    const dm = Math.log2((i * sr) / (512 * mfFreq));
    const vm = f * Math.exp(-dm * dm * mfNorm);
    if (vm > motionAmp) motionAmp = vm;
  }
  prevSpectrum.set(spectrum);
  lowPassDrift = Math.max(lowPassDrift, 0);
  motionAmp = Math.max(motionAmp, 0);

  const lowPass = Math.pow(motionAmp, get('Motion Power')) * 0.0001;
  lastHueFlux = lowPassDrift * 0.001; // linear flux for hue shift

  // --- Latent composition (StyleGAN branch) ---
  const cutoff = Math.round(get('Cutoff'));
  const audioWeight = get('Audio Weight');
  const noiseWeight = get('Noise Weight');

  for (let i = 0; i < DIM; i++) {
    const v = i < cutoff ? 0 : smoothed[lookup[i]] * audioWeight;
    audioNoise[i] = v;
    z[i] = a[i] * noiseWeight + v;
  }

  // --- LSD latent modulation ---
  const lz = lsd.step({
    pulseAmp: Math.max(pulseAmp, 0),
    motionAmp: lowPass,
    music: audioNoise,
    pulseMode: get('Pulse Mode'),
    pulseReact: get('Pulse React'),
    pulsePower: get('Pulse Power'),
    brightnessReact: get('Brightness React'),
    brightnessDir,
    brightnessAmp: Math.max(lowPassBright, 0),
    motionReact: get('Motion React'),
    motionRandomness: get('Motion Randomness'),
    truncation: get('Truncation'),
    fps: 1 / Math.max(dt, 1e-6),
    pulseSmooth: get('Pulse Smooth'),
    motionSmooth: get('Motion Smooth'),
  });
  thisHue = Math.round(lastHueFlux * get('Hue Shift'));
  return lz;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let lastT = performance.now() / 1000;
let fpsFrames = 0;
let fpsWindow = performance.now();
let inferSum = 0;
let inferCount = 0;
let thisHue = 0;

function loop(now) {
  const nowSec = now / 1000;
  const dt = Math.min(Math.max(nowSec - lastT, 1e-4), 0.1);
  lastT = nowSec;

  if (ready && lsd) {
    const z = computeLatent(dt, nowSec);
    // Latest-wins: post every frame; the worker drops stale work.
    worker.postMessage({
      type: 'z',
      z,
      tanh: get('Tanh Output') !== 0,
      hue: thisHue,
    });
    window.__dbg.zPosted++;
  }

  if (window.__dbg && (window.__dbg.zPosted % 30) === 0) {
    let mx = 0;
    for (let i = 0; i < BINS; i++) if (spectrum[i] > mx) mx = spectrum[i];
    window.__dbg.spectrumMax = mx;
    window.__dbg.audioActive = audio.active;
    window.__dbg.demo = demo;
    window.__dbg.workletMsgs = audio.msgCount;
  }

  // Panel spectrum display: raw spectrum, view applies its own smoothing.
  // The filter points (draggable) visualize the Brightness/Pulse/Motion bands;
  // their vertical position and bell height mirror each source's react value.
  spectrumView.update(
    spectrum,
    audio.lastSampleRate || 48000,
    dt,
    {
      pulse: filterInfo('Pulse Freq', 'Pulse Width', 'Pulse React'),
      brightness: filterInfo('Brightness Freq', 'Brightness Width', 'Brightness React'),
      motion: filterInfo('Motion Freq', 'Motion Width', 'Motion React'),
    }
  );

  // Meters once per half second.
  if (now - fpsWindow >= 500) {
    elFps.textContent = String(Math.round(fpsFrames / ((now - fpsWindow) / 1000)));
    elFps.classList.toggle('on', fpsFrames > 0);
    if (inferCount) {
      elInfer.textContent = `${(inferSum / inferCount).toFixed(1)}ms`;
      inferSum = 0;
      inferCount = 0;
    }
    fpsFrames = 0;
    fpsWindow = now;
  }

  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// UI setup
// ---------------------------------------------------------------------------
function setStatus(text, cls) {
  elStatus.textContent = text;
  if (cls) elStatus.classList.add(cls);
}

function buildSliders() {
  const container = document.getElementById('sliders');
  for (const group of GROUP_ORDER) {
    const visible = Object.values(SETTINGS).filter(
      (def) => !def.hidden && def.group === group
    );
    if (!visible.length) continue;

    const section = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'group-title';
    title.textContent = group;
    section.appendChild(title);

    for (const [name, def] of Object.entries(SETTINGS)) {
      if (def.hidden) continue;
      if (def.group !== group) continue;

      const row = document.createElement('div');
      row.className = 'slider';

      const label = document.createElement('div');
      label.className = 'label';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'name';
      nameSpan.textContent = name;
      const valSpan = document.createElement('span');
      valSpan.className = 'val';
      valSpan.textContent = def.dec ? def.value.toFixed(def.dec) : def.value;
      label.append(nameSpan, valSpan);

      const input = document.createElement('input');
      input.type = 'range';
      input.min = def.min;
      input.max = def.max;
      input.step = def.step;
      input.value = def.value;
      const setVal = () => {
        valSpan.textContent = def.dec ? def.value.toFixed(def.dec) : def.value;
      };
      // Keep UI references so external changes (e.g. dragging the filter
      // point on the spectrum) can sync the slider position and label.
      def.uiInput = input;
      def.uiVal = setVal;
      input.addEventListener('input', () => {
        def.value = parseFloat(input.value);
        setVal();
      });

      row.append(label, input);
      section.appendChild(row);
    }
    container.appendChild(section);
  }
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);

// Push an externally-changed setting into its slider UI (if visible).
function syncSettingUI(name) {
  const def = SETTINGS[name];
  if (!def || !def.uiInput) return;
  def.uiInput.value = def.value;
  def.uiVal();
}

function setupUI() {
  buildSliders();
  resize();

  // Dragging a filter point on the spectrum retunes that source's band (the
  // horizontal axis = center frequency, the vertical axis = react amount); the
  // mouse wheel changes its width. null args leave that axis as-is.
  const FILTER_KEYS = {
    pulse: { freq: 'Pulse Freq', width: 'Pulse Width', react: 'Pulse React' },
    brightness: { freq: 'Brightness Freq', width: 'Brightness Width', react: 'Brightness React' },
    motion: { freq: 'Motion Freq', width: 'Motion Width', react: 'Motion React' },
  };
  spectrumView.onFilterChange = (id, freq, width, react) => {
    const keys = FILTER_KEYS[id] || FILTER_KEYS.brightness;
    if (freq != null) {
      SETTINGS[keys.freq].value = Math.min(
        Math.max(Math.round(freq), SETTINGS[keys.freq].min),
        SETTINGS[keys.freq].max
      );
      syncSettingUI(keys.freq);
    }
    if (width != null) {
      SETTINGS[keys.width].value = Math.min(
        Math.max(width, SETTINGS[keys.width].min),
        SETTINGS[keys.width].max
      );
      syncSettingUI(keys.width);
    }
    if (react != null) {
      SETTINGS[keys.react].value = Math.min(
        Math.max(react, SETTINGS[keys.react].min),
        SETTINGS[keys.react].max
      );
      syncSettingUI(keys.react);
    }
  };

  // Populate the thread dropdown: Auto (returns to the benchmarked best) plus
  // bare-number options. The selected option always shows just the current
  // thread count, even while Auto is in effect.
  for (const t of threadOptions()) {
    const opt = document.createElement('option');
    opt.value = String(t);
    opt.textContent = String(t);
    elThreads.add(opt);
  }
  elThreads.disabled = true;
  elThreads.addEventListener('change', () => applyThreads(elThreads.value));

  // Populate the provider dropdown with every execution provider this browser
  // exposes. Auto uses the benchmark winner; picking one forces it.
  for (const p of detectProviders()) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p;
    elProvider.add(opt);
  }
  if (providerOverride) elProvider.value = providerOverride;
  elProvider.disabled = true;
  elProvider.addEventListener('change', () => applyProvider(elProvider.value));

  elToast.addEventListener('click', hideBenchToast);

  document.getElementById('panel-toggle').addEventListener('click', () => {
    document.getElementById('panel').classList.add('collapsed');
  });
  document.getElementById('panel-open').addEventListener('click', () => {
    document.getElementById('panel').classList.remove('collapsed');
  });

  const btnMic = document.getElementById('btn-mic');
  btnMic.addEventListener('click', async () => {
    if (audio.active) {
      await audio.stop();
      btnMic.querySelector('.btn-label').textContent = 'Enable Microphone';
      btnMic.classList.remove('active');
      elAudio.textContent = 'off';
      elAudio.classList.remove('on');
    } else {
      try {
        await audio.start();
        btnMic.querySelector('.btn-label').textContent = 'Microphone: ON';
        btnMic.classList.add('active');
        elAudio.textContent = 'on';
        elAudio.classList.add('on');
      } catch (err) {
        setStatus('microphone unavailable');
      }
    }
  });

  const btnDemo = document.getElementById('btn-demo');
  btnDemo.addEventListener('click', () => {
    demo = !demo;
    btnDemo.classList.toggle('active', demo);
    if (demo && !audio.active) {
      elAudio.textContent = 'demo';
      elAudio.classList.add('on');
    } else {
      elAudio.textContent = audio.active ? 'on' : 'off';
      if (!audio.active) elAudio.classList.remove('on');
    }
  });

  document.getElementById('btn-fullscreen').addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  });

  const btnMirror = document.getElementById('btn-mirror');
  const mirrorLabel = btnMirror.querySelector('span');
  const MIRROR_LABELS = ['Mirror', 'Mirror', 'Mirror + Blur'];
  const updateMirrorUI = () => {
    btnMirror.classList.toggle('active', mirrorMode !== 0);
    mirrorLabel.textContent = MIRROR_LABELS[mirrorMode];
  };
  btnMirror.addEventListener('click', () => {
    mirrorMode = (mirrorMode + 1) % 3;
    updateMirrorUI();
  });
  updateMirrorUI();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
window.__dbg = { ready: false, zPosted: 0, results: 0 };
setupUI();
initWorker();
requestAnimationFrame(loop);
