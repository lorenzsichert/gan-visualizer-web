/**
 * Inference worker: owns the onnxruntime-web session.
 *
 * The main thread posts `{ type: 'z', z, tanh, hue }` messages (one per
 * animation frame) and this worker runs them "latest-wins": if a newer latent
 * arrives while one is being inferred, it is picked up as soon as the current
 * run finishes. The final image is converted to RGBA bytes and hue-shifted
 * here (the heaviest per-pixel loops), then transferred back so the main
 * thread only has to blit it to a canvas.
 *
 * Before the live session is created, this worker CALIBRATES the compute
 * backend: it benchmarks every available execution provider (`webgpu`, `webnn`,
 * `wasm`, …) and, for WASM, every candidate thread count, then picks the
 * fastest. Each candidate runs in a throwaway worker (js/bench-worker.js) so it
 * gets a fresh module/realm. The chosen configuration is reported to the main
 * thread, which caches it in localStorage, so the calibration only runs once
 * per machine/model.
 *
 * The onnxruntime-web build is chosen per provider: the small WASM-only module
 * for `wasm`, and the full module for GPU providers. This keeps the large JSEP
 * WASM binary off the wire for machines that stay on CPU.
 *
 * Brightness-direction discovery (port of main.py `_discover_brightness_*`)
 * runs here too so the UI never blocks. All `session.run` calls are serialized
 * through a promise lock so discovery and the live render loop never race.
 */
import { fetchModelBytes } from './model-cache.js';

const INPUT = 'var';
const OUTPUT = 'img';
const HAS_SAB = typeof SharedArrayBuffer !== 'undefined';

// The active onnxruntime-web module (loaded lazily per provider) and the
// provider it belongs to.
let ort = null;
let activeProvider = 'wasm';

async function loadOrt(provider) {
  if (ort && activeProvider === provider) return ort;
  ort = await import(
    provider === 'wasm'
      ? '/lib/ort-wasm/ort.wasm.min.mjs'
      : '/lib/ort-wasm/ort.all.min.mjs'
  );
  ort.env.wasm.wasmPaths = '/lib/ort-wasm/';
  ort.env.logLevel = 'warning';
  // Default thread pool (only read when the WASM module first initializes);
  // the calibrated value is applied by init() below.
  ort.env.wasm.numThreads = defaultThreads();
  activeProvider = provider;
  return ort;
}

let session = null;
let dim = 512;
let latestZ = null;
let latestTanh = false;
let latestHue = 0;
let pumping = false;

// Ack-throttling: the main thread sends `render-done` after blitting a result.
// The worker holds its next inference until that ack arrives, which bounds the
// worker->main result queue to ONE in-flight image. Without this, a fast worker
// (multi-threaded WASM) would flood a slow main thread's message queue with
// results and the display age would grow without bound as FPS drops.
let ackWaiter = null;

// Serializes ALL onnxruntime calls (session creation + session.run) into a
// single in-flight operation. This is required, not just an optimization: the
// JSEP (WebGPU/WebNN) WASM glue keeps one global "current call" slot and throws
// "Session already started" / "Session mismatch" if two run calls — even on
// different sessions — overlap. Discovery and live rendering therefore share
// this lock and the same session; individual inferences still interleave at run
// granularity, so the render loop keeps progressing during discovery.
let lock = Promise.resolve();
function withLock(fn) {
  const run = lock.then(fn, fn);
  lock = run.then(() => {}, () => {});
  return run;
}

async function brightnessOfOn(z) {
  const result = await withLock(() =>
    session.run({
      [INPUT]: new ort.Tensor('float32', z, [1, z.length]),
    })
  );
  const d = result[OUTPUT].data;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i];
  return s / d.length;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    init(msg);
  } else if (msg.type === 'z') {
    latestZ = msg.z;
    latestTanh = !!msg.tanh;
    latestHue = msg.hue || 0;
    if (session && !pumping) pump();
  } else if (msg.type === 'brightness') {
    discoverBrightness(msg.samples || 256);
  } else if (msg.type === 'render-done') {
    if (ackWaiter) {
      const resolve = ackWaiter;
      ackWaiter = null;
      resolve();
    }
  }
};

function readDim(session) {
  const meta = session.inputMetadata;
  const entry = Array.isArray(meta) ? meta.find((m) => m.name === INPUT) : meta?.[INPUT];
  const shape = entry && entry.shape;
  if (shape && shape.length > 1 && Number.isFinite(shape[1]) && shape[1] > 0) {
    return Number(shape[1]);
  }
  return 512;
}

// ---------------------------------------------------------------------------
// Compute-provider benchmarking
// ---------------------------------------------------------------------------
// onnxruntime-web bakes `numThreads` into the WASM module at first init (the
// pool can't be resized later in the same realm), and a session's execution
// providers are fixed at creation. So before the live session is created we
// spawn one throwaway worker per candidate configuration (each a fresh module,
// js/bench-worker.js) and time a warmed run.
//
// Candidates are `{ provider, threads }`:
//   - every provider the main thread detected (`webgpu`, `webnn`, `wasm`), once;
//   - for WASM, every candidate thread count.
//
// Across providers we pick the fastest. Within WASM we pick the FEWEST threads
// whose median latency is within BENCH_TOLERANCE of the fastest WASM run: the
// cores those extra threads would hog are better spent on the render loop and
// audio worklet, and a few percent is imperceptible in the animation.
const BENCH_RUNS = 8;
const BENCH_WARMUP = 2;
const BENCH_TOLERANCE = 1.05;
const BENCH_TIMEOUT_MS = 60000;

let lastBench = null;

function defaultThreads() {
  return HAS_SAB
    ? Math.min(4, Math.max(2, navigator.hardwareConcurrency || 4))
    : 1;
}

function candidateThreads() {
  if (!HAS_SAB) return [1];
  const hc = navigator.hardwareConcurrency || 4;
  const set = new Set([1, hc]);
  for (let t = 2; t <= hc; t *= 2) set.add(t);
  return [...set].sort((a, b) => a - b);
}

function chooseConfig(results) {
  const valid = results.filter((r) => Number.isFinite(r.ms));
  if (!valid.length) return { provider: 'wasm', threads: defaultThreads() };
  let best = valid[0];
  for (const r of valid) if (r.ms < best.ms) best = r;
  if (best.provider === 'wasm') {
    const wasm = valid
      .filter((r) => r.provider === 'wasm')
      .sort((a, b) => a.threads - b.threads);
    const fastest = Math.min(...wasm.map((r) => r.ms));
    const pick = wasm.find((r) => r.ms <= fastest * BENCH_TOLERANCE);
    return { provider: 'wasm', threads: pick ? pick.threads : best.threads };
  }
  return { provider: best.provider, threads: 1 };
}

function runBenchWorker(url, provider, threads) {
  return new Promise((resolve) => {
    let w = null;
    let timer = 0;
    const done = (ms) => {
      if (w) w.terminate();
      clearTimeout(timer);
      resolve(ms);
    };
    timer = setTimeout(() => done(Infinity), BENCH_TIMEOUT_MS);
    try {
      w = new Worker('/js/bench-worker.js', { type: 'module' });
    } catch (err) {
      done(Infinity);
      return;
    }
    w.onmessage = (e) => {
      if (e.data && e.data.type === 'result') done(e.data.ms);
    };
    w.onerror = () => done(Infinity);
    w.postMessage({
      type: 'bench',
      url,
      provider,
      threads,
      runs: BENCH_RUNS,
      warmup: BENCH_WARMUP,
    });
  });
}

async function benchmarkProviders(url, providers) {
  const results = [];
  const total = providers.reduce(
    (n, p) => n + (p === 'wasm' ? candidateThreads().length : 1),
    0
  );
  let done = 0;
  postMessage({
    type: 'status',
    phase: 'bench',
    text: 'Benchmarking compute providers&hellip;',
    progress: 0,
  });
  for (const provider of providers) {
    const candidates = provider === 'wasm' ? candidateThreads() : [1];
    for (const threads of candidates) {
      const label =
        provider === 'wasm'
          ? `wasm (${threads} thread${threads === 1 ? '' : 's'})`
          : provider;
      postMessage({
        type: 'status',
        phase: 'bench',
        text: `Benchmarking ${label}&hellip;`,
        progress: total ? done / total : 0,
      });
      const ms = await runBenchWorker(url, provider, threads);
      const result = { provider, threads, ms };
      results.push(result);
      done += 1;
      postMessage({ type: 'bench-result', ...result });
    }
  }
  const chosen = chooseConfig(results);
  lastBench = { provider: chosen.provider, threads: chosen.threads, results };
  return chosen;
}

function isValidConfig(config, providers) {
  return (
    config &&
    typeof config.provider === 'string' &&
    providers.includes(config.provider) &&
    Number.isInteger(config.threads) &&
    config.threads >= 1
  );
}

async function init(msg) {
  const url = msg.url;
  if (!url) {
    postMessage({ type: 'status', phase: 'idle', text: 'no model selected' });
    return;
  }
  const label = msg.modelName || 'model';
  const providers =
    Array.isArray(msg.providers) && msg.providers.length ? msg.providers : ['wasm'];

  // Fetch (or read from cache) the model BEFORE benchmarking. The first
  // benchmark candidate would otherwise be what pulls the file, so the user
  // would watch "Benchmarking …" while 100 MB streams in the background. Doing
  // it up front also means every benchmark candidate gets a warm cache hit.
  await fetchModelBytes(url, (loaded, total, cached) => {
    postMessage({
      type: 'status',
      phase: 'download',
      text: cached ? 'Loading model from cache&hellip;' : 'Downloading model&hellip;',
      progress: total ? loaded / total : undefined,
    });
  });

  let config;
  if (typeof msg.providerOverride === 'string' && msg.providerOverride) {
    config = {
      provider: msg.providerOverride,
      threads: Number.isInteger(msg.threads) && msg.threads >= 1 ? msg.threads : defaultThreads(),
    };
    lastBench = { ...config, results: null, manual: true };
  } else if (Number.isInteger(msg.threads) && msg.threads >= 1) {
    config = { provider: 'wasm', threads: msg.threads };
    lastBench = { ...config, results: null, manual: true };
  } else if (!msg.forceBench && isValidConfig(msg.cachedConfig, providers)) {
    config = { provider: msg.cachedConfig.provider, threads: msg.cachedConfig.threads };
    lastBench = { ...config, results: null, cached: true };
  } else {
    config = await benchmarkProviders(url, providers);
  }

  await loadOrt(config.provider);
  if (config.provider === 'wasm') ort.env.wasm.numThreads = config.threads;

  postMessage({ type: 'status', phase: 'load', text: `Loading ${label}&hellip;` });
  try {
    session = await withLock(async () =>
      ort.InferenceSession.create(await fetchModelBytes(url), {
        executionProviders: [config.provider],
        graphOptimizationLevel: 'all',
      })
    );
  } catch (err) {
    console.error(err);
    postMessage({
      type: 'status',
      phase: 'idle',
      text: `Failed to load ${label}: ${err && err.message ? err.message : err}`,
    });
    return;
  }
  dim = readDim(session);
  postMessage({
    type: 'ready',
    dim,
    provider: config.provider,
    threads: config.provider === 'wasm' ? ort.env.wasm.numThreads : 0,
    multithreaded: HAS_SAB,
    bench: lastBench,
  });
}

async function pump() {
  pumping = true;
  while (session && latestZ != null) {
    const z = latestZ;
    const tanhOut = latestTanh;
    const hue = latestHue;
    latestZ = null;

    const t0 = performance.now();
    const out = await withLock(() =>
      session.run({
        [INPUT]: new ort.Tensor('float32', z, [1, dim]),
      })
    );
    const tensor = out[OUTPUT];
    const data = tensor.data;
    const dims = tensor.dims;
    const C = dims[1] || 3;
    const HH = dims[2] || 512;
    const WW = dims[3] || 512;
    const plane = HH * WW;
    // Model outputs NCHW float planes; interleave into RGBA bytes.
    const bytes = new Uint8ClampedArray(plane * 4);
    const scale = (v) => (v + 1) * 127.5;
    if (tanhOut) {
      for (let y = 0; y < HH; y++) {
        const row = y * WW * 4;
        const prow = y * WW;
        for (let x = 0; x < WW; x++) {
          const p = prow + x;
          const o = row + x * 4;
          bytes[o] = (Math.tanh(data[p]) + 1) * 127.5;
          bytes[o + 1] = (Math.tanh(data[p + plane]) + 1) * 127.5;
          bytes[o + 2] = (Math.tanh(data[p + plane * 2]) + 1) * 127.5;
          bytes[o + 3] = 255;
        }
      }
    } else {
      for (let y = 0; y < HH; y++) {
        const row = y * WW * 4;
        const prow = y * WW;
        for (let x = 0; x < WW; x++) {
          const p = prow + x;
          const o = row + x * 4;
          bytes[o] = scale(data[p]);
          bytes[o + 1] = scale(data[p + plane]);
          bytes[o + 2] = scale(data[p + plane * 2]);
          bytes[o + 3] = 255;
        }
      }
    }
    const shift = hue % WW;
    if (shift) hueShiftBytes(bytes, shift, WW, HH);
    postMessage(
      { type: 'result', bytes, dims: [HH, WW], ms: performance.now() - t0 },
      [bytes.buffer]
    );

    // Wait for the main thread to render this frame before inferring the next.
    // This is the pacing signal: at low FPS the worker stalls alongside the
    // display instead of queuing an ever-growing backlog of results.
    await new Promise((resolve) => {
      ackWaiter = resolve;
    });
  }
  pumping = false;
}

/**
 * RGB channel roll = hue rotation, ported from the Python app's hue-shift:
 * red rolls horizontally, green vertically, blue stays. Operates on the RGBA
 * byte buffer (layout-independent), on the worker so the main thread only
 * blits.
 */
let rowTmp = new Uint8Array(512);
let colTmp = new Uint8Array(512);

function hueShiftBytes(bytes, shift, WW, HH) {
  // Grow the scratch rows/cols for model resolutions above 512.
  if (rowTmp.length < WW) rowTmp = new Uint8Array(WW);
  if (colTmp.length < HH) colTmp = new Uint8Array(HH);
  for (let y = 0; y < HH; y++) {
    const row = y * WW * 4;
    for (let x = 0; x < WW; x++) rowTmp[x] = bytes[row + x * 4];
    for (let x = 0; x < WW; x++) {
      const nx = ((x - shift) % WW + WW) % WW;
      bytes[row + x * 4] = rowTmp[nx];
    }
  }
  for (let x = 0; x < WW; x++) {
    for (let y = 0; y < HH; y++) colTmp[y] = bytes[y * WW * 4 + x * 4 + 1];
    for (let y = 0; y < HH; y++) {
      const ny = ((y - shift) % HH + HH) % HH;
      bytes[y * WW * 4 + x * 4 + 1] = colTmp[ny];
    }
  }
}

async function discoverBrightness(n = 256) {
  if (!session) return;
  postMessage({
    type: 'status',
    phase: 'sample',
    text: 'Sampling brightness direction&hellip;',
    progress: 0,
  });

  const Z = new Float32Array(n * dim);
  for (let i = 0; i < Z.length; i++) {
    let v = 0, s = 0;
    do {
      const u = Math.random() * 2 - 1;
      const vv = Math.random() * 2 - 1;
      s = u * u + vv * vv;
      v = u * Math.sqrt((-2 * Math.log(s)) / s);
    } while (s >= 1 || s === 0 || !isFinite(v));
    Z[i] = v;
  }

  const b = new Float32Array(n);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    b[i] = await brightnessOfOn(Z.subarray(i * dim, (i + 1) * dim));
    if ((i + 1) % 8 === 0) {
      postMessage({
        type: 'status',
        phase: 'sample',
        text: `Brightness ${i + 1}/${n}&hellip;`,
        progress: (i + 1) / n,
      });
    }
  }

  // Linear regression: d = (Z - z_mean)^T (b - b_mean), then unit normalize.
  const zMean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < dim; j++) zMean[j] += Z[i * dim + j];
  }
  for (let j = 0; j < dim; j++) zMean[j] /= n;
  let bMean = 0;
  for (let i = 0; i < n; i++) bMean += b[i];
  bMean /= n;

  const d = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    const db = b[i] - bMean;
    const row = Z.subarray(i * dim, (i + 1) * dim);
    for (let j = 0; j < dim; j++) d[j] += (row[j] - zMean[j]) * db;
  }
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += d[j] * d[j];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) for (let j = 0; j < dim; j++) d[j] /= norm;

  // Sign: make +d point toward brighter images.
  if (norm > 1e-12) {
    const pos = await brightnessOfOn(d);
    const negArr = new Float32Array(dim);
    for (let j = 0; j < dim; j++) negArr[j] = -d[j];
    const neg = await brightnessOfOn(negArr);
    if (pos < neg) for (let j = 0; j < dim; j++) d[j] = -d[j];
  } else {
    d.fill(0);
  }

  postMessage({
    type: 'brightness',
    dir: d,
    ms: performance.now() - t0,
    samples: n,
  });
  postMessage({ type: 'status', phase: 'idle', text: `Brightness ready (${n} samples)` });
}
