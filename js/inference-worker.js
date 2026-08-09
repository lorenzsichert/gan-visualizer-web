/**
 * Inference worker: owns the onnxruntime-web WASM session.
 *
 * The main thread posts `{ type: 'z', z, tanh, hue }` messages (one per
 * animation frame) and this worker runs them "latest-wins": if a newer latent
 * arrives while one is being inferred, it is picked up as soon as the current
 * run finishes. The final image is converted to RGBA bytes and hue-shifted
 * here (the heaviest per-pixel loops), then transferred back so the main
 * thread only has to blit it to a canvas.
 *
 * Execution runs on multi-threaded **WASM** (single-threaded when
 * SharedArrayBuffer is unavailable).
 *
 * Before the live session is created, the thread count is chosen by a short
 * benchmark: one throwaway worker per candidate thread count measures median
 * inference latency, and the FEWEST threads within tolerance of the fastest are
 * used (sparing cores for the render loop and audio). The chosen thread count
 * is cached by the main thread (localStorage) so the calibration only runs once
 * per machine/model.
 *
 * Brightness-direction discovery (port of main.py `_discover_brightness_*`)
 * runs here too so the UI never blocks. All `session.run` calls are serialized
 * through a promise lock so discovery and the live render loop never race.
 */
import * as ort from '/lib/ort-wasm/ort.wasm.min.mjs';

const INPUT = 'var';
const OUTPUT = 'img';
const HAS_SAB = typeof SharedArrayBuffer !== 'undefined';

ort.env.wasm.wasmPaths = '/lib/ort-wasm/';
ort.env.logLevel = 'warning';
// Default thread pool (only read when the WASM module first initializes); the
// real value is set after the benchmark below picks the best thread count.
ort.env.wasm.numThreads = defaultThreads();

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

// Serializes all session.run calls (single inference at a time).
let lock = Promise.resolve();
function withLock(fn) {
  const run = lock.then(fn, fn);
  lock = run.then(() => {}, () => {});
  return run;
}

// Brightness discovery runs on its OWN session and lock so its (potentially
// many) inferences never queue behind — and never delay — live rendering.
let modelUrl = '/models/EndToEndNetwork.onnx';
let discoverySession = null;
let discoveryLock = Promise.resolve();
function withDiscoveryLock(fn) {
  const run = discoveryLock.then(fn, fn);
  discoveryLock = run.then(() => {}, () => {});
  return run;
}

async function getDiscoverySession() {
  if (discoverySession) return discoverySession;
  discoverySession = await withDiscoveryLock(() =>
    ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  );
  return discoverySession;
}

async function brightnessOfOn(sess, z) {
  const result = await withDiscoveryLock(() =>
    sess.run({ [INPUT]: new ort.Tensor('float32', z, [1, z.length]) })
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
    discoverBrightness(msg.samples || 48);
  } else if (msg.type === 'render-done') {
    if (ackWaiter) {
      const resolve = ackWaiter;
      ackWaiter = null;
      resolve();
    }
  }
};

function readDim(session) {
  const shape = session.inputMetadata ? session.inputMetadata[INPUT]?.shape : null;
  if (shape && shape.length > 1 && Number.isFinite(shape[1]) && shape[1] > 0) {
    return Number(shape[1]);
  }
  return 512;
}

// ---------------------------------------------------------------------------
// Thread-count benchmarking
// ---------------------------------------------------------------------------
// onnxruntime-web bakes `numThreads` into the WASM module at first init; the
// pool can't be resized later in the same realm. So before the live session is
// created we spawn one throwaway worker per candidate thread count (each a
// fresh module, js/bench-worker.js) and time a warmed run. We then pick the
// FEWEST threads whose median latency is within BENCH_TOLERANCE of the fastest:
// the cores those extra threads would hog are better spent on the render loop
// and audio worklet, and a few percent is imperceptible in the animation.
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

function chooseThreads(results) {
  const valid = results
    .filter((r) => Number.isFinite(r.ms))
    .sort((a, b) => a.threads - b.threads);
  if (!valid.length) return defaultThreads();
  let best = Infinity;
  for (const r of valid) if (r.ms < best) best = r.ms;
  for (const r of valid) if (r.ms <= best * BENCH_TOLERANCE) return r.threads;
  return valid[0].threads;
}

function runBenchWorker(url, threads) {
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
      threads,
      runs: BENCH_RUNS,
      warmup: BENCH_WARMUP,
    });
  });
}

async function benchmarkThreads(url, cachedThreads = null, force = false, override = null) {
  if (Number.isInteger(override) && override >= 1) {
    lastBench = { threads: override, results: null, manual: true };
    return override;
  }
  if (!force && Number.isInteger(cachedThreads) && cachedThreads >= 1) {
    lastBench = { threads: cachedThreads, results: null };
    return cachedThreads;
  }

  const candidates = candidateThreads();
  const results = [];
  postMessage({ type: 'status', text: 'Checking ONNX thread performance&hellip;' });
  for (const threads of candidates) {
    postMessage({
      type: 'status',
      text: `Benchmarking ${threads} thread${threads === 1 ? '' : 's'}&hellip;`,
    });
    results.push({ threads, ms: await runBenchWorker(url, threads) });
  }

  const chosen = chooseThreads(results);
  lastBench = { threads: chosen, results };
  return chosen;
}

async function init(msg) {
  const url = msg.url || '/models/EndToEndNetwork.onnx';
  modelUrl = url;
  const threads = await benchmarkThreads(url, msg.cachedThreads, !!msg.forceBench, msg.threads);
  ort.env.wasm.numThreads = threads;
  postMessage({ type: 'status', text: 'Loading model (14 MB)&hellip;' });
  session = await withLock(() =>
    ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  );
  dim = readDim(session);
  postMessage({
    type: 'ready',
    dim,
    threads: ort.env.wasm.numThreads,
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
      session.run({ [INPUT]: new ort.Tensor('float32', z, [1, dim]) })
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
const rowTmp = new Uint8Array(512);
const colTmp = new Uint8Array(512);

function hueShiftBytes(bytes, shift, WW, HH) {
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

async function discoverBrightness(n = 48) {
  if (!session) return;
  postMessage({ type: 'status', text: 'Discovering brightness direction&hellip;' });
  const dsession = await getDiscoverySession();

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
    b[i] = await brightnessOfOn(dsession, Z.subarray(i * dim, (i + 1) * dim));
    if ((i + 1) % 8 === 0) {
      postMessage({ type: 'status', text: `Brightness ${i + 1}/${n}&hellip;` });
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
    const pos = await brightnessOfOn(dsession, d);
    const negArr = new Float32Array(dim);
    for (let j = 0; j < dim; j++) negArr[j] = -d[j];
    const neg = await brightnessOfOn(dsession, negArr);
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
  postMessage({ type: 'status', text: `Brightness ready (${n} samples)` });
}
