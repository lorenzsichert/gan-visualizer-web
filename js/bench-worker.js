/**
 * One-shot benchmark worker for a single ONNX compute configuration.
 *
 * onnxruntime-web bakes `env.wasm.numThreads` into the WASM module when it first
 * initializes and cannot resize the thread pool afterward inside the same realm,
 * and a session's execution providers are fixed at creation. So the inference
 * worker spawns one of these per candidate configuration: each instance is a
 * fresh realm with its OWN module, giving every provider/thread count a clean
 * start.
 *
 * A candidate is `{ provider, threads }` where provider is one of the
 * onnxruntime-web execution providers (`wasm`, `webgpu`, `webnn`, `webgl`) and
 * `threads` only matters for `wasm`. The worker loads the matching runtime:
 *
 *   - `wasm`  -> the small WASM-only build (lib/ort-wasm/ort.wasm.min.mjs)
 *   - others  -> the full build (lib/ort-wasm/ort.all.min.mjs), which pulls in
 *                the JSEP WASM binary that backs WebGPU/WebNN/WebGL.
 *
 * It loads the model with the same options the live session uses, warms up, then
 * times `runs` inferences and reports the MEDIAN (which survives GC spikes).
 * The parent picks the fastest configuration. This worker terminates itself
 * after posting the result.
 */
function moduleUrl(provider) {
  return provider === 'wasm'
    ? '/lib/ort-wasm/ort.wasm.min.mjs'
    : '/lib/ort-wasm/ort.all.min.mjs';
}

function readDim(session) {
  const meta = session.inputMetadata;
  const entry = Array.isArray(meta) ? meta.find((m) => m.name === 'var') : meta?.var;
  const shape = entry && entry.shape;
  if (shape && shape.length > 1 && Number.isFinite(shape[1]) && shape[1] > 0) {
    return Number(shape[1]);
  }
  return 512;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== 'bench') return;

  const provider = msg.provider || 'wasm';
  const threads = Math.max(1, msg.threads | 0);
  let ms = Infinity;
  try {
    const ort = await import(moduleUrl(provider));
    ort.env.wasm.wasmPaths = '/lib/ort-wasm/';
    ort.env.logLevel = 'warning';
    ort.env.wasm.numThreads = threads;

    const session = await ort.InferenceSession.create(msg.url, {
      executionProviders: [provider],
      graphOptimizationLevel: 'all',
    });

    const dim = readDim(session);
    const z = new Float32Array(dim);
    for (let i = 0; i < dim; i++) z[i] = Math.random() * 2 - 1;
    const feeds = { var: new ort.Tensor('float32', z, [1, dim]) };

    const warmup = msg.warmup ?? 2;
    for (let i = 0; i < warmup; i++) await session.run(feeds);

    const runs = msg.runs ?? 8;
    const times = [];
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      await session.run(feeds);
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const mid = times.length >> 1;
    ms = times.length % 2 ? times[mid] : (times[mid - 1] + times[mid]) / 2;
  } catch (err) {
    ms = Infinity;
    postMessage({ type: 'result', provider, threads, ms, error: String(err) });
    self.close();
    return;
  }

  postMessage({ type: 'result', provider, threads, ms });
  self.close();
};
