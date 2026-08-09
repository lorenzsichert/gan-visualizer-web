/**
 * One-shot benchmark worker for a single ONNX thread count.
 *
 * onnxruntime-web bakes `env.wasm.numThreads` into the WASM module when it first
 * initializes and cannot resize the thread pool afterward inside the same realm.
 * So the inference worker spawns one of these per candidate thread count: each
 * instance is a fresh realm with its OWN WASM module, giving each thread count
 * a genuinely fresh pool.
 *
 * It loads the model with the same options the live session uses, warms up, then
 * times `runs` inferences and reports the MEDIAN (which survives GC spikes).
 * The parent picks the fewest threads whose median is within tolerance of the
 * fastest. This worker terminates itself after posting the result.
 */
import * as ort from '/lib/ort-wasm/ort.wasm.min.mjs';

ort.env.wasm.wasmPaths = '/lib/ort-wasm/';
ort.env.logLevel = 'warning';

function readDim(session) {
  const shape = session.inputMetadata ? session.inputMetadata.var?.shape : null;
  if (shape && shape.length > 1 && Number.isFinite(shape[1]) && shape[1] > 0) {
    return Number(shape[1]);
  }
  return 512;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type !== 'bench') return;

  let ms = Infinity;
  try {
    ort.env.wasm.numThreads = Math.max(1, msg.threads | 0);
    const session = await ort.InferenceSession.create(msg.url, {
      executionProviders: ['wasm'],
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
    postMessage({ type: 'result', threads: msg.threads, ms, error: String(err) });
    self.close();
    return;
  }

  postMessage({ type: 'result', threads: msg.threads, ms });
  self.close();
};
