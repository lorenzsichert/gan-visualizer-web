#!/usr/bin/env node
/**
 * Times a single WASM inference of the EndToEndNetwork.onnx model, mirroring
 * the app's inference worker (js/inference-worker.js) minus the hue/RGBA
 * post-processing. Uses onnxruntime-web's WASM backend in Node with threads
 * disabled (numThreads=1) — the browser runs the same runtime multi-threaded
 * via SharedArrayBuffer, so this is a conservative upper bound.
 *
 * Run:  node scripts/bench-ort.mjs     (local server not required)
 */
import * as ort from '../lib/ort-wasm/ort.wasm.min.mjs';
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';

ort.env.wasm.wasmBinary = readFileSync(new URL('../lib/ort-wasm/ort-wasm-simd-threaded.wasm', import.meta.url));
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';

const session = await ort.InferenceSession.create(
  readFileSync(new URL('../models/EndToEndNetwork.onnx', import.meta.url)),
  { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
);

console.log('inputs:', session.inputNames);
console.log('outputs:', session.outputNames, JSON.stringify(session.outputMetadata.img?.shape));

const z = new Float32Array(512);
for (let i = 0; i < 512; i++) z[i] = (Math.random() * 2 - 1) * 2;
await session.run({ var: new ort.Tensor('float32', z, [1, 512]) }); // warmup

const N = 20;
let sum = 0, best = Infinity;
for (let i = 0; i < N; i++) {
  const t = performance.now();
  await session.run({ var: new ort.Tensor('float32', z, [1, 512]) });
  const ms = performance.now() - t;
  sum += ms;
  best = Math.min(best, ms);
}
console.log(`inference: avg ${(sum / N).toFixed(1)} ms, best ${best.toFixed(1)} ms (Node, single-thread WASM)`);
