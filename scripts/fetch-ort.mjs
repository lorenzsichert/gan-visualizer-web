#!/usr/bin/env node
/**
 * Copies the onnxruntime-web WASM runtime from node_modules into lib/ort-wasm/.
 *
 * The checked-in copies mean the app never needs a CDN. Bump the version below
 * (and run `npm install onnxruntime-web@<version>`) to upgrade, then re-run:
 *
 *   node scripts/fetch-ort.mjs
 */
import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const SRC = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
const DST = join(ROOT, 'lib', 'ort-wasm');

const FILES = [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort.wasm.min.js',
  'ort.wasm.min.mjs',
  'ort.wasm.min.mjs.map',
  'ort.wasm.js',
  'ort.wasm.mjs',
];

await mkdir(DST, { recursive: true });

const available = await readdir(SRC).catch(() => {
  console.error('node_modules/onnxruntime-web/dist not found. Run: npm install');
  process.exit(1);
});

for (const f of FILES) {
  if (!available.includes(f)) {
    console.error(`missing ${f} in dist; check the installed onnxruntime-web version`);
    process.exit(1);
  }
  await copyFile(join(SRC, f), join(DST, f));
}

console.log('WASM runtime copied to lib/ort-wasm/');
