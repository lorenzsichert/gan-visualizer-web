#!/usr/bin/env node
/**
 * Static file server for the GAN Audio Visualizer web app.
 *
 * Serves ./ with the headers required for high-performance onnxruntime-web
 * WASM inference:
 *
 *   - Cross-Origin-Opener-Policy: same-origin
 *   - Cross-Origin-Embedder-Policy: require-corp
 *
 * These two headers make SharedArrayBuffer available, which lets the WASM
 * backend run multi-threaded (pthreads) and use every available CPU core.
 * Without them inference silently falls back to a single thread.
 *
 * Usage:  node server.js [port]   (default port 8080)
 */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = Number(process.env.PORT || process.argv[2] || 8080);
const ROOT = fileURLToPath(new URL('./', import.meta.url));

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.map': 'application/json',
};

// Large, content-addressable-ish binaries can be cached forever; they only
// change when onnxruntime-web or the model is explicitly upgraded. This avoids
// re-downloading the 14 MB model and 13 MB WASM on every reload.
const CACHE = {
  '.wasm': 'public, max-age=604800, immutable',
  '.onnx': 'public, max-age=604800, immutable',
};

const server = createServer(async (req, res) => {
  // Cross-origin isolation => SharedArrayBuffer => multi-threaded WASM.
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');

  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    res.writeHead(400).end('Bad request');
    return;
  }

  if (pathname.endsWith('/')) pathname += 'index.html';

  const filePath = normalize(join(ROOT, pathname));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  try {
    const info = await stat(filePath);
    if (!info.isFile()) throw new Error('not a file');
    const data = await readFile(filePath);
    const ext = extname(filePath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': CACHE[ext] || 'no-cache',
    });
    res.end(data);
  } catch {
    res.writeHead(404).end('Not found');
  }
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    server.listen(0, '127.0.0.1'); // retry on any free port
  } else {
    console.error(err);
    process.exit(1);
  }
});

server.listen(PORT, () => {
  const addr = server.address();
  const port = typeof addr === 'object' && addr ? addr.port : PORT;
  console.log(`GAN Audio Visualizer running at http://localhost:${port}`);
  console.log('Open in Chrome/Edge/Firefox. WASM runs multi-threaded when SharedArrayBuffer is available.');
});
