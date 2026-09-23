/**
 * Persistent model cache shared by the ONNX workers.
 *
 * All models are hosted on the Hugging Face Hub (none are shipped in the repo:
 * GitHub LFS bandwidth is far too small for 100 MB+ files). HF serves them via
 * a `no-store` redirect to a short-lived signed CDN URL, so the browser's HTTP
 * cache never reuses a download.
 *
 * This layer downloads a model once, stores the bytes in the Cache Storage API
 * (available in workers, shared across the origin, persistent across reloads),
 * and hands the bytes to onnxruntime-web so later loads are instant. Callers can
 * pass an `onProgress(loaded, total, cached)` callback: it fires while the body
 * streams in and once more with `cached = true` (and a complete fraction) when
 * the bytes were already stored. The inference worker uses it to drive the
 * "Downloading model" status phase.
 */
const CACHE_NAME = 'gan-model-bytes-v1';
const inflight = new Map();

export function fetchModelBytes(url, onProgress) {
  let pending = inflight.get(url);
  if (!pending) {
    pending = load(url, onProgress).catch((err) => {
      inflight.delete(url);
      throw err;
    });
    inflight.set(url, pending);
  }
  return pending;
}

async function load(url, onProgress) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      const total = Number(hit.headers.get('content-length')) || buf.byteLength;
      onProgress?.(total, total, true);
      return new Uint8Array(buf);
    }
  }

  onProgress?.(0, 0, false);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);

  const total = Number(res.headers.get('content-length')) || 0;

  // Tee the response: one branch fills the cache, the other is read here so we
  // can report progress. cache.put failing (quota / blocked storage) must not
  // break the load.
  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch (err) {
      /* Non-fatal — we still return the bytes we are about to read. */
    }
  }

  const bytes = await readBody(res, onProgress, total);
  onProgress?.(bytes.length, total || bytes.length, false);
  return bytes;
}

async function readBody(res, onProgress, total) {
  if (!res.body || typeof res.body.getReader !== 'function') {
    return new Uint8Array(await res.arrayBuffer());
  }
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    // Report about every 1% (or every 1 MB when the length is unknown).
    const step = total ? total / 100 : 1 << 20;
    if (onProgress && loaded - lastReport >= step) {
      lastReport = loaded;
      onProgress(loaded, total, false);
    }
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch (err) {
    return null; // Cache Storage unavailable (e.g. insecure context).
  }
}
