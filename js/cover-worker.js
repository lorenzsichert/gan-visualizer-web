/**
 * Cover-image worker: renders one deterministic thumbnail per generator model.
 *
 * For each requested model it creates a short-lived onnxruntime-web session,
 * runs a single inference on a fixed, seeded latent, downsamples the 512x512
 * output to a small RGBA square and posts it back. The main thread turns that
 * into a cached JPEG data URL for the model picker.
 *
 * This runs in its own realm with `numThreads = 1` so it doesn't compete with
 * the live inference worker for every core, and it releases each session as
 * soon as its cover is produced. A fixed seed keeps a model's cover identical
 * across reloads and machines, which makes the on-disk cache valid.
 */
const DEFAULT_DIM = 512;
const COVER_SIZE = 160;
const DEFAULT_SEED = 0x5eed;

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard-normal latent from a seeded PRNG (Box-Muller), so the cover is
// reproducible.
function seededLatent(dim, seed) {
  const rnd = mulberry32(seed);
  const z = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rnd() * 2 - 1;
      v = rnd() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    z[i] = u * Math.sqrt((-2 * Math.log(s)) / s);
  }
  return z;
}

function readInputDim(session) {
  const meta = session.inputMetadata;
  const entry = Array.isArray(meta) ? meta.find((m) => m.name === 'var') : meta?.var;
  const shape = entry && entry.shape;
  if (shape && shape.length > 1 && Number.isFinite(shape[1]) && shape[1] > 0) {
    return Number(shape[1]);
  }
  return DEFAULT_DIM;
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (!msg || msg.type !== 'cover') return;

  try {
    const ort = await import('/lib/ort-wasm/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = '/lib/ort-wasm/';
    ort.env.logLevel = 'error';
    ort.env.wasm.numThreads = 1;

    const session = await ort.InferenceSession.create(msg.url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    try {
      const dim = readInputDim(session);
      const z = seededLatent(dim, msg.seed || DEFAULT_SEED);
      const feeds = { var: new ort.Tensor('float32', z, [1, dim]) };
      // Feed zeros for any extra required input (e.g. the W offset `w_add`) so
      // multi-input models can be rendered.
      const names =
        (session.inputNames && session.inputNames.length && [...session.inputNames]) ||
        (session.inputMetadata || []).map((m) => m.name);
      for (const name of names) {
        if (name === 'var') continue;
        const shape = (session.inputMetadata || []).find((m) => m.name === name)?.shape;
        const dims =
          shape && shape.length
            ? shape.map((d) => (Number.isFinite(d) && d > 0 ? Number(d) : 1))
            : [1, dim];
        const count = dims.reduce((a, b) => a * b, 1);
        feeds[name] = new ort.Tensor('float32', new Float32Array(count), dims);
      }
      const out = await session.run(feeds);
      const tensor = out[Object.keys(out)[0]];
      const data = tensor.data;
      const dims = tensor.dims;
      const planes = dims[1] || 3;
      const HH = dims[2] || 512;
      const WW = dims[3] || 512;
      const plane = HH * WW;

      const bytes = new Uint8ClampedArray(COVER_SIZE * COVER_SIZE * 4);
      const sx = WW / COVER_SIZE;
      const sy = HH / COVER_SIZE;
      const useTanh = planes === 3 && msg.tanh;
      for (let y = 0; y < COVER_SIZE; y++) {
        const yy = Math.min(HH - 1, (y * sy) | 0);
        for (let x = 0; x < COVER_SIZE; x++) {
          const xx = Math.min(WW - 1, (x * sx) | 0);
          const p = yy * WW + xx;
          const o = (y * COVER_SIZE + x) * 4;
          const r = data[p];
          const g = planes > 1 ? data[p + plane] : r;
          const b = planes > 2 ? data[p + plane * 2] : r;
          bytes[o] = useTanh ? (Math.tanh(r) + 1) * 127.5 : (r + 1) * 127.5;
          bytes[o + 1] = useTanh ? (Math.tanh(g) + 1) * 127.5 : (g + 1) * 127.5;
          bytes[o + 2] = useTanh ? (Math.tanh(b) + 1) * 127.5 : (b + 1) * 127.5;
          bytes[o + 3] = 255;
        }
      }

      postMessage(
        { type: 'cover', url: msg.url, size: COVER_SIZE, bytes },
        [bytes.buffer]
      );
    } finally {
      await session.release();
    }
  } catch (err) {
    postMessage({
      type: 'cover',
      url: msg.url,
      error: err && err.message ? err.message : String(err),
    });
  }
};
