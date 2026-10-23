/**
 * Main thread: UI, audio state, latent modulation (port of main.py
 * `update_frame` + the StyleGAN branch), rendering, and FPS tracking.
 *
 * Heavy work is split for maximum FPS:
 *   - Audio DSP  ......... AudioWorkletProcessor (off-thread FFT)
 *   - ONNX inference ..... inference worker (off-thread WASM, multi-threaded)
 *   - Rendering .......... this thread (single GPU blit)
 */
import { SETTINGS, get } from './settings.js';
import { LSDLatent, randn } from './lsd.js';
import { AudioPipeline } from './audio.js';
import { SpectrumView, aWeightDb, REF_MAG } from './spectrum.js';
import { PreampSlider } from './preamp.js';

const FFT_SIZE = 8192;         // FFT points (power of two)
const BINS = FFT_SIZE / 2 + 1; // spectrum bins (4097)

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');

const elStatus = document.getElementById('status');
const elFps = document.getElementById('fps');
const elInfer = document.getElementById('infer');
const elThreads = document.getElementById('threads');
const elProvider = document.getElementById('provider');
const elModelGallery = document.getElementById('model-gallery');
const elAudio = document.getElementById('audio');
const elToast = document.getElementById('bench-toast');
const elToastRows = document.getElementById('bench-rows');
const elToastNote = document.getElementById('bench-note');
const elModelRows = document.getElementById('model-rows');
const elModelProgress = document.getElementById('model-progress-fill');

// ---------------------------------------------------------------------------
// Persistent latent state (mirrors GANVisualizer.__init__ / _resize_latent_state)
// ---------------------------------------------------------------------------
let DIM = 512;
let lookup = new Uint16Array(DIM);
let a = randn(DIM);
let smoothed = new Float32Array(BINS);
// Instantaneous A-weighted spectrum (no temporal EMA) used by the panel
// display, so the spectrum view follows the audio with no time smoothing.
let instantSpectrum = new Float32Array(BINS);
let prevSpectrum = new Float32Array(BINS);
let spectrum = new Float32Array(BINS);
let audioNoise = new Float32Array(DIM);
let z = new Float32Array(DIM);
let lsd = null;
let brightnessDir = null;
let lastHueFlux = 0;
let audioRandTimer = 0;

// Per-bin A-weighting gain, cached per sample rate. It is applied once to the
// raw spectrum to produce `smoothed`; that smoothed spectrum is the single
// source shared by the latent/LSD path and the panel spectrum view, so the
// weighting must not be applied again downstream.
let aWeightGain = null;
let aWeightSR = 0;

// Rebuild per-dimension state (called on model ready).
function initLatentState(dim) {
  DIM = dim;
  lookup = new Uint16Array(dim);
  for (let i = 0; i < dim; i++) lookup[i] = i % BINS;
  a = randn(dim);
  smoothed.fill(0);
  instantSpectrum.fill(0);
  prevSpectrum.fill(0);
  pulseSpectrum.fill(0);
  lsd = new LSDLatent(dim, get('Truncation'), get('Motion Randomness'));
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------
const audio = new AudioPipeline();
const spectrumView = new SpectrumView(document.getElementById('spectrum'));

// Preamp gain is not a tab slider: it lives as the vertical pill left of the
// spectrum and writes straight into the shared setting read by the audio path.
const preampSlider = new PreampSlider(document.getElementById('preamp'), {
  min: SETTINGS['Preamp Gain'].min,
  max: SETTINGS['Preamp Gain'].max,
  step: SETTINGS['Preamp Gain'].step,
  value: SETTINGS['Preamp Gain'].value,
  onInput: (value) => { SETTINGS['Preamp Gain'].value = value; },
});
SETTINGS['Preamp Gain'].uiSync = () => preampSlider.setValue(SETTINGS['Preamp Gain'].value);

let demo = false;

// Pulse injection source: the first PULSE_DIM spectrum bins, filtered by the
// blue pulse band in computeLatent (A-weighted, no display smoothing) and
// smoothed by Pulse Smooth. buildPulseInjection adds it straight into the
// latent z (see the main loop), not into W space.
const PULSE_DIM = 512; // first 512 FFT bins drive the pulse (~0–3 kHz at 48 kHz)
let pulseSpectrum = new Float32Array(PULSE_DIM);
// 0 = off, 1 = mirror (sharp tiles), 2 = mirror + blurred side tiles.
let mirrorMode = 2;
// When true, the visualizer stops asking for new frames and painting them;
// the last frame stays on screen while audio/spectrum keep running.
let paused = false;

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
  // Peak centers/widths are bin indices for the 8192-point FFT, chosen to sit
  // at the same frequencies in Hz as the original 1024-point demo. Scaling by
  // the FFT-size ratio keeps the demo signal identical at any resolution.
  const scale = FFT_SIZE / 2048;
  const peaks = [
    { f: 32, amp: 0.9 * beat1, w: 12 },
    { f: 80, amp: 0.7 * beat1, w: 16 },
    { f: 180, amp: 0.6 * beat2, w: 24 },
    { f: 360, amp: 0.5 * beat2, w: 32 },
    { f: 600, amp: 0.35, w: 40 },
    { f: 880, amp: 0.3 * (0.5 + 0.5 * Math.sin(t * 5)), w: 48 },
  ];
  for (const p of peaks) {
    const center = p.f * scale;
    const width = p.w * scale;
    for (let i = 0; i < BINS; i++) {
      const d = (i - center) / width;
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
const BRIGHTNESS_KEY = 'brightness-dir';
const OVERRIDE_KEY = 'threads-override';
const PROVIDER_KEY = 'provider-override';
const MODEL_KEY = 'model-url';
const BRIGHTNESS_SAMPLES = 256;

// The model to load (a Hub resolve URL); null means the discovery list has not
// been fetched yet.
let modelUrl = null;
let modelList = [];

// Every model lives on the Hugging Face Hub (none are shipped in the repo —
// GitHub LFS bandwidth is far too small for 100 MB+ files and they bloat every
// deploy). js/model-cache.js caches the bytes so each model is downloaded at
// most once per browser.
const HF_REPO = 'lorenzsichert/gan-visualizer';
const HF_BASE = `https://huggingface.co/${HF_REPO}/resolve/main/`;
const HF_TREE = `https://huggingface.co/api/models/${HF_REPO}/tree/main`;

// Used only when the Hub listing can't be fetched (offline / API hiccup).
const FALLBACK_MODELS = [
  'abstract_art.onnx',
  'abstract_photo.onnx',
  'abstract_photo_medium.onnx',
  'abstract_photo_big.onnx',
  'flowers_big.onnx',
  'visionary.onnx',
  'wildlife_big.onnx',
];

function hubUrl(name) {
  return HF_BASE + encodeURIComponent(name);
}

// List the models from the Hub tree API (names + sizes), falling back to the
// built-in list so the app still boots when the API is unreachable. Dropping a
// new .onnx into the Hub repo makes it selectable without touching the app.
async function discoverModels() {
  let entries = null;
  try {
    const res = await fetch(HF_TREE, { cache: 'no-store' });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        entries = data
          .filter((e) => e.type === 'file' && /\.onnx$/i.test(e.path))
          .map((e) => ({ name: e.path, size: e.size || 0 }));
      }
    }
  } catch (err) {
    /* Hub unreachable — use the fallback below. */
  }
  if (!entries || !entries.length) {
    entries = FALLBACK_MODELS.map((name) => ({ name, size: 0 }));
  }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  return entries.map((m) => ({ name: m.name, size: m.size, url: hubUrl(m.name) }));
}

// Human-readable label for a model file: drop the directory, extension and the
// common `_EndToEndNetwork` suffix.
function modelLabel(name) {
  return name
    .replace(/\.onnx$/i, '')
    .replace(/_EndToEndNetwork$/i, '')
    .replace(/_/g, ' ');
}

// The .onnx files are served with a long immutable cache, so a re-exported model
// would otherwise stay stale in the browser. The listing carries each file's
// size; append it as a version query so a changed model gets a fresh URL.
function versionedUrl(url) {
  const m = modelList.find((x) => x.url === url);
  if (!m || !m.size) return url;
  return `${url}?v=${m.size}`;
}

function loadSavedModel() {
  const override = new URLSearchParams(location.search).get('model');
  if (override) return override;
  try {
    const saved = localStorage.getItem(MODEL_KEY);
    if (saved) return saved;
  } catch (err) {
    /* storage unavailable */
  }
  return null;
}

function saveModel(url) {
  try {
    localStorage.setItem(MODEL_KEY, url);
  } catch (err) {
    /* non-fatal */
  }
}

// Switch models: persist the choice and rebuild the worker so it creates a
// session against the new file.
function applyModel(url) {
  if (!url || url === modelUrl) return;
  modelUrl = url;
  saveModel(url);
  markSelectedModel();
  if (worker) restartWorker();
}

// Generator families (e.g. "abstract photo") and their size variants. Models
// without a size suffix are the small variant; `_medium` / `_big` add the rest.
const SIZE_ORDER = ['small', 'medium', 'big'];
const SIZE_LABELS = { small: 'Small', medium: 'Medium', big: 'Big' };
const SIZE_SUFFIX = /_(small|medium|big)$/i;

// Model authorship, shown as credit under the size options of each expanded
// generator. Keys are the family base names (filename without `.onnx`).
const MODEL_CREDITS = {
  abstract_art: {
    author: 'Michael Friesen',
    links: ['https://x.com/MichaelFriese10'],
  },
  abstract_photo: {
    author: 'Michael Friesen',
    links: ['https://x.com/MichaelFriese10'],
  },
  flowers: {
    author: 'Justin Pinkney',
    links: ['https://www.justinpinkney.com'],
  },
  visionary: {
    author: 'Jeremy Torman',
    links: ['https://x.com/TormanJeremy'],
  },
  wildlife: {
    author: 'Michael Friesen',
    links: ['https://x.com/MichaelFriese10'],
  },
};

let modelFamilies = [];
let modelFamilyByUrl = new Map();

function groupModels(entries) {
  const byBase = new Map();
  for (const m of entries) {
    const file = m.name.replace(/^.*\//, '').replace(/\.onnx$/i, '');
    const match = file.match(SIZE_SUFFIX);
    const base = (match ? file.slice(0, match.index) : file).replace(/_+$/, '');
    const size = match ? match[1].toLowerCase() : 'small';
    let family = byBase.get(base);
    if (!family) {
      family = { base, sizes: {} };
      byBase.set(base, family);
    }
    family.sizes[size] = m;
  }
  return [...byBase.values()].sort((a, b) => a.base.localeCompare(b.base));
}

// The variant a family shows before the user picks one: prefer the smallest.
function preferredUrl(family) {
  for (const size of SIZE_ORDER) {
    if (family.sizes[size]) return family.sizes[size].url;
  }
  return null;
}

function familyUrl(family) {
  for (const size of SIZE_ORDER) {
    if (family.sizes[size] && family.sizes[size].url === modelUrl) return modelUrl;
  }
  return preferredUrl(family);
}

function familyItem(family) {
  if (!elModelGallery) return null;
  for (const item of elModelGallery.children) {
    if (item.dataset.base === family.base) return item;
  }
  return null;
}

function setFamilyExpanded(item, open) {
  if (!item) return;
  item.classList.toggle('open', open);
  const row = item.querySelector('.model-row');
  if (row) row.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// The cover to show as the row icon: the active variant when selected, else the
// family's preferred (smallest) variant.
function iconUrlFor(family) {
  if (modelUrl) {
    for (const size of SIZE_ORDER) {
      const m = family.sizes[size];
      if (m && m.url === modelUrl) return modelUrl;
    }
  }
  return preferredUrl(family);
}

function updateFamilyIcon(family, item) {
  const url = iconUrlFor(family);
  if (!url || item.dataset.iconUrl === url) return;
  const thumb = item.querySelector('.model-thumb');
  const img = item.querySelector('img');
  if (!thumb || !img) return;

  item.dataset.iconUrl = url;
  const model = SIZE_ORDER.map((s) => family.sizes[s]).find((m) => m && m.url === url);
  const cover = model ? loadCachedCover(model) : null;
  if (cover) {
    img.src = cover;
    img.classList.add('ready');
    thumb.classList.remove('loading');
  } else {
    img.removeAttribute('src');
    img.classList.remove('ready');
    thumb.classList.add('loading');
  }
}

// Highlight the row for the active family, open it, and mark the active size.
function markSelectedModel() {
  if (!elModelGallery) return;
  const active = modelUrl ? modelFamilyByUrl.get(modelUrl) : null;
  for (const item of elModelGallery.children) {
    const family = modelFamilies.find((f) => f.base === item.dataset.base);
    if (!family) continue;
    const on = family === active;
    item.classList.toggle('selected', on);
    setFamilyExpanded(item, on);
    const row = item.querySelector('.model-row');
    row.classList.toggle('selected', on);
    row.setAttribute('aria-checked', on ? 'true' : 'false');
    for (const option of item.querySelectorAll('.size-option')) {
      const selected = option.dataset.url === modelUrl;
      option.classList.toggle('active', selected);
      option.classList.toggle('inactive', !selected);
      option.setAttribute('aria-pressed', selected ? 'true' : 'false');
    }
    updateFamilyIcon(family, item);
  }
}

// One row per generator family (icon + name) that expands into a row of size
// options. Missing sizes stay visible but disabled.
function buildModelGallery() {
  if (!elModelGallery) return;
  elModelGallery.innerHTML = '';
  modelFamilies = groupModels(modelList);
  modelFamilyByUrl = new Map();
  for (const family of modelFamilies) {
    for (const size of SIZE_ORDER) {
      if (family.sizes[size]) modelFamilyByUrl.set(family.sizes[size].url, family);
    }
  }

  modelFamilies.forEach((family, index) => {
    const item = document.createElement('div');
    item.className = 'model-item';
    item.dataset.base = family.base;

    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'model-row';
    row.setAttribute('role', 'radio');
    row.setAttribute('aria-checked', 'false');
    row.setAttribute('aria-expanded', 'false');
    row.setAttribute('aria-controls', `model-sizes-${index}`);

    const thumb = document.createElement('span');
    thumb.className = 'model-thumb loading';
    const img = document.createElement('img');
    img.alt = '';
    const glyph = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    glyph.setAttribute('class', 'model-glyph');
    glyph.setAttribute('viewBox', '0 0 24 24');
    glyph.setAttribute('fill', 'none');
    glyph.setAttribute('stroke', 'currentColor');
    glyph.setAttribute('stroke-width', '1.5');
    glyph.setAttribute('stroke-linecap', 'round');
    glyph.setAttribute('stroke-linejoin', 'round');
    glyph.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M3 17l5-6 4 4 3-4 6 7M3 7h18');
    glyph.appendChild(path);
    thumb.append(glyph, img);

    const name = document.createElement('span');
    name.className = 'model-name';
    name.textContent = modelLabel(family.base);

    row.append(thumb, name);

    const sizes = document.createElement('div');
    sizes.className = 'model-sizes';
    sizes.id = `model-sizes-${index}`;
    const inner = document.createElement('div');
    inner.className = 'model-sizes-inner';
    const panel = document.createElement('div');
    panel.className = 'model-sizes-panel';

    for (const size of SIZE_ORDER) {
      const m = family.sizes[size];
      const option = document.createElement('button');
      option.type = 'button';
      option.className = 'size-option';
      option.dataset.size = size;

      const label = document.createElement('span');
      label.className = 'size-label';
      label.textContent = SIZE_LABELS[size];

      const meta = document.createElement('span');
      meta.className = 'size-meta';
      meta.textContent = m && m.size ? `${Math.round(m.size / 1048576)} MB` : '\u2014';

      option.append(label, meta);
      if (m) {
        option.dataset.url = m.url;
        option.title = m.size ? `${m.name} (${(m.size / 1048576).toFixed(1)} MB)` : m.name;
        option.addEventListener('click', () => {
          setFamilyExpanded(item, true);
          applyModel(m.url);
        });
      } else {
        option.disabled = true;
        option.setAttribute('aria-disabled', 'true');
        option.title = `${SIZE_LABELS[size]} not available for ${modelLabel(family.base)}`;
      }
      panel.appendChild(option);
    }

    const credit = MODEL_CREDITS[family.base];
    if (credit) {
      const creditEl = document.createElement('div');
      creditEl.className = 'model-credit';

      const author = document.createElement('span');
      author.className = 'credit-author';
      author.textContent = `Author: ${credit.author}`;
      creditEl.appendChild(author);

      for (const href of credit.links) {
        const link = document.createElement('a');
        link.className = 'credit-link';
        link.href = href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = href;
        creditEl.appendChild(link);
      }
      panel.appendChild(creditEl);
    }

    inner.appendChild(panel);
    sizes.appendChild(inner);
    item.append(row, sizes);

    row.addEventListener('click', () => {
      if (item.classList.contains('selected')) {
        setFamilyExpanded(item, !item.classList.contains('open'));
        return;
      }
      const url = familyUrl(family);
      if (url === modelUrl) markSelectedModel();
      else applyModel(url);
    });

    elModelGallery.appendChild(item);
  });

  markSelectedModel();
}

// Discover the models, build the picker and pick which one to start with.
async function bootModels() {
  modelList = await discoverModels();
  const valid = (url) => modelList.some((m) => m.url === url);

  let wanted = loadSavedModel();
  if (wanted && !valid(wanted)) {
    // Accept a bare filename or an absolute path in the `?model=` override.
    const guess = '/models/' + encodeURIComponent(wanted.replace(/^.*\//, ''));
    wanted = valid(guess) ? guess : null;
  }
  modelUrl = wanted || (modelList[0] && modelList[0].url) || null;
  buildModelGallery();
}

// ---------------------------------------------------------------------------
// Pulse injection (blue pulse filter -> latent z offset)
// ---------------------------------------------------------------------------
// The blue pulse filter is applied to the first PULSE_DIM spectrum bins (in
// computeLatent), and those filtered values are added straight into the latent
// z every frame, so the band shapes which latent dimensions move.
let pulseScratch = null;
let lastPulseMag = 0;

// Scale the pulse-filtered spectrum by Pulse React; REF_MAG keeps it calibrated
// to full scale (a full-scale tone maps to ~Pulse React). Reuses a scratch
// buffer; it is added element-wise into the posted latent.
function buildPulseInjection() {
  lastPulseMag = 0;
  const n = PULSE_DIM;
  if (!pulseScratch || pulseScratch.length !== n) pulseScratch = new Float32Array(n);
  const gain = get('Pulse React') / REF_MAG;
  let mag = 0;
  for (let i = 0; i < n; i++) {
    const v = pulseSpectrum[i] * gain;
    pulseScratch[i] = v;
    if (Math.abs(v) > mag) mag = Math.abs(v);
  }
  lastPulseMag = mag;
  return pulseScratch;
}

// ---------------------------------------------------------------------------
// Model cover images
// ---------------------------------------------------------------------------
// Covers are rendered by js/cover-worker.js (one deterministic inference per
// model) and cached as small JPEG data URLs keyed by URL + file size, so a
// model is only ever rendered once per browser.
const COVER_PREFIX = 'model-cover:';
const COVER_SIZE = 160;
const COVER_SEED = 0x5eed;

let coverWorker = null;
let coverBusy = false;
let coverCurrent = null;
let coversRequested = false;
const coverQueue = [];
const coverQueued = new Set();
// Data URLs for covers already produced this session; localStorage is the
// durable cache, this just avoids a storage read every time a row is repainted.
const coverCache = new Map();

function coverKey(m) {
  return `${COVER_PREFIX}${m.url}:${m.size || 0}`;
}

function loadCachedCover(m) {
  if (coverCache.has(m.url)) return coverCache.get(m.url);
  try {
    const dataUrl = localStorage.getItem(coverKey(m));
    if (dataUrl) coverCache.set(m.url, dataUrl);
    return dataUrl;
  } catch (err) {
    return null;
  }
}

function saveCachedCover(m, dataUrl) {
  coverCache.set(m.url, dataUrl);
  try {
    localStorage.setItem(coverKey(m), dataUrl);
  } catch (err) {
    /* Quota exceeded — regenerate next load. */
  }
}

function setCover(url, dataUrl) {
  const family = modelFamilyByUrl.get(url);
  if (!family) return;
  const item = familyItem(family);
  if (!item) return;
  // Only paint the row if this variant is the one its icon represents.
  if (iconUrlFor(family) !== url) return;
  const img = item.querySelector('img');
  const thumb = item.querySelector('.model-thumb');
  if (img) {
    img.src = dataUrl;
    img.classList.add('ready');
  }
  if (thumb) thumb.classList.remove('loading');
  item.dataset.iconUrl = url;
}

function rgbaToDataUrl(bytes, size) {
  const raw = bytes instanceof Uint8ClampedArray ? bytes : new Uint8ClampedArray(bytes);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(raw, size, size), 0, 0);
  return canvas.toDataURL('image/jpeg', 0.72);
}

function ensureCoverWorker() {
  if (coverWorker) return coverWorker;
  coverWorker = new Worker('/js/cover-worker.js', { type: 'module' });
  coverWorker.onmessage = (e) => onCoverMessage(e.data);
  coverWorker.onerror = () => {
    coverBusy = false;
    coverCurrent = null;
    pumpCoverQueue();
  };
  return coverWorker;
}

// Cover generation is a full model inference per image, so it must not run while
// the site is closed or in a background tab. A hidden tab also stops painting,
// which means the generated images would never be seen; suspending the worker
// saves the GPU/CPU work until the site is opened again.
function suspendCoverWorker() {
  if (coverWorker) {
    coverWorker.terminate();
    coverWorker = null;
  }
  if (coverBusy && coverCurrent) {
    // The interrupted job will never post a result — requeue it at the front.
    coverQueue.unshift(coverCurrent);
  }
  coverBusy = false;
  coverCurrent = null;
}

function pumpCoverQueue() {
  if (coverBusy || !coverQueue.length) return;
  if (document.visibilityState !== 'visible') return;
  coverBusy = true;
  coverCurrent = coverQueue.shift();
  ensureCoverWorker().postMessage({
    type: 'cover',
    url: versionedUrl(coverCurrent.url),
    seed: COVER_SEED,
  });
}

// Resume rendering as soon as the site is visible again; stop right away when
// it is hidden or closed.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') pumpCoverQueue();
  else suspendCoverWorker();
});

// Cover bfcache / navigation as well: a page in the back-forward cache is not
// open, so stop the worker and restart it if the page is restored.
window.addEventListener('pagehide', suspendCoverWorker);
window.addEventListener('pageshow', pumpCoverQueue);

function onCoverMessage(msg) {
  coverBusy = false;
  coverCurrent = null;
  // The worker echoes back the versioned URL it was given; resolve it to the
  // model's plain URL, which is what the tiles and cover cache are keyed by.
  const m =
    modelList.find((x) => versionedUrl(x.url) === msg.url) ||
    modelList.find((x) => x.url === msg.url);
  const key = m ? m.url : msg.url;
  coverQueued.delete(key);
  if (msg.error) {
    const family = modelFamilyByUrl.get(key);
    const item = family ? familyItem(family) : null;
    if (item && iconUrlFor(family) === key) {
      item.classList.add('cover-failed');
      const thumb = item.querySelector('.model-thumb');
      if (thumb) thumb.classList.remove('loading');
    }
  } else {
    const dataUrl = rgbaToDataUrl(msg.bytes, msg.size || COVER_SIZE);
    if (m) saveCachedCover(m, dataUrl);
    setCover(key, dataUrl);
  }
  pumpCoverQueue();
}

// Render covers for every model, the selected one first so it shows up soonest.
function generateCovers() {
  if (!elModelGallery) return;
  const ordered = [...modelList].sort((a, b) => {
    if (a.url === modelUrl) return -1;
    if (b.url === modelUrl) return 1;
    return 0;
  });
  for (const m of ordered) {
    const cached = loadCachedCover(m);
    if (cached) {
      setCover(m.url, cached);
      continue;
    }
    if (coverQueued.has(m.url)) continue;
    coverQueued.add(m.url);
    coverQueue.push(m);
  }
  pumpCoverQueue();
}

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

// Compute config and brightness direction are cached per model, so switching
// back to a model skips both the benchmark and the direction sampling.
function configKeyFor(url) {
  return `${CONFIG_KEY}:${url}`;
}

function brightnessKeyFor(url) {
  return `${BRIGHTNESS_KEY}:${url}`;
}

function cachedConfig() {
  if (!modelUrl) return null;
  try {
    const data = JSON.parse(localStorage.getItem(configKeyFor(modelUrl)) || 'null');
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

function loadCachedBrightness() {
  if (!modelUrl) return null;
  try {
    const data = JSON.parse(localStorage.getItem(brightnessKeyFor(modelUrl)) || 'null');
    if (data && Array.isArray(data.dir) && data.dir.length) {
      return Float32Array.from(data.dir);
    }
  } catch (err) {
    /* ignore — direction is recomputed */
  }
  return null;
}

function saveCachedBrightness(dir) {
  if (!modelUrl || !dir || !dir.length) return;
  try {
    localStorage.setItem(brightnessKeyFor(modelUrl), JSON.stringify({ dir: Array.from(dir) }));
  } catch (err) {
    /* ignore — caching is best-effort */
  }
}

function initWorker() {
  worker = new Worker('/js/inference-worker.js', { type: 'module' });
  worker.onmessage = (e) => handleWorker(e.data);
  worker.onerror = (e) => {
    setStatus('worker error: ' + e.message, 'idle');
    console.error(e);
  };
  resetModelProgress();
  const src = modelUrl ? versionedUrl(modelUrl) : undefined;
  if (window.__dbg) window.__dbg.modelSrc = src || null;
  worker.postMessage({
    type: 'init',
    providers: detectProviders(),
    threads: threadsOverride,
    providerOverride,
    cachedConfig:
      threadsOverride == null && providerOverride == null ? cachedConfig() : null,
    forceBench: new URLSearchParams(location.search).has('bench'),
    url: modelUrl ? versionedUrl(modelUrl) : undefined,
    modelName: modelList.find((m) => m.url === modelUrl)?.name,
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
  setStatus('restarting&hellip;', 'idle');
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

// ---------------------------------------------------------------------------
// Status bar: phase text with a progress fill behind it
// ---------------------------------------------------------------------------
const elStatusCell = elStatus.closest('.stat-status');

let statusText = 'initializing&hellip;';
let statusPhase = 'idle';
let statusStep = null; // 0..1 reported by the worker for download / bench / sample
let download = { active: false, done: false }; // drives the toast's model rows

function modelSizeFor(url) {
  const m = modelList.find((x) => x.url === url);
  return m ? m.size : 0;
}

function formatMB(bytes) {
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function renderStatus() {
  const known = modelSizeFor(modelUrl);

  let frac = null;
  let indeterminate = false;
  let text = statusText;

  if (statusPhase === 'download') {
    if (typeof statusStep === 'number') {
      frac = Math.min(1, statusStep);
      const loaded = known ? known * frac : 0;
      text = `${statusText} ${Math.round(frac * 100)}%${known ? ` (${formatMB(loaded)}/${formatMB(known)})` : ''}`;
    } else {
      indeterminate = true;
    }
  } else if (statusPhase === 'idle') {
    frac = null; // nothing in progress — clear the fill
  } else if (statusPhase === 'bench' || statusPhase === 'sample') {
    if (typeof statusStep === 'number') frac = statusStep;
    else indeterminate = true;
  } else if (statusPhase === 'load') {
    indeterminate = true;
  }

  elStatus.innerHTML = text;
  if (!elStatusCell) return;
  if (frac == null) elStatusCell.style.removeProperty('--load');
  else elStatusCell.style.setProperty('--load', String(frac));
  elStatusCell.classList.toggle('indeterminate', indeterminate);
  elStatusCell.dataset.phase = statusPhase;
}

function setStatus(text, phase = 'idle', step = null) {
  statusText = text;
  statusPhase = phase;
  statusStep = step;
  renderStatus();
}

function resetModelProgress() {
  download = { active: false, done: false };
  updateModelToast();
  elToastRows.innerHTML = '';
  showBenchToast();
  setStatus('starting&hellip;', 'idle');
}

// ---------------------------------------------------------------------------
// Model rows in the benchmark toast
// ---------------------------------------------------------------------------
function modelRow(key, label, value) {
  if (!elModelRows) return;
  let row = elModelRows.querySelector(`[data-key="${key}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'bench-row';
    row.dataset.key = key;
    const name = document.createElement('span');
    name.className = 'bench-name';
    name.textContent = label;
    const val = document.createElement('span');
    val.className = 'bench-ms';
    row.append(name, val);
    elModelRows.appendChild(row);
  }
  row.querySelector('.bench-ms').textContent = value;
}

function updateModelToast() {
  if (!elModelRows) return;
  const m = modelList.find((x) => x.url === modelUrl);
  const known = m ? m.size : 0;
  const frac = download.done
    ? 1
    : statusPhase === 'download' && typeof statusStep === 'number'
      ? statusStep
      : 0;
  const loaded = known * frac;

  let status = 'pending';
  if (download.active) status = 'downloading';
  else if (download.done) status = 'downloaded';

  const pct = Math.round(frac * 100);
  modelRow('model', 'Model', m ? m.name : '—');
  modelRow('size', 'Size', known ? formatMB(known) : '—');
  modelRow('source', 'Source', 'Hugging Face');
  modelRow('status', 'Status', status);
  modelRow('loaded', 'Loaded', known ? `${pct}% · ${formatMB(loaded)}` : `${pct}%`);
  if (elModelProgress) elModelProgress.style.width = `${pct}%`;
}

function handleWorker(msg) {
  switch (msg.type) {
    case 'ready': {
      ready = true;
      window.__dbg.ready = true;
      initLatentState(msg.dim || DIM);
      setStatus('model ready', 'idle');
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
      // Persist the auto-calibrated config (never a manual override) per model,
      // so the next load of this model can skip the benchmark entirely.
      if (msg.bench && !msg.bench.manual && msg.bench.results) {
        try {
          localStorage.setItem(
            configKeyFor(modelUrl),
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
      // Brightness-direction discovery (cached per model, so it only runs the
      // first time a model is used; it never blocks rendering).
      const cachedDir = loadCachedBrightness();
      if (cachedDir) {
        brightnessDir = cachedDir;
        setStatus('brightness direction cached', 'idle');
      } else {
        brightnessDir = null;
        worker.postMessage({ type: 'brightness', samples: BRIGHTNESS_SAMPLES });
      }
      // Generate model cover thumbnails once the live model is up, so the
      // cover work never delays the first frame.
      if (!coversRequested) {
        coversRequested = true;
        generateCovers();
      }
      break;
    }
    case 'status': {
      const wasDownload = statusPhase === 'download';
      statusPhase = msg.phase || 'idle';
      statusText = msg.text;
      statusStep = typeof msg.progress === 'number' ? msg.progress : null;
      if (statusPhase === 'download') {
        download.active = true;
        showBenchToast();
      } else if (wasDownload) {
        download.active = false;
        download.done = true;
      }
      updateModelToast();
      renderStatus();
      break;
    }
    case 'bench-result':
      addBenchRow(msg.provider, msg.threads, msg.ms);
      break;
    case 'result':
      renderResult(msg.bytes, msg.ms, msg.dims);
      // Ack the frame so the worker starts the next inference only after this
      // one was actually displayed (bounds the worker->main result queue).
      worker.postMessage({ type: 'render-done' });
      break;
    case 'brightness':
      brightnessDir =
        msg.dir instanceof Float32Array ? msg.dir : Float32Array.from(msg.dir || []);
      saveCachedBrightness(brightnessDir);
      setStatus(`brightness ready (${msg.samples} samples)`, 'idle');
      break;
  }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
// Offscreen frame, resized to whatever resolution the active model outputs
// (256, 512, 1024, ...) — the worker reports its output dims with each frame.
const off = document.createElement('canvas');
const offCtx = off.getContext('2d');
let renderW = 512;
let renderH = 512;
off.width = renderW;
off.height = renderH;
let img = offCtx.createImageData(renderW, renderH);

function ensureRenderSize(w, h) {
  if (w === renderW && h === renderH) return;
  renderW = w;
  renderH = h;
  off.width = w;
  off.height = h;
  img = offCtx.createImageData(w, h);
}

// Low-res blurred copy of the frame, pre-rendered once per frame when mirror
// mode 2 is active. Blurring with ctx.filter is expensive on the main thread
// (especially Safari), so it is computed once at a fraction of the resolution;
// every tile just blits this small canvas scaled up, which looks identical for
// a soft background and is a fraction of the cost.
const blurTile = document.createElement('canvas');
const blurTileCtx = blurTile.getContext('2d');

function renderResult(bytes, ms, dims) {
  // Never paint while the site is hidden (background tab, minimized window or
  // an OS/compositor that keeps the page "visible" while it is not on screen),
  // or while the user has paused the visualizer. The worker is still acked
  // below so its one in-flight frame completes.
  if (document.visibilityState !== 'visible' || paused) return;
  // dims is [height, width] as reported by the worker.
  const rw = dims && dims[1] > 0 ? dims[1] : renderW;
  const rh = dims && dims[0] > 0 ? dims[0] : renderH;
  ensureRenderSize(rw, rh);
  img.data.set(bytes);
  offCtx.putImageData(img, 0, 0);

  const cw = canvas.width;
  const ch = canvas.height;
  const scale = Math.min(cw / rw, ch / rh);
  const dw = rw * scale;
  const dh = rh * scale;
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

  // --- Preamp ---------------------------------------------------------------
  // Broadband input gain (dB), applied to the raw magnitude spectrum before any
  // downstream processing, so it scales the whole chain — A-weighting,
  // smoothing, band filters and the latent/LSD modulation — for both the mic
  // and the demo signal. Linear gain: dB = 20·log10(g).
  const preampDb = get('Preamp Gain');
  if (preampDb !== 0) {
    const preampGain = Math.pow(10, preampDb / 20);
    for (let i = 0; i < BINS; i++) spectrum[i] *= preampGain;
  }

  // --- A-weighting (perceptual loudness, same curve as the panel display) ---
  // The preamped, A-weighted spectrum is the shared signal consumed by the
  // latent/LSD path and the panel display; brightness and pulse read the same
  // weighting directly (without either the display or latent smoothing).
  // Gain is cached per sample rate; bin i's frequency is i * sr / FFT_SIZE.
  const sr = audio.lastSampleRate || 48000;
  if (!aWeightGain || aWeightSR !== sr) {
    if (!aWeightGain) aWeightGain = new Float32Array(BINS);
    for (let i = 0; i < BINS; i++) {
      aWeightGain[i] = Math.pow(10, aWeightDb((i * sr) / FFT_SIZE) / 20);
    }
    aWeightSR = sr;
  }

  // --- Smoothing ---
  // One A-weighted value per bin feeds both paths: `instantSpectrum` (no EMA)
  // drives the panel display, while `smoothed` (the Smoothing Factor EMA)
  // drives the latent/LSD modulation.
  const smoothingFactor = get('Smoothing Factor');
  const smoothing = 1 - Math.exp((-dt * 10) / Math.max(smoothingFactor, 1e-6));
  for (let i = 0; i < BINS; i++) {
    const v = spectrum[i] * aWeightGain[i];
    instantSpectrum[i] = v;
    smoothed[i] = smoothing * smoothed[i] + (1 - smoothing) * v;
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
  // log-frequency space (draggable on the spectrum): they react only around
  // their center frequency, decaying smoothly to both sides. Motion reacts to
  // flux; brightness and the blue pulse react to A-weighted magnitude WITHOUT
  // the display smoothing, so they follow the band immediately. The pulse
  // filter's per-bin output is added to the latent z (see buildPulseInjection).
  const bfFreq = Math.max(get('Brightness Freq'), 1);
  const bfWidth = Math.max(get('Brightness Width'), 0.05);
  const bfNorm = 1 / (2 * bfWidth * bfWidth);
  const mfFreq = Math.max(get('Motion Freq'), 1);
  const mfWidth = Math.max(get('Motion Width'), 0.05);
  const mfNorm = 1 / (2 * mfWidth * mfWidth);
  const pfFreq = Math.max(get('Pulse Freq'), 1);
  const pfWidth = Math.max(get('Pulse Width'), 0.05);
  const pfNorm = 1 / (2 * pfWidth * pfWidth);

  let lowPassBright = 0;
  for (let i = 1; i < BINS; i++) {
    const d = Math.log2((i * sr) / (FFT_SIZE * bfFreq));
    // A-weighted raw (preamped) spectrum, not the EMA-smoothed one, so the
    // brightness react follows the band immediately instead of through the
    // display smoothing.
    const v = spectrum[i] * aWeightGain[i] * Math.exp(-d * d * bfNorm);
    if (v > lowPassBright) lowPassBright = v;
  }

  // Blue pulse filter, applied per bin to the first PULSE_DIM bins of the
  // A-weighted raw spectrum (no display smoothing). These become the z-space
  // injection directly, so the band shapes which latent dimensions move. Pulse
  // Smooth EMA (60 fps-normalized, matching the LSD EMAs) glides the vector.
  const pulsePs = Math.pow(get('Pulse Smooth'), 60 * dt);
  for (let i = 0; i < PULSE_DIM && i < BINS; i++) {
    const f = (i * sr) / FFT_SIZE;
    const d = Math.log2(f / pfFreq);
    const w = Math.exp(-d * d * pfNorm);
    const target = spectrum[i] * aWeightGain[i] * w;
    pulseSpectrum[i] = pulseSpectrum[i] * pulsePs + target * (1 - pulsePs);
  }

  const invDt = 1 / Math.max(dt, 1e-6);
  let lowPassDrift = 0;
  let motionAmp = 0;
  for (let i = 0; i < BINS; i++) {
    const f = (spectrum[i] - prevSpectrum[i]) * invDt;
    if (f > lowPassDrift) lowPassDrift = f;
    const dm = Math.log2((i * sr) / (FFT_SIZE * mfFreq));
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
    pulseAmp: 0, // the blue pulse filter is added to the latent directly (buildPulseInjection), not via the LSD pulse path
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
    brightnessSmooth: get('Brightness Smooth'),
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

// Keep the frame clock honest across a hide/show cycle so the first visible
// frame does not report a bogus FPS or a huge dt.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    lastT = performance.now() / 1000;
    fpsWindow = performance.now();
    fpsFrames = 0;
  }
});

function loop(now) {
  // rAF normally pauses in a hidden tab, but some browsers/OS window managers
  // keep running it for minimized or occluded windows. Stop all inference and
  // painting explicitly until the site is visible again.
  if (document.visibilityState !== 'visible') {
    lastT = now / 1000;
    fpsWindow = now;
    fpsFrames = 0;
    requestAnimationFrame(loop);
    return;
  }

  const nowSec = now / 1000;
  const dt = Math.min(Math.max(nowSec - lastT, 1e-4), 0.1);
  lastT = nowSec;

  if (!paused && ready && lsd) {
    const z = computeLatent(dt, nowSec);
    // Add the blue pulse filter's output to the latent (z) space. Tiled if the
    // latent is longer than the pulse vector.
    const pulse = buildPulseInjection();
    for (let i = 0; i < z.length; i++) z[i] += pulse[i % pulse.length];
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
    window.__dbg.pulseInj = +lastPulseMag.toFixed(4);
  }

  // Panel spectrum display: the A-weighted spectrum WITHOUT the Smoothing
  // Factor EMA — only the worklet's short 2048-sample analysis window shapes
  // it, so the trace follows transients immediately. The view adds spatial
  // smoothing only.
  spectrumView.update(
    instantSpectrum,
    audio.lastSampleRate || 48000,
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
// The panel settings split into tabs below the spectrum. Each tab button is
// tinted with the same color as its draggable band filter on the spectrum
// (pulse blue, brightness white, motion yellow); "General" has no spectrum band
// so it keeps the neutral signal accent. Every tab is built once (sliders stay in
// the DOM) and switching only toggles visibility, so external sync via
// `syncSettingUI` keeps working regardless of which tab is showing.
const TAB_KEY = 'control-tab';
const CONTROL_TABS = [
  { id: 'general', label: 'General', color: '#e8eaee', soft: 'rgba(255, 255, 255, 0.14)' },
  { id: 'motion', label: 'Motion', color: '#ffd966', soft: 'rgba(255, 217, 102, 0.16)' },
  { id: 'pulse', label: 'Pulse', color: '#53c1f1', soft: 'rgba(83, 193, 241, 0.16)' },
  { id: 'brightness', label: 'Bright', color: '#ffffff', soft: 'rgba(255, 255, 255, 0.14)' },
];

// Which spectrum filter handles blink when a tab is selected. General has no
// band of its own, so it blinks all three.
const TAB_FILTERS = {
  general: ['pulse', 'brightness', 'motion'],
  motion: ['motion'],
  pulse: ['pulse'],
  brightness: ['brightness'],
};

// Which tab a setting belongs to, keyed off its name prefix (the settings
// groups predate this split and lump Brightness in with Pulse).
function tabForSetting(name) {
  if (name.startsWith('Motion ')) return 'motion';
  if (name.startsWith('Pulse ')) return 'pulse';
  if (name.startsWith('Brightness ')) return 'brightness';
  return 'general';
}

// Set by buildSliders; lets spectrum interactions bring up their tab.
let activateTab = null;

function buildSliderRow(name, def) {
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
  // Keep UI references so external changes (e.g. dragging the filter point on
  // the spectrum) can sync the slider position and label.
  def.uiInput = input;
  def.uiVal = setVal;
  input.addEventListener('input', () => {
    def.value = parseFloat(input.value);
    setVal();
  });

  row.append(label, input);
  return row;
}

function buildSliders() {
  const container = document.getElementById('sliders');
  container.innerHTML = '';

  const tabbar = document.createElement('div');
  tabbar.className = 'tabs';
  tabbar.setAttribute('role', 'tablist');

  const panels = document.createElement('div');
  panels.className = 'tab-panels';

  const buttons = new Map();
  const panes = new Map();

  const activate = (id, blink) => {
    for (const [tid, btn] of buttons) {
      const on = tid === id;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    }
    for (const [tid, pane] of panes) pane.classList.toggle('active', tid === id);
    // Draw that band's filtered spectrum in its color on the spectrum view.
    spectrumView.setActiveFilter(id === 'general' ? null : id);
    try {
      localStorage.setItem(TAB_KEY, id);
    } catch (err) {
      /* storage unavailable */
    }
    if (blink) spectrumView.flash(TAB_FILTERS[id] || []);
  };
  activateTab = activate;

  for (const tab of CONTROL_TABS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tab';
    btn.textContent = tab.label;
    btn.style.setProperty('--tab-color', tab.color);
    btn.style.setProperty('--tab-soft', tab.soft);
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', 'false');
    btn.addEventListener('click', () => activate(tab.id, true));
    buttons.set(tab.id, btn);
    tabbar.appendChild(btn);

    const pane = document.createElement('div');
    pane.className = 'tab-panel';
    pane.setAttribute('role', 'tabpanel');
    pane.dataset.tab = tab.id;
    for (const [name, def] of Object.entries(SETTINGS)) {
      if (def.hidden || def.detached || tabForSetting(name) !== tab.id) continue;
      pane.appendChild(buildSliderRow(name, def));
    }
    panes.set(tab.id, pane);
    panels.appendChild(pane);
  }

  container.append(tabbar, panels);

  let saved = null;
  try {
    saved = localStorage.getItem(TAB_KEY);
  } catch (err) {
    /* storage unavailable */
  }
  activate(buttons.has(saved) ? saved : CONTROL_TABS[0].id);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);

// Push an externally-changed setting into its slider UI (if visible).
function syncSettingUI(name) {
  const def = SETTINGS[name];
  if (!def) return;
  if (def.uiSync) def.uiSync();
  if (!def.uiInput) return;
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
  spectrumView.onFilterSelect = (id) => {
    // Grabbing a band handle on the spectrum brings up that band's controls.
    if (CONTROL_TABS.some((tab) => tab.id === id)) activateTab?.(id, false);
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

  // Hide the floating sidebar opener while the pointer is idle; it reappears on
  // any movement (motion, click, touch or keypress).
  const IDLE_MS = 2500;
  let idleTimer = 0;
  const markPointerActive = () => {
    document.body.classList.remove('pointer-idle');
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => document.body.classList.add('pointer-idle'), IDLE_MS);
  };
  window.addEventListener('mousemove', markPointerActive, { passive: true });
  window.addEventListener('mousedown', markPointerActive, { passive: true });
  window.addEventListener('touchstart', markPointerActive, { passive: true });
  window.addEventListener('keydown', markPointerActive);
  markPointerActive();

  const btnMic = document.getElementById('btn-mic');
  const micIndicator = document.getElementById('mic-indicator');

  function updateAudioStatus() {
    if (audio.active) {
      elAudio.textContent = 'on';
      elAudio.classList.add('on');
    } else if (demo) {
      elAudio.textContent = 'demo';
      elAudio.classList.add('on');
    } else {
      elAudio.textContent = 'off';
      elAudio.classList.remove('on');
    }
  }

  function setMicUI(active) {
    btnMic.querySelector('.btn-label').textContent = active
      ? 'Microphone: ON'
      : 'Enable Microphone';
    btnMic.classList.toggle('active', active);
    micIndicator.classList.toggle('off', !active);
    micIndicator.setAttribute('aria-label', active ? 'Microphone on' : 'Microphone off');
    micIndicator.title = active ? 'Microphone on' : 'Microphone off';
    preampSlider.setEnabled(active);
    updateAudioStatus();
  }

  btnMic.addEventListener('click', async () => {
    if (audio.active) {
      await audio.stop();
      setMicUI(false);
    } else {
      try {
        await audio.start();
        setMicUI(true);
      } catch (err) {
        setStatus('microphone unavailable');
      }
    }
  });

  const btnDemo = document.getElementById('btn-demo');
  btnDemo.addEventListener('click', () => {
    demo = !demo;
    btnDemo.classList.toggle('active', demo);
    updateAudioStatus();
  });

  const btnPause = document.getElementById('btn-pause');
  const pauseLabel = btnPause.querySelector('span');
  function updatePauseUI() {
    btnPause.classList.toggle('active', paused);
    btnPause.setAttribute('aria-pressed', paused ? 'true' : 'false');
    pauseLabel.textContent = paused ? 'Resume' : 'Pause';
    btnPause.title = paused
      ? 'Resume the visualizer'
      : 'Pause the visualizer; audio keeps running';
  }
  btnPause.addEventListener('click', () => {
    paused = !paused;
    updatePauseUI();
  });
  updatePauseUI();

  // Mic starts off: the indicator is struck through and the preamp is inert.
  setMicUI(false);

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
await bootModels();
initWorker();
requestAnimationFrame(loop);
