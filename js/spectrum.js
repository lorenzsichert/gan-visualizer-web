/**
 * Power-spectrum display for the right panel.
 *
 * The magnitude spectrum is resampled onto pixel columns spaced logarithmically
 * in frequency (equal width per semitone — exactly how the keys of a piano are
 * laid out), converted to power (10·log10(mag²)), and drawn as a single clean
 * white curve with faint octave gridlines at the C keys.
 *
 * The input is the A-weighted, `Smoothing Factor`-smoothed spectrum from
 * `main.js computeLatent`. Because it is already A-weighted, this view does not
 * weight it again, and it draws the data raw: no attack/release envelope, no
 * auto-gain averaging, no blur. The brightness and pulse filters shape the same
 * A-weighted spectrum BEFORE the display smoothing, so they track their bands
 * immediately while the curve itself stays smoothed.
 */

const F_MIN = 50; // skip the sub-bass sliver: below ~50 Hz the bins are
                  // still close together
const F_MAX = 18000;
// Fixed 0 dB reference (linear magnitude), so the curve shows absolute level
// and never re-scales itself to the loudness of the input. The worklet rescales
// its 2048-point FFT to the original 1024-point reference (a full-scale sine is
// ~256 there), so this reference stays valid across FFT sizes. Typical signals
// sit well below it — use the Preamp Gain slider, or lower REF_MAG, to bring the
// trace up.
export const REF_MAG = 128; // full-scale sine magnitude (see buildWInjection loudness)
const REF_DB = -60; // display floor (dB below REF_MAG)

/** A-weighting curve in dB (0 dB at 1 kHz). */
export function aWeightDb(f) {
  const f2 = f * f;
  const num = 12194 ** 2 * f2 * f2;
  const den =
    (f2 + 20.6 ** 2) *
    Math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
    (f2 + 12194 ** 2);
  return 20 * Math.log10(num / den) + 2.0;
}

export class SpectrumView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    // Per-pixel-column displayed height (0..1), drawn each frame.
    this.level = new Float32Array(0);

    // Per-source band filters (brightness, pulse, motion), each a Gaussian in
    // log-frequency space visualized as a draggable point with Gaussian decay.
    // Dragging the point moves the filter nearest the pointer (horizontal =
    // center frequency, vertical = how strongly that source reacts); the wheel
    // tunes its width. The bell's peak height mirrors the react value.
    this.filters = [
      { id: 'pulse', color: '#53c1f1',
        fillColor: 'rgba(83, 193, 241, 0.05)', strokeColor: 'rgba(83, 193, 241, 0.30)',
        ringColor: 'rgba(83, 193, 241, 0.28)', freq: null, widthOct: 1,
        react: 0.01, reactMin: 0, reactMax: 1.5 },
      { id: 'brightness', color: '#ffffff',
        fillColor: 'rgba(255, 255, 255, 0.05)', strokeColor: 'rgba(255, 255, 255, 0.22)',
        ringColor: 'rgba(255, 255, 255, 0.28)', freq: null, widthOct: 1,
        react: 0.04, reactMin: 0, reactMax: 1.5 },
      { id: 'motion', color: '#ffd966',
        fillColor: 'rgba(255, 217, 102, 0.05)', strokeColor: 'rgba(255, 217, 102, 0.30)',
        ringColor: 'rgba(255, 217, 102, 0.28)', freq: null, widthOct: 1,
        react: 0.01, reactMin: 0, reactMax: 0.5 },
    ];
    // Called as onFilterChange(id, freq|null, widthOct|null, react|null) from
    // interactions. null args leave that axis unchanged.
    this.onFilterChange = null;

    // Frequency-axis mapping of the most recent frame, reused by handlers.
    this._logMin = Math.log(F_MIN);
    this._logSpan = Math.log(F_MAX / F_MIN);
    this._fMax = F_MAX;

    let dragging = false;
    let activeFilter = null; // filter grabbed on pointerdown; held for the drag
    let dragPointerId = null; // only one pointer drives a drag at a time
    const freqAt = (e) => {
      const r = this.canvas.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      return Math.exp(this._logMin + t * this._logSpan);
    };
    // React amount for a filter at the pointer's vertical position (top = max).
    const reactAt = (e, filter) => {
      const r = this.canvas.getBoundingClientRect();
      const t = 1 - Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
      return filter.reactMin + t * (filter.reactMax - filter.reactMin);
    };
    // Nearest visible filter (by center column) to the pointer, else the
    // brightness filter as a safe default.
    const filterAt = (e) => {
      const r = this.canvas.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      let best = null, bestDist = Infinity;
      for (const f of this.filters) {
        if (f.freq == null || f.freq <= 0) continue;
        const t0 = (Math.log(Math.min(Math.max(f.freq, F_MIN), this._fMax)) - this._logMin) / this._logSpan;
        const d = Math.abs(t0 - t);
        if (d < bestDist) { bestDist = d; best = f; }
      }
      return best || this.filters.find((f) => f.id === 'brightness');
    };
    // Stick the drag to the element for its whole duration: listen on window so
    // the pointer can leave the small canvas (it is only ~110px tall) while
    // dragging up/down to change react and the filter keeps tracking. Keep
    // pointer capture as a belt-and-suspenders fallback for browsers where
    // window listeners would otherwise fire outside the element.
    const onMove = (e) => {
      if (!dragging || !activeFilter || e.pointerId !== dragPointerId) return;
      e.preventDefault();
      this.onFilterChange?.(activeFilter.id, freqAt(e), null, reactAt(e, activeFilter));
    };
    const endDrag = (e) => {
      if (e && e.pointerId !== dragPointerId) return;
      dragging = false;
      activeFilter = null;
      dragPointerId = null;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
    };
    canvas.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      dragging = true;
      dragPointerId = e.pointerId;
      activeFilter = filterAt(e);
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', endDrag);
      window.addEventListener('pointercancel', endDrag);
      try { canvas.setPointerCapture(e.pointerId); } catch {}
      this.onFilterChange?.(activeFilter.id, freqAt(e), null, reactAt(e, activeFilter));
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = filterAt(e);
      const w = Math.min(4, Math.max(0.25, f.widthOct * Math.exp(-e.deltaY * 0.0012)));
      this.onFilterChange?.(f.id, null, w);
    }, { passive: false });
  }

  /**
   * Feed the A-weighted, smoothed magnitude spectrum (linear, one bin per FFT
   * frequency) shared with the latent/LSD path. `filters` maps a filter id to
   * { freq: Hz, width: octaves, react, reactMin, reactMax }.
   */
  update(mags, sampleRate, filters) {
    for (const f of this.filters) {
      const s = filters[f.id];
      if (s) {
        if (s.freq != null) f.freq = s.freq;
        if (s.width != null) f.widthOct = s.width;
        if (s.react != null) f.react = s.react;
        if (s.reactMin != null) f.reactMin = s.reactMin;
        if (s.reactMax != null) f.reactMax = s.reactMax;
      }
    }
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (!cw || !ch) return;
    const w = Math.round(cw * dpr);
    const h = Math.round(ch * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      this.level = new Float32Array(w);
    }

    const n = mags.length - 1; // highest bin index
    const nyq = sampleRate / 2;
    const fMax = Math.min(F_MAX, nyq);
    const logMin = Math.log(F_MIN);
    const logSpan = Math.log(fMax) - logMin;
    // Cache the mapping for the interaction handlers and the filter overlay.
    this._logMin = logMin;
    this._logSpan = logSpan;
    this._fMax = fMax;

    // 0 dB reference is a fixed magnitude, not this frame's peak: the curve is
    // drawn at its absolute level with no auto-gain of any kind.
    const ref2 = REF_MAG * REF_MAG;

    // Resample: max magnitude per log-spaced column -> dB (already A-weighted
    // upstream, so no perceptual weighting is applied here).
    const level = this.level;
    for (let x = 0; x < w; x++) {
      const f0 = Math.exp(logMin + (logSpan * x) / w);
      const f1 = Math.exp(logMin + (logSpan * (x + 1)) / w);
      let b0 = Math.max(1, Math.floor((f0 / nyq) * n));
      let b1 = Math.min(n, Math.max(b0 + 1, Math.ceil((f1 / nyq) * n)));
      let m = 0;
      for (let b = b0; b <= b1; b++) if (mags[b] > m) m = mags[b];
      const db = 10 * Math.log10((m * m) / ref2 + 1e-12);
      level[x] = Math.min(1, Math.max(0, (db - REF_DB) / -REF_DB));
    }

    this.draw(w, h, dpr);
  }

  draw(w, h, dpr) {
    const g = this.ctx;
    const level = this.level;
    g.clearRect(0, 0, w, h);

    const padTop = 5 * dpr;
    const baseY = h; // curve + gradient run to the container bottom
    const plotH = baseY - padTop;

    if (!plotH) return;

    // Band filters (brightness, pulse, motion): each a Gaussian envelope in
    // log-frequency space (decays smoothly to the left and right) with a
    // draggable point at its center, in a distinct color per source. The
    // bell's peak height mirrors how strongly that source reacts (its mix value
    // normalized to the plot), so grabbing the point and dragging up/down
    // retunes the react amount.
    for (const f of this.filters) {
      if (f.freq == null || f.freq <= 0) continue;
      const fc = Math.min(Math.max(f.freq, F_MIN), this._fMax);
      const cx = (w * (Math.log(fc) - this._logMin)) / this._logSpan;
      const sigma = Math.max(f.widthOct, 0.05);
      const inv = 1 / (2 * sigma * sigma);
      const gain = Math.min(1, Math.max(0, (f.react - f.reactMin) / (f.reactMax - f.reactMin)));
      g.beginPath();
      for (let x = 0; x <= w; x++) {
        const fq = Math.exp(this._logMin + (this._logSpan * x) / w);
        const d = Math.log2(fq / fc);
        const env = Math.exp(-d * d * inv) * gain;
        if (x === 0) g.moveTo(x, baseY - env * plotH);
        else g.lineTo(x, baseY - env * plotH);
      }
      // Soft fill under the bell + faint outline of its decay shape.
      g.lineTo(w, baseY);
      g.lineTo(0, baseY);
      g.fillStyle = f.fillColor;
      g.fill();
      g.strokeStyle = f.strokeColor;
      g.lineWidth = 1;
      g.stroke();

      // Center hairline and handle dot at the bell peak (the react height).
      const dotY = baseY - gain * plotH;
      g.strokeStyle = f.strokeColor;
      g.beginPath();
      g.moveTo(cx + 0.5, baseY);
      g.lineTo(cx + 0.5, dotY);
      g.stroke();
      g.beginPath();
      g.arc(cx, dotY, 4.5 * dpr, 0, Math.PI * 2);
      g.fillStyle = f.color;
      g.fill();
      g.beginPath();
      g.arc(cx, dotY, 8 * dpr, 0, Math.PI * 2);
      g.strokeStyle = f.ringColor;
      g.lineWidth = Math.max(1, Math.round(dpr));
      g.stroke();
    }

    // Filled area under the curve, fading toward the bottom edge.
    g.beginPath();
    g.moveTo(0, baseY);
    for (let x = 0; x < w; x++) g.lineTo(x, baseY - level[x] * plotH);
    g.lineTo(w - 1, baseY);
    g.closePath();
    const grad = g.createLinearGradient(0, padTop, 0, baseY);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.38)');
    grad.addColorStop(0.55, 'rgba(255, 255, 255, 0.16)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0.04)');
    g.fillStyle = grad;
    g.fill();

    // Crisp white curve along the top edge, drawn through segment midpoints
    // (quadratic Béziers) so no corner is ever visible. Start where the curve
    // actually leaves the baseline so nothing is drawn flat along the bottom.
    g.beginPath();
    let started = false;
    let px = 0;
    let py = 0;
    for (let x = 0; x < w; x++) {
      const y = baseY - level[x] * plotH;
      if (!started) {
        if (level[x] < 1e-3) continue;
        g.moveTo(x, y);
        px = x;
        py = y;
        started = true;
        continue;
      }
      const mx = (px + x) / 2;
      const my = (py + y) / 2;
      g.quadraticCurveTo(px, py, mx, my);
      px = x;
      py = y;
    }
    if (started) g.lineTo(px, py);
    g.strokeStyle = 'rgba(255, 255, 255, 0.92)';
    g.lineWidth = Math.max(1, Math.round(dpr));
    g.lineJoin = 'round';
    g.stroke();
  }
}
