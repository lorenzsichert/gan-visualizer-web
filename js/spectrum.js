/**
 * Power-spectrum display for the right panel.
 *
 * The magnitude spectrum is resampled onto a fixed number of sample points
 * spaced logarithmically in frequency (equal width per semitone — exactly how
 * the keys of a piano are laid out), converted to power (10·log10(mag²)), and
 * drawn as a single clean white curve with faint octave gridlines at the C
 * keys. The sample count is fixed, so the curve resolution does not depend on
 * the canvas width.
 *
 * The input is the instantaneous A-weighted spectrum from `main.js
 * computeLatent` (no Smoothing Factor EMA on this path). Because it is already
 * A-weighted, this view does not weight it again, and it draws the data raw:
 * no attack/release envelope and no auto-gain averaging. A wide spatial blur
 * over the pixel columns is applied only to what is drawn, to remove the
 * stair-steps the max-per-column resample produces. The brightness and pulse
 * filters shape the same A-weighted spectrum BEFORE the display smoothing, so
 * they track their bands immediately while the curve itself stays smoothed.
 * The selected control tab's band is additionally drawn as its filtered
 * spectrum (the curve shaped by that band's Gaussian response) in the band's
 * color.
 */

const F_MIN = 50; // skip the sub-bass sliver: below ~50 Hz the bins are
                  // still close together
const F_MAX = 18000;
// Fixed 0 dB reference (linear magnitude), so the curve shows absolute level
// and never re-scales itself to the loudness of the input. The worklet rescales
// its 8192-point FFT to the original 1024-point reference (a full-scale sine is
// ~256 there), so this reference stays valid across FFT sizes. Typical signals
// sit well below it — use the Preamp Gain slider, or lower REF_MAG, to bring the
// trace up.
export const REF_MAG = 128; // full-scale sine magnitude (see buildPulseInjection loudness)
const REF_DB = -80; // display floor (dB below REF_MAG)

// Spatial smoothing of the drawn curve. Each display sample takes the max
// magnitude in its log-spaced frequency slice, which leaves stair-steps where a
// slice spans a single FFT bin. A Gaussian-equivalent blur over the samples
// rounds those off while keeping the broad peaks and dips where they are. The
// blur width tapers from wide at the low end to narrow at the high end, so bass
// stays smooth while treble detail survives.
//
// This is SPATIAL only: the spectrum arrives without any temporal smoothing, so
// no attack/release is added here either.
const SAMPLES = 8192; // display sample points across the log-frequency axis
// Gaussian-equivalent sigma in display samples at the left (low) and right
// (high) edge of the log-frequency axis, interpolated linearly in log-frequency
// between them. Implemented as three box passes (below), so the cost is
// O(SAMPLES) and independent of the blur width.
const SMOOTH_SIGMA_LOW = 160;
const SMOOTH_SIGMA_HIGH = 5;
// Per-sample box radius for each of the three passes. Three box passes of this
// radius have combined variance r(r+1), so r is chosen to match the local sigma.
const SMOOTH_RADII = (() => {
  const radii = new Int32Array(SAMPLES);
  for (let x = 0; x < SAMPLES; x++) {
    const t = x / (SAMPLES - 1);
    const sigma = SMOOTH_SIGMA_LOW + (SMOOTH_SIGMA_HIGH - SMOOTH_SIGMA_LOW) * t;
    radii[x] = Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2);
  }
  return radii;
})();
const SMOOTH_RADIUS_MAX = SMOOTH_RADII[0];

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
    // Display sample points (0..1 height) across the log-frequency axis, drawn
    // each frame. `smooth` is the spatially blurred copy the fill and curve are
    // actually drawn from. Both are a fixed SAMPLES long, independent of the
    // canvas pixel width.
    this.level = new Float32Array(SAMPLES);
    this.smooth = new Float32Array(SAMPLES);
    // Selected band's Gaussian response (one weight per display sample) and the
    // filter it belongs to. The response curves the display spectrum so the
    // corresponding part of the trace can be drawn in the tab's color.
    this.activeFilterId = null;
    this._activeBand = null;
    this._bandWeight = new Float32Array(SAMPLES);
    this._bandKey = '';

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
    // Called as onFilterSelect(id) when a filter handle is grabbed, so the UI
    // can bring up the matching control tab.
    this.onFilterSelect = null;

    // Transient highlight: `flash(ids)` makes the named handles blink once
    // (used when the matching control tab is selected).
    this._flashIds = new Set();
    this._flashStart = 0;
    this._flashUntil = 0;

    // Frequency-axis mapping of the most recent frame, reused by handlers.
    this._logMin = Math.log(F_MIN);
    this._logSpan = Math.log(F_MAX / F_MIN);
    this._fMax = F_MAX;
    // Bin range per display sample on the log axis. It depends only on the FFT
    // size and sample rate, not the data, so it is rebuilt only when those
    // change instead of recomputing exp/floor/ceil for every sample each frame.
    this._mapB0 = new Int32Array(SAMPLES);
    this._mapB1 = new Int32Array(SAMPLES);
    this._mapKey = '';

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
      this.onFilterSelect?.(activeFilter.id);
      this.onFilterChange?.(activeFilter.id, freqAt(e), null, reactAt(e, activeFilter));
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = filterAt(e);
      const w = Math.min(4, Math.max(0.25, f.widthOct * Math.exp(-e.deltaY * 0.0012)));
      this.onFilterChange?.(f.id, null, w);
    }, { passive: false });
  }

  /** Blink the given filter handles once (a single decaying pulse). */
  flash(ids) {
    if (!ids || !ids.length) return;
    const now = performance.now();
    this._flashIds = new Set(ids);
    this._flashStart = now;
    this._flashUntil = now + 620;
  }

  /**
   * Highlight the band of one source (pass null for none). The selected band's
   * filtered spectrum is drawn in its color on the next update.
   */
  setActiveFilter(id) {
    this.activeFilterId = id || null;
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

    // Resample: max magnitude per log-spaced sample -> dB (already A-weighted
    // upstream, so no perceptual weighting is applied here). Always SAMPLES
    // points, independent of the canvas pixel width. The bin range per sample
    // is cached; only the per-bin scan below runs every frame.
    const mapKey = `${n}|${sampleRate}`;
    if (mapKey !== this._mapKey) {
      this._mapKey = mapKey;
      const b0s = this._mapB0;
      const b1s = this._mapB1;
      for (let i = 0; i < SAMPLES; i++) {
        const f0 = Math.exp(logMin + (logSpan * i) / (SAMPLES - 1));
        const f1 = Math.exp(logMin + (logSpan * (i + 1)) / (SAMPLES - 1));
        const b0 = Math.max(1, Math.floor((f0 / nyq) * n));
        b0s[i] = b0;
        b1s[i] = Math.min(n, Math.max(b0 + 1, Math.ceil((f1 / nyq) * n)));
      }
    }
    const level = this.level;
    const b0s = this._mapB0;
    const b1s = this._mapB1;
    for (let i = 0; i < SAMPLES; i++) {
      const b0 = b0s[i];
      const b1 = b1s[i];
      let m = 0;
      for (let b = b0; b <= b1; b++) if (mags[b] > m) m = mags[b];
      const db = 10 * Math.log10((m * m) / ref2 + 1e-12);
      level[i] = Math.min(1, Math.max(0, (db - REF_DB) / -REF_DB));
    }

    this.smoothLevel();
    this.computeBand();
    this.draw(w, h, dpr);
  }

  // Gaussian response of the selected band across the display samples. Uses the
  // same log-frequency math as the draggable bells so the colored filtered
  // spectrum lines up exactly with the band drawn behind it. The weights only
  // depend on the band parameters and the frequency axis, so they are cached
  // and rebuilt only while the band is being dragged.
  computeBand() {
    const active = this.activeFilterId
      ? this.filters.find((f) => f.id === this.activeFilterId)
      : null;
    if (!active || active.freq == null || active.freq <= 0) {
      this._activeBand = null;
      this._bandKey = '';
      return;
    }
    this._activeBand = active;

    const key = `${active.id}|${active.freq}|${active.widthOct}|${this._logMin}|${this._logSpan}`;
    if (key === this._bandKey) return;
    this._bandKey = key;

    const fc = Math.min(Math.max(active.freq, F_MIN), this._fMax);
    const logFc = Math.log(fc);
    const sigma = Math.max(active.widthOct, 0.05);
    const inv = 1 / (2 * sigma * sigma);
    const span = this._logSpan / (SAMPLES - 1);
    const weights = this._bandWeight;
    for (let i = 0; i < SAMPLES; i++) {
      const d = (this._logMin + span * i - logFc) / Math.LN2;
      weights[i] = Math.exp(-d * d * inv);
    }
  }

  // Blur the per-sample level into `this.smooth` (edge-clamped), which is what
  // the fill and curve are drawn from. Three box passes approximate a Gaussian;
  // running prefix sums make each pass O(n) and the per-sample radius lets the
  // blur width taper with frequency.
  smoothLevel() {
    const src = this.level;
    const w = src.length;
    if (this.smooth.length !== w) this.smooth = new Float32Array(w);
    if (!this._tmp || this._tmp.length !== w) this._tmp = new Float32Array(w);
    const tmp = this._tmp;
    this._boxBlur(src, tmp);
    this._boxBlur(tmp, this.smooth);
    this._boxBlur(this.smooth, tmp);
    this.smooth.set(tmp);
  }

  // One box blur with a per-sample radius and edge-clamped sampling, via prefix
  // sums. The padded array is built once with SMOOTH_RADIUS_MAX edge samples on
  // each side; each output window is then a constant-time range sum.
  _boxBlur(src, dst) {
    const w = src.length;
    const rMax = SMOOTH_RADIUS_MAX;
    const need = w + 2 * rMax + 1;
    if (!this._prefix || this._prefix.length < need) {
      this._prefix = new Float64Array(need);
    }
    const p = this._prefix;
    let acc = 0;
    p[0] = 0;
    // Prefix sums of the source padded with edge values, so the window never
    // runs off the ends.
    for (let i = 0; i < rMax; i++) p[i + 1] = (acc += src[0]);
    for (let i = 0; i < w; i++) p[rMax + 1 + i] = (acc += src[i]);
    for (let i = 0; i < rMax; i++) p[rMax + w + 1 + i] = (acc += src[w - 1]);
    for (let x = 0; x < w; x++) {
      const r = SMOOTH_RADII[x];
      const lo = x + rMax - r;
      dst[x] = (p[lo + 2 * r + 1] - p[lo]) / (2 * r + 1);
    }
  }

  draw(w, h, dpr) {
    const g = this.ctx;
    const level = this.smooth;
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
      const now = performance.now();
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

      // One-shot blink, triggered by selecting the matching control tab.
      if (this._flashIds.has(f.id)) {
        const dur = this._flashUntil - this._flashStart;
        const progress = dur > 0 ? (now - this._flashStart) / dur : 1;
        if (progress >= 1) {
          this._flashIds.delete(f.id);
        } else {
          const k = Math.sin(Math.PI * Math.min(1, Math.max(0, progress)));
          g.save();
          g.globalAlpha = k;
          g.beginPath();
          g.arc(cx, dotY, (8 + 9 * (1 - k)) * dpr, 0, Math.PI * 2);
          g.strokeStyle = f.color;
          g.lineWidth = Math.max(1.5, 2 * dpr);
          g.stroke();
          g.beginPath();
          g.arc(cx, dotY, 6 * dpr, 0, Math.PI * 2);
          g.fillStyle = f.color;
          g.fill();
          g.restore();
        }
      }
    }

    // Map sample index -> pixel column. `level` holds SAMPLES points across the
    // same log-frequency span the curve covers.
    const n = level.length;
    const xAt = (i) => (n <= 1 ? 0 : (i / (n - 1)) * (w - 1));

    // Filled area under the curve, fading toward the bottom edge.
    g.beginPath();
    g.moveTo(0, baseY);
    for (let i = 0; i < n; i++) g.lineTo(xAt(i), baseY - level[i] * plotH);
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
    for (let i = 0; i < n; i++) {
      const x = xAt(i);
      const y = baseY - level[i] * plotH;
      if (!started) {
        if (level[i] < 1e-3) continue;
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

    // Selected band: the display spectrum shaped by that band's Gaussian
    // response, drawn in the band's color. The filtered curve meets the main
    // trace at the band center and decays to the floor on both sides, so the
    // active tab's contribution is visible directly on the spectrum.
    const band = this._activeBand;
    if (band) {
      const bw = this._bandWeight;
      const yAt = (i) => baseY - level[i] * bw[i] * plotH;

      // Only span the part of the curve where the filtered level is actually
      // visible; otherwise the colored stroke would run flat along the floor.
      let i0 = 0;
      while (i0 < n && level[i0] * bw[i0] < 1e-3) i0++;
      let i1 = n - 1;
      while (i1 > i0 && level[i1] * bw[i1] < 1e-3) i1--;

      if (i1 >= i0) {
        // Soft colored area beneath the filtered curve.
        g.beginPath();
        g.moveTo(xAt(i0), baseY);
        for (let i = i0; i <= i1; i++) g.lineTo(xAt(i), yAt(i));
        g.lineTo(xAt(i1), baseY);
        g.closePath();
        g.save();
        g.globalAlpha = 0.18;
        g.fillStyle = band.color;
        g.fill();
        g.restore();

        // Crisp colored edge with a faint glow, on top of the white trace.
        g.beginPath();
        for (let i = i0; i <= i1; i++) {
          const x = xAt(i);
          const y = yAt(i);
          if (i === i0) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        g.save();
        g.globalAlpha = 0.95;
        g.strokeStyle = band.color;
        g.lineWidth = Math.max(1, 2 * dpr);
        g.lineJoin = 'round';
        g.shadowColor = band.color;
        g.shadowBlur = 10 * dpr;
        g.stroke();
        g.restore();
      }
    }
  }
}
