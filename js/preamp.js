/**
 * Vertical preamp-gain control shown to the left of the spectrum.
 *
 * Styled after the iOS volume slider: a tall rounded pill whose fill rises from
 * the bottom in proportion to the current value, with no separate handle. The
 * whole pill is the hit target; dragging anywhere scrubs the value, and the
 * fill edge tracks the pointer exactly while dragging. A small readout follows
 * the fill edge on hover, focus, or drag so the dB value is still discoverable
 * without adding a permanent label next to the (height-matched) spectrum.
 */
export class PreampSlider {
  constructor(root, { min, max, step, value, onInput } = {}) {
    this.root = root;
    this.min = min;
    this.max = max;
    this.step = step;
    this.onInput = onInput || null;
    this.value = this._quantize(value);

    this._fill = root.querySelector('.preamp-fill');
    this._readout = root.querySelector('.preamp-db');
    this._dragging = false;
    this._pointerId = null;
    this.enabled = true;

    root.setAttribute('aria-valuemin', String(min));
    root.setAttribute('aria-valuemax', String(max));
    this._render();
    this._bind();
  }

  /** Set the value from outside (slider stays the source of truth for input). */
  setValue(value) {
    this._setValue(value, false);
  }

  /** Enable/disable interaction; a disabled preamp is visually greyed out. */
  setEnabled(enabled) {
    this.enabled = enabled !== false;
    this.root.classList.toggle('disabled', !this.enabled);
    this.root.setAttribute('aria-disabled', this.enabled ? 'false' : 'true');
    this.root.tabIndex = this.enabled ? 0 : -1;
    if (!this.enabled) {
      // Abort a drag in progress; the window listeners clean themselves up on
      // the next pointerup.
      this._dragging = false;
      this._pointerId = null;
      this.root.classList.remove('is-dragging');
    }
  }

  _quantize(value) {
    if (!Number.isFinite(value)) return this.min;
    const n = Math.round((value - this.min) / this.step);
    const snapped = this.min + n * this.step;
    // Kill float noise from the step math (e.g. -11.999999999).
    return Math.min(this.max, Math.max(this.min, Number(snapped.toFixed(6))));
  }

  _setValue(value, notify = true) {
    const next = this._quantize(value);
    const changed = next !== this.value;
    this.value = next;
    this._render();
    if (notify && changed) this.onInput?.(this.value);
  }

  // Pointer position -> value, with the top of the pill at max.
  _valueAt(e) {
    const r = this.root.getBoundingClientRect();
    const t = 1 - Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    return this.min + t * (this.max - this.min);
  }

  _bind() {
    const root = this.root;

    const onMove = (e) => {
      if (!this._dragging || e.pointerId !== this._pointerId) return;
      e.preventDefault();
      this._setValue(this._valueAt(e));
    };
    const endDrag = (e) => {
      if (e && e.pointerId !== this._pointerId) return;
      this._dragging = false;
      this._pointerId = null;
      root.classList.remove('is-dragging');
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
    };

    root.addEventListener('pointerdown', (e) => {
      if (!this.enabled) return;
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      this._dragging = true;
      this._pointerId = e.pointerId;
      root.classList.add('is-dragging');
      root.focus({ preventScroll: true });
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', endDrag);
      window.addEventListener('pointercancel', endDrag);
      try { root.setPointerCapture(e.pointerId); } catch {}
      this._setValue(this._valueAt(e));
    });

    root.addEventListener('keydown', (e) => {
      if (!this.enabled) return;
      let delta = 0;
      switch (e.key) {
        case 'ArrowUp':
        case 'ArrowRight':
          delta = this.step;
          break;
        case 'ArrowDown':
        case 'ArrowLeft':
          delta = -this.step;
          break;
        case 'PageUp':
          delta = this.step * 10;
          break;
        case 'PageDown':
          delta = -this.step * 10;
          break;
        case 'Home':
          e.preventDefault();
          this._setValue(this.min);
          return;
        case 'End':
          e.preventDefault();
          this._setValue(this.max);
          return;
        default:
          return;
      }
      e.preventDefault();
      this._setValue(this.value + delta);
    });
  }

  _render() {
    const span = this.max - this.min;
    const t = span > 0 ? (this.value - this.min) / span : 0;
    this._fill.style.height = `${t * 100}%`;
    // Readout follows the fill edge, clamped so the bubble stays inside the card.
    this.root.style.setProperty('--pos', `${(1 - t) * 100}%`);

    const text = `${this.value > 0 ? '+' : ''}${this.value.toFixed(1)}`.replace('-', '\u2212');
    if (this._readout) this._readout.textContent = `${text} dB`;
    this.root.setAttribute('aria-valuenow', String(this.value));
    this.root.setAttribute('aria-valuetext', `${text} dB`);
  }
}
