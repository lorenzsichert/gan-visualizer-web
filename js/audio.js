/**
 * Microphone capture + spectrum management for the main thread.
 *
 * Mic audio flows through an AudioWorkletProcessor (worklets/audio-worklet.js)
 * which computes the 257-bin magnitude spectrum off-thread and posts it here.
 * The newest spectrum is stored in `state.spectrum`; the render loop reads it
 * without touching the Web Audio graph.
 */

export class AudioPipeline {
  constructor() {
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.running = false;
    this.spectrum = new Float32Array(257);
    this.onStart = null;
    this.onStop = null;
    this.msgCount = 0;
    // Audio-clock frame index + sample rate of the newest sample in the latest
    // posted spectrum, so the render loop can measure the true audio -> GAN
    // input age (see `audioAgeMs`).
    this.lastFrame = null;
    this.lastSampleRate = null;
    this._pendingUserGesture = false;
  }

  async start() {
    if (this.running) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const ctx = new AudioContext({ latencyHint: 'interactive' });
      await ctx.audioWorklet.addModule('/worklets/audio-worklet.js');
      await ctx.resume();

      const worklet = new AudioWorkletNode(ctx, 'audio-spectrum', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
      });
      worklet.port.onmessage = (e) => {
        // Copy into the shared spectrum buffer (transfer via message would
        // allocate per block; a fixed buffer copy avoids churn).
        const s = e.data.spectrum;
        this.spectrum.set(s.subarray(0, this.spectrum.length));
        if (Number.isFinite(e.data.frame)) {
          this.lastFrame = e.data.frame;
          this.lastSampleRate = e.data.sampleRate || ctx.sampleRate;
        }
        this.msgCount++;
      };

      const source = ctx.createMediaStreamSource(stream);
      source.connect(worklet);

      this.ctx = ctx;
      this.stream = stream;
      this.source = source;
      this.running = true;
      if (this.onStart) this.onStart();
    } catch (err) {
      console.error('Microphone failed:', err);
      throw err;
    }
  }

  async stop() {
    if (!this.running) return;
    try { this.stream.getTracks().forEach((t) => t.stop()); } catch {}
    try { await this.ctx.close(); } catch {}
    this.running = false;
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.spectrum.fill(0);
    this.lastFrame = null;
    this.lastSampleRate = null;
    if (this.onStop) this.onStop();
  }

  /**
   * Age (ms) of the audio driving the GAN right now: the audio-clock time that
   * has passed since the newest sample in the freshest spectrum was captured.
   * This includes the input-device latency, worklet windowing, message transit
   * and the 60 fps render-loop read. null when the mic is off.
   */
  get audioAgeMs() {
    if (!this.ctx || this.lastFrame === null) return null;
    const newestSec = this.lastFrame / this.lastSampleRate;
    return (this.ctx.currentTime - newestSec) * 1000;
  }

  get active() {
    return this.running;
  }
}
