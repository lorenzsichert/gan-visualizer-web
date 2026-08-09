# GAN Audio Visualizer — Web

A browser port of the Python Qt app in the repo root. It runs the **StyleGAN
end-to-end ONNX model** (`EndToEndNetwork.onnx`) in the browser using
**onnxruntime-web (WebAssembly)**, drives it from your **microphone**, and
renders the generated frames to a canvas.

Everything you need lives in this `web/` folder.

## Quick start

```bash
cd web
npm install          # fetches onnxruntime-web (for the wasm runtime files)
npm run deps         # copies the wasm runtime into lib/ort-wasm/ (see below)
npm start            # starts the server on http://localhost:8080
```

Then open **http://localhost:8080** and click **Enable Microphone**.

> The server must be used (not `file://`). It sets the
> `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers that
> enable **SharedArrayBuffer**, which onnxruntime-web needs for **multi-threaded
> WASM inference** — the single biggest lever for FPS. On a normal machine this
> roughly doubles throughput over single-threaded WASM.

## Features

- **Microphone input** — `getUserMedia` → `AudioWorklet`, which computes the
  Hann-windowed 512-point FFT (256+1 bins) **off the main thread**, exactly
  mirroring the Python `recording.get_sample` pipeline. The FFT runs on a
  *sliding* 512-sample window posted after every 128-frame render quantum
  (~2.9 ms), so capture-side latency is ~2.9 ms instead of a full 512-sample
  block (~11.6 ms).
- **Demo signal** — a synthesized audio signal so you can see it react with no
  mic connected.
- **Lucid-Sonic-Dreams latent modulation** — full port of `lsd.py`
  (pulse/motion/brightness, truncation, motion-sign bouncing).
- **Brightness-direction discovery** — regression of sampled latents against
  output brightness (`main.py:_discover_brightness_*`), run in the worker so
  the UI never stutters.
- **Every parameter slider** from the Python app (smoothing, weights, lowpass,
  pulse/motion, hue shift, tanh). `Smoothing Factor` and `Pulse Smooth` are
  exposed because they dominate the audio → latent latency.
- **Fullscreen** and an **FPS / inference-latency HUD**.

## Architecture (why it's fast)

| Thread | Work |
|---|---|
| **Main** | render loop (`requestAnimationFrame`), latent math, one GPU `drawImage` blit |
| **Audio worklet** | Hann window + FFT, posts the 257-bin spectrum (~344 Hz) |
| **Inference worker** | owns the ORT WASM session; latest-wins queue; converts CHW→RGBA, applies the hue roll, and transfers the frame buffer |

- WASM runs **multi-threaded** (pthreads) thanks to the cross-origin-isolated
  server — all cores work on the 512×512 conv stack.
- All per-pixel work (float → RGBA, hue shift) happens in the **worker**, off
  the UI thread.
- Rendering is a single **GPU-accelerated** `drawImage` of a 512×512 offscreen
  canvas scaled to the window; no per-pixel canvas work on the main thread.
- `computeLatent` reuses scratch buffers instead of allocating per frame, and
  the audio worklet reuses its spectrum buffer (postMessage clones
  synchronously) to avoid GC jank.
- The mic `AudioContext` uses `latencyHint: 'interactive'` with all signal
  processing (echo cancellation, noise suppression, AGC) disabled.

The Python app's 60 FPS timer is replaced by a decoupled "latest-wins" pipeline:
the worker runs inference as fast as it can on the freshest latent, and every
completed frame is drawn immediately.

## Files

```
index.html             page + control panel markup
style.css              dark neon theme
server.js              static server with cross-origin isolation headers
lib/ort-wasm/          self-hosted onnxruntime-web WASM runtime (no CDN)
models/                EndToEndNetwork.onnx (embedded, ~14 MB)
js/main.js             render loop, latent math, UI
js/lsd.js              LSDLatent port
js/audio.js            microphone pipeline (main-thread side)
js/settings.js         parameter definitions (port of midi.settings)
js/inference-worker.js ORT WASM session + RGBA conversion + hue + brightness discovery
worklets/audio-worklet.js  AudioWorklet FFT processor (sliding window)
scripts/latency-test.mjs   node script: measures the audio -> GAN-input DSP latency
scripts/bench-ort.mjs      node script: times a single WASM inference of the ONNX model
```

`npm run deps` is an idempotent helper that re-copies the wasm runtime from
`node_modules` into `lib/ort-wasm/`; the checked-in copies mean the app runs
without a CDN. To upgrade onnxruntime-web, bump the version in
`scripts/fetch-ort.mjs` and re-run it.

## Notes

- Best performance in **Chrome / Edge / Firefox** with cross-origin isolation
  enabled (the bundled server already sets it).
- The model is the same `EndToEndNetwork.onnx` used by the Python app
  (input `var` [1, 512] → output `img` [1, 3, 512, 512]); its external data has
  been embedded so it loads as a single file.
- Output mapping matches the Python app: `(x + 1) * 127.5` clamped to
  `[0, 255]`. A **Tanh Output** slider is provided to soften highlights.
