#!/usr/bin/env node
/**
 * Demonstrates why display lag grows without bound when the worker outruns the
 * main thread, and that ack-throttling fixes it.
 *
 * Model:
 *   - worker produces a `result` postMessage every INFER_MS (as fast as it can)
 *   - main thread consumes one queued result every RENDER_MS (putImageData +
 *     drawImage + GC). If results arrive faster than the main thread can draw,
 *     the worker -> main message queue grows.
 *   - the displayed frame is the oldest in the queue, so a growing queue = a
 *     frame that is increasingly stale.
 *
 * "Delay keeps building up" == the queue length grows over time.
 */

const INFER_MS = 11;     // multi-threaded WASM inference (fast!)
const RENDER_MS = 20;    // main thread is slow this session (busy/GC)
const SIM_MS = 8000;

function simulate({ ack = false } = {}) {
  let queue = 0;              // results waiting to be rendered
  let nextInferAt = 0;        // when the worker emits its next result
  let renderDoneAt = 0;       // when the main thread finishes the current render
  let waitingAck = false;     // worker blocked until the main thread acks
  let history = [];

  for (let t = 0; t < SIM_MS; t += 1) {
    // Worker emits a result every INFER_MS (unless ack-throttled).
    if (t >= nextInferAt) {
      if (ack) {
        if (!waitingAck) {
          queue++;            // one in flight
          waitingAck = true;
          nextInferAt = t + INFER_MS;
        }
      } else {
        queue++;
        nextInferAt = t + INFER_MS;
      }
    }

    // Main thread renders one queued result every RENDER_MS.
    if (t >= renderDoneAt && queue > 0) {
      queue--;
      renderDoneAt = t + RENDER_MS;
      if (ack) waitingAck = false; // ack sent after the render
    }

    history.push({ t, queue });
  }
  return history;
}

function report(label, h) {
  const q = h.map((x) => x.queue);
  const avg = q.reduce((a, b) => a + b, 0) / q.length;
  const last = q[q.length - 1];
  console.log(`\n--- ${label} ---`);
  console.log(`  avg queued results  : ${avg.toFixed(1)}`);
  console.log(`  queue at end        : ${last}   ${last > 5 ? '<-- backlog keeps growing' : '(bounded)'}`);
  console.log(`  approx display lag  : ${(avg * RENDER_MS).toFixed(0)} ms`);
}

console.log(`worker produces every ${INFER_MS} ms; main thread renders every ${RENDER_MS} ms (main thread is the bottleneck)\n`);
report('BEFORE: worker unthrottled (result postMessages pile up)', simulate({ ack: false }));
report('AFTER : worker ack-throttled (waits for render before next)', simulate({ ack: true }));
