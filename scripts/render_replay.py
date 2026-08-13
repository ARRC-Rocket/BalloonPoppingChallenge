#!/usr/bin/env python3
"""Turn a submission into a replay page that opens in any browser.

Closes the gap in #43: the leaderboard renders a run and a competitor cannot,
so the only way to watch your own flight is to keep a matplotlib window open on
the machine that produced it. A submission already carries everything a viewer
needs -- ``balloon_world_data.trajectories`` holds the rocket state, every
balloon state and every balloon status, once per step -- so nothing has to be
re-simulated and no server is involved.

    python scripts/render_replay.py 20260728T115315.000Z_team_x_submission.json

Writes ``<input>.html`` next to the input unless ``-o`` says otherwise. The page
is self-contained: the data is inlined, the drawing is plain canvas, and there
are no external requests, so it works offline and inside a strict content
policy.

What is deliberately left out
-----------------------------
A submission carries ``team.secret``, which is a credential, and
``agent_info.agent_module_file``, which is the agent's source. A replay page is
made to be shared -- attached to an issue, sent to a teammate, posted in a chat
-- so neither is written into it, and ``_page_payload`` is the only place that
decides what goes in. The check at the end of ``render`` is not decoration: it
reads the finished document back and refuses to write a file that contains the
secret.

Size
----
Scenario 1 is 15,000 steps of a hundred balloons, which is more than a viewer
can use and more than a browser wants to parse. ``--stride`` samples every Nth
step, the default of 20 giving 50 ms between frames, and positions are rounded
to 0.1 m because no display pixel resolves better. A scenario-1 run comes out
around a megabyte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VIEWER = r"""<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    color-scheme: light dark;
    --ground: #f4f2ec; --panel: #fbfaf6; --ink: #1d2125; --muted: #6b6f76;
    --hair: #cdc7b8; --rocket: #bf4413; --live: #0d7a84; --inert: #a8a294;
    --scored: #b4860c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground: #14181c; --panel: #1b2026; --ink: #e8e6e1; --muted: #96a0aa;
      --hair: #333c45; --rocket: #ff7a45; --live: #35c8d4; --inert: #5d6670;
      --scored: #f0c14b;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header { padding: 12px 16px 0; }
  h1 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 13px; }
  .stage { position: relative; margin: 12px 16px; }
  canvas {
    width: 100%; height: 62vh; min-height: 320px; display: block;
    background: var(--panel); border: 1px solid var(--hair); border-radius: 8px;
    touch-action: none; cursor: grab;
  }
  canvas:active { cursor: grabbing; }
  .hint {
    position: absolute; left: 10px; bottom: 8px; color: var(--muted);
    font-size: 12px; pointer-events: none;
  }
  .bar { display: flex; gap: 8px; align-items: center; margin: 0 16px 8px; }
  .views { display: flex; gap: 6px; margin: 0 16px 10px; flex-wrap: wrap; }
  button {
    font: inherit; color: inherit; background: var(--panel);
    border: 1px solid var(--hair); border-radius: 6px; padding: 5px 11px;
    cursor: pointer;
  }
  button[aria-pressed="true"] { border-color: var(--ink); font-weight: 600; }
  input[type=range] { flex: 1; min-width: 120px; accent-color: var(--rocket); }
  .readout {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
    gap: 8px; margin: 0 16px 12px;
  }
  .cell {
    background: var(--panel); border: 1px solid var(--hair);
    border-radius: 6px; padding: 6px 9px;
  }
  .k { color: var(--muted); font-size: 11px; text-transform: uppercase; }
  .v { font-variant-numeric: tabular-nums; font-size: 15px; }
  .legend { display: flex; gap: 14px; margin: 0 16px 16px; color: var(--muted);
            font-size: 12px; flex-wrap: wrap; }
  .legend i { width: 9px; height: 9px; border-radius: 50%; display: inline-block;
              margin-right: 5px; }
</style>

<header>
  <h1>__HEADING__</h1>
  <div class="sub">__SUBTITLE__</div>
</header>

<div class="stage">
  <canvas id="c"></canvas>
  <div class="hint">drag to orbit &middot; scroll or pinch to zoom</div>
</div>

<div class="bar">
  <button id="play" aria-pressed="false">&#9654;&nbsp;Play</button>
  <input type="range" id="scrub" min="0" value="0" step="1" aria-label="Time">
  <button id="speed">1&times;</button>
</div>

<div class="views">
  <button id="v-orbit" aria-pressed="true">Orbit</button>
  <button id="v-top" aria-pressed="false">Top-down</button>
  <button id="v-side" aria-pressed="false">Elevation</button>
  <button id="v-trail" aria-pressed="true">Trail</button>
</div>

<div class="readout">
  <div class="cell"><div class="k">T+</div><div class="v" id="r-t">&mdash;</div></div>
  <div class="cell"><div class="k">Altitude</div><div class="v" id="r-alt">&mdash;</div></div>
  <div class="cell"><div class="k">Range</div><div class="v" id="r-rng">&mdash;</div></div>
  <div class="cell"><div class="k">Speed</div><div class="v" id="r-spd">&mdash;</div></div>
  <div class="cell"><div class="k">Popped</div><div class="v" id="r-pop">&mdash;</div></div>
  <div class="cell"><div class="k">Airborne</div><div class="v" id="r-air">&mdash;</div></div>
</div>

<div class="legend">
  <span><i style="background:var(--rocket)"></i>rocket</span>
  <span><i style="background:var(--inert)"></i>on the ground</span>
  <span><i style="background:var(--live)"></i>airborne</span>
  <span><i style="background:var(--scored)"></i>popped</span>
</div>

<script>
const DATA = __DATA__;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const scrub = document.getElementById('scrub');

let frame = 0, playing = false, speed = 1, trail = true, view = 'orbit';
let yaw = -0.6, pitch = 0.45, zoom = 1;
scrub.max = DATA.frames.length - 1;

// One scale for the whole run, so the eye can compare two moments. Recomputed
// only on resize.
let extent = 1, cx = 0, cy = 0, cz = 0;
(function bounds() {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const eat = p => { for (let i = 0; i < 3; i++) {
    if (p[i] < lo[i]) lo[i] = p[i];
    if (p[i] > hi[i]) hi[i] = p[i];
  } };
  for (const f of DATA.frames) { if (f.r) eat(f.r); for (const b of f.b) eat(b); }
  cx = (lo[0] + hi[0]) / 2; cy = (lo[1] + hi[1]) / 2; cz = (lo[2] + hi[2]) / 2;
  extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1) / 2;
})();

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function project(p) {
  const x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
  const rect = canvas.getBoundingClientRect();
  const unit = (Math.min(rect.width, rect.height) * 0.42 * zoom) / extent;
  let u, v;
  if (view === 'top') { u = x; v = -y; }
  else if (view === 'side') { u = Math.hypot(x, y) * Math.sign(x || 1); v = -z; }
  else {
    const cw = Math.cos(yaw), sw = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    u = x * cw - y * sw;
    v = -(z * cp - (x * sw + y * cw) * sp);
  }
  return [rect.width / 2 + u * unit, rect.height / 2 + v * unit];
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawGround() {
  const step = Math.pow(10, Math.round(Math.log10(extent / 3)));
  ctx.strokeStyle = css('--hair');
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.55;
  for (let i = -4; i <= 4; i++) {
    for (const line of [
      [[cx + i * step, cy - 4 * step, DATA.ground], [cx + i * step, cy + 4 * step, DATA.ground]],
      [[cx - 4 * step, cy + i * step, DATA.ground], [cx + 4 * step, cy + i * step, DATA.ground]],
    ]) {
      const a = project(line[0]), b = project(line[1]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawGround();

  const f = DATA.frames[frame];
  const colours = [css('--inert'), css('--live'), css('--scored')];

  // The trail starts at launch, not at frame 0: everything before it has no
  // rocket state at all, and a lineTo(NaN) silently drops the rest of the path.
  if (trail && frame > 0) {
    ctx.strokeStyle = css('--rocket');
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= frame; i++) {
      if (!DATA.frames[i].r) continue;
      const p = project(DATA.frames[i].r);
      started ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
      started = true;
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  for (let i = 0; i < f.b.length; i++) {
    const p = project(f.b[i]);
    ctx.fillStyle = colours[f.s[i]] || colours[0];
    ctx.globalAlpha = f.s[i] === 0 ? 0.5 : 1;
    ctx.beginPath(); ctx.arc(p[0], p[1], f.s[i] === 2 ? 4.5 : 3.2, 0, 6.2832);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  if (f.r) {
    const r = project(f.r);
    ctx.fillStyle = css('--rocket');
    ctx.beginPath(); ctx.arc(r[0], r[1], 5, 0, 6.2832); ctx.fill();
  }
  readout(f);
}

function fmt(v, unit, digits) {
  return v === null || v === undefined || !isFinite(v)
    ? '—' : v.toFixed(digits === undefined ? 1 : digits) + unit;
}

function readout(f) {
  document.getElementById('r-t').textContent = fmt(f.t, ' s');
  document.getElementById('r-alt').textContent =
    f.r ? fmt(f.r[2] - DATA.ground, ' m') : 'on the pad';
  document.getElementById('r-rng').textContent =
    f.r ? fmt(Math.hypot(f.r[0] - DATA.pad[0], f.r[1] - DATA.pad[1]), ' m') : '—';
  document.getElementById('r-spd').textContent = f.r ? fmt(f.v, ' m/s') : '—';
  document.getElementById('r-pop').textContent = String(f.s.filter(x => x === 2).length);
  document.getElementById('r-air').textContent = String(f.s.filter(x => x === 1).length);
}

scrub.addEventListener('input', () => { frame = +scrub.value; draw(); });

function setPlaying(on) {
  playing = on;
  const b = document.getElementById('play');
  b.setAttribute('aria-pressed', String(on));
  b.innerHTML = on ? '&#10073;&#10073;&nbsp;Pause' : '&#9654;&nbsp;Play';
}
document.getElementById('play').addEventListener('click', () => setPlaying(!playing));
document.getElementById('speed').addEventListener('click', e => {
  speed = speed === 8 ? 1 : speed * 2;
  e.currentTarget.textContent = speed + '×';
});

function pickView(name) {
  view = name;
  for (const key of ['orbit', 'top', 'side']) {
    document.getElementById('v-' + key)
      .setAttribute('aria-pressed', String(key === name));
  }
  draw();
}
for (const key of ['orbit', 'top', 'side']) {
  document.getElementById('v-' + key).addEventListener('click', () => pickView(key));
}
document.getElementById('v-trail').addEventListener('click', e => {
  trail = !trail;
  e.currentTarget.setAttribute('aria-pressed', String(trail));
  draw();
});

const pointers = new Map();
let pinch = 0;
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
});
canvas.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 1 && view === 'orbit') {
    yaw += (e.clientX - prev[0]) * 0.008;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - prev[1]) * 0.006));
    draw();
  } else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const span = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) { zoom = Math.max(0.2, Math.min(12, zoom * (span / pinch))); draw(); }
    pinch = span;
  }
});
for (const kind of ['pointerup', 'pointercancel', 'pointerleave']) {
  canvas.addEventListener(kind, e => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = 0;
  });
}
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom = Math.max(0.2, Math.min(12, zoom * Math.exp(-e.deltaY * 0.0012)));
  draw();
}, { passive: false });

let last = 0;
function tick(now) {
  if (playing && now - last > 40 / speed) {
    last = now;
    frame = frame + 1 >= DATA.frames.length ? 0 : frame + 1;
    scrub.value = frame;
    draw();
  }
  requestAnimationFrame(tick);
}
addEventListener('resize', resize);
resize();
requestAnimationFrame(tick);
</script>
"""


def _page_payload(submission, stride):
    """Everything the page is allowed to see, and nothing else.

    One function rather than a filter applied later, so that adding a field to
    the viewer is a decision made here in front of the secret and the agent
    source rather than somewhere downstream that has forgotten about them.
    """
    world = submission["balloon_world_data"]
    info = submission.get("leaderboard_info", {})
    trajectories = world["trajectories"]
    if not trajectories:
        raise ValueError("the submission carries no trajectory to render")

    elevation = float(world["scenario_parameters"]["environment"]["elevation"])
    # The first step with a rocket in it, which is the step launch was commanded
    # on rather than step 0. Reading step 0 gives null, for the same reason the
    # frames below have to tolerate it.
    pad = [0.0, 0.0]
    for record in trajectories:
        state = record["rocket_states"]
        if all(isinstance(v, (int, float)) for v in state[:2]):
            pad = [round(float(state[0]), 1), round(float(state[1]), 1)]
            break

    frames = []
    for record in trajectories[::stride]:
        rocket = record["rocket_states"]
        # Every step before launch has no rocket state. `_json_safe` writes those
        # as null and its own docstring calls them ordinary data rather than an
        # edge case, so a viewer that assumes numbers here breaks on the first
        # second of every run. The frame is kept, since the balloons are already
        # flying, and simply carries no rocket.
        flying = all(isinstance(v, (int, float)) for v in rocket[:6])
        frames.append(
            {
                "t": round(float(record["time"]), 2),
                "r": [round(float(v), 1) for v in rocket[:3]] if flying else None,
                # Speed is carried rather than differenced in the browser: with
                # a stride the frames are no longer one step apart, and a
                # difference across the gap is not the speed at either end.
                "v": round(sum(float(v) ** 2 for v in rocket[3:6]) ** 0.5, 1)
                if flying
                else None,
                "b": [
                    [round(float(v), 1) for v in b[:3]]
                    for b in record["balloon_states"]
                ],
                "s": [int(v) for v in record["balloon_status"]],
            }
        )
    return {
        "frames": frames,
        "ground": round(elevation, 1),
        "pad": pad,
        "team": str(info.get("team_name", "")),
        "agent": str(info.get("agent_name", "")),
        "scenario": info.get("scenario_number"),
        "seed": info.get("random_seed"),
        "score": info.get("final_reward"),
    }


def render(submission, stride):
    """The finished HTML document."""
    payload = _page_payload(submission, stride)
    heading = f"{payload['team'] or 'Replay'} — {payload['agent'] or 'agent'}"
    subtitle = (
        f"scenario {payload['scenario']} · seed {payload['seed']} · "
        f"{payload['score']} popped · {len(payload['frames'])} frames"
    )
    page = (
        VIEWER.replace("__TITLE__", heading)
        .replace("__HEADING__", heading)
        .replace("__SUBTITLE__", subtitle)
        .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    )

    # Read back rather than trust the payload builder. A page is shared, and a
    # leaked team_secret is a credential in someone else's hands.
    secret = str(submission.get("team", {}).get("secret", ""))
    if secret and secret in page:
        raise AssertionError("refusing to write a page containing the team secret")
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("submission", type=Path, help="a submission .json file")
    parser.add_argument(
        "-o", "--out", type=Path, help="output .html (default: alongside)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="keep every Nth step (default 20, which is 50 ms at the shipped time step)",
    )
    args = parser.parse_args(argv)

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    version = submission.get("format_version")
    if version != 1:
        parser.error(f"unsupported submission format_version {version!r}, expected 1")

    out = args.out or args.submission.with_suffix(".html")
    out.write_text(render(submission, args.stride), encoding="utf-8")
    size = out.stat().st_size
    scale, unit = (1e6, "MB") if size >= 1e6 else (1e3, "kB")
    print(f"Replay written to:\n{out}  ({size / scale:.1f} {unit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
