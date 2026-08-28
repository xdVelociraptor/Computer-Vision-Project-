"""Export a point cloud as a self-contained interactive HTML viewer.

Produces a single .html file with the geometry embedded as base64 and a small
hand-written WebGL renderer. No external libraries, no server, no network - it
opens straight from disk in any browser and can be shared as one file.

Controls in the viewer: left-drag orbit, wheel zoom, right-drag (or shift-drag)
pan, plus point-size and background controls.

Examples
--------
    python scripts/export_web_viewer.py --scene scene_01
    python scripts/export_web_viewer.py --scene scene_01 --max-points 250000 --clean
    python scripts/export_web_viewer.py --scene scene_01 --what sparse
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ConfigError, load_config  # noqa: E402
from src.utils import ensure_dir, human_size, setup_logging  # noqa: E402

#: Rough bytes per point once base64-encoded (12 B position + 3 B colour, x4/3).
BYTES_PER_POINT_B64 = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an interactive, self-contained HTML point-cloud viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, action="append", default=None, metavar="YAML")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--what", choices=["dense", "sparse", "two_view"], default="dense")
    parser.add_argument("--path", type=Path, default=None, help="Explicit .ply to export.")
    parser.add_argument("--output", type=Path, default=None, help="Output .html path.")
    parser.add_argument(
        "--max-points", type=int, default=400000,
        help="Subsample above this. Keeps the file openable in a browser.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove statistical outliers and the 1%%/99%% tails before export.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    return parser


def _resolve_path(cfg, args) -> Path:
    if args.path:
        return Path(args.path)
    out = cfg.scene_output_dir
    candidates = {
        "dense": [out / "dense" / "fused.ply"],
        "sparse": sorted(out.glob("sparse/*_points.ply")),
        "two_view": [out / "two_view" / "two_view_points.ply"],
    }[args.what]
    for path in candidates:
        if Path(path).is_file():
            return Path(path)
    raise FileNotFoundError(
        f"No '{args.what}' cloud found for scene '{cfg.scene.name}'. "
        f"Looked for: {[str(c) for c in candidates]}"
    )


def build_html(points: np.ndarray, colors: np.ndarray, *, title: str, subtitle: str,
               point_size: float) -> str:
    """Assemble the self-contained viewer HTML.

    Geometry is centred on the origin and scaled so the largest dimension spans
    2 units, which lets the camera and zoom limits be fixed constants rather
    than depending on the reconstruction's arbitrary units.
    """
    centre = (points.max(axis=0) + points.min(axis=0)) / 2.0
    centred = points - centre
    scale = float(np.abs(centred).max())
    if scale <= 0:
        scale = 1.0
    normalised = (centred / scale).astype(np.float32)

    pos_b64 = base64.b64encode(normalised.tobytes()).decode("ascii")
    col_b64 = base64.b64encode(colors.astype(np.uint8).tobytes()).decode("ascii")

    return _TEMPLATE.replace("__TITLE__", title) \
                    .replace("__SUBTITLE__", subtitle) \
                    .replace("__NPOINTS__", str(len(normalised))) \
                    .replace("__POINTSIZE__", f"{point_size:g}") \
                    .replace("__POS_B64__", pos_b64) \
                    .replace("__COL_B64__", col_b64)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0e1013; --panel:#181b20; --fg:#e8eaed; --muted:#9aa3ad; --accent:#4a9eff; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; overflow:hidden;
              font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  body { background:var(--bg); color:var(--fg); }
  canvas { display:block; width:100vw; height:100vh; cursor:grab; }
  canvas.dragging { cursor:grabbing; }
  #hud { position:fixed; top:14px; left:14px; background:rgba(24,27,32,.86);
         border:1px solid #2a2f37; border-radius:10px; padding:12px 14px;
         backdrop-filter:blur(8px); max-width:290px; }
  #hud h1 { margin:0 0 2px; font-size:14px; font-weight:600; letter-spacing:.2px; }
  #hud .sub { color:var(--muted); font-size:11px; margin-bottom:10px; }
  .row { display:flex; align-items:center; gap:8px; margin:7px 0; font-size:11px;
         color:var(--muted); }
  .row label { min-width:62px; }
  input[type=range] { flex:1; accent-color:var(--accent); }
  button { background:#232830; color:var(--fg); border:1px solid #2f353e; border-radius:6px;
           padding:5px 10px; font-size:11px; cursor:pointer; }
  button:hover { background:#2b313a; }
  #help { position:fixed; bottom:14px; left:14px; color:var(--muted); font-size:11px;
          background:rgba(24,27,32,.86); border:1px solid #2a2f37; border-radius:8px;
          padding:8px 12px; line-height:1.65; }
  kbd { background:#232830; border:1px solid #333a44; border-radius:4px; padding:1px 5px;
        font-size:10px; font-family:ui-monospace,monospace; }
  #stats { position:fixed; top:14px; right:14px; color:var(--muted); font-size:11px;
           background:rgba(24,27,32,.86); border:1px solid #2a2f37; border-radius:8px;
           padding:8px 12px; font-family:ui-monospace,monospace; }
  #loading { position:fixed; inset:0; display:grid; place-items:center; background:var(--bg);
             font-size:13px; color:var(--muted); z-index:10; }
</style>
</head>
<body>
<div id="loading">decoding __NPOINTS__ points…</div>
<canvas id="c"></canvas>

<div id="hud">
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="row"><label>Point size</label>
    <input id="ps" type="range" min="0.5" max="8" step="0.1" value="__POINTSIZE__"></div>
  <div class="row"><label>Brightness</label>
    <input id="br" type="range" min="0.5" max="2.5" step="0.05" value="1"></div>
  <div class="row" style="gap:6px">
    <button id="reset">Reset view</button>
    <button id="bg">Light / dark</button>
    <button id="spin">Auto-spin</button>
  </div>
</div>

<div id="stats"><span id="fps">–</span> fps · <span id="np">–</span> pts</div>

<div id="help">
  <kbd>drag</kbd> orbit &nbsp; <kbd>wheel</kbd> zoom &nbsp;
  <kbd>right-drag</kbd> pan &nbsp; <kbd>R</kbd> reset
</div>

<script>
const POS_B64 = "__POS_B64__";
const COL_B64 = "__COL_B64__";
const N = __NPOINTS__;

function b64ToBytes(b64) {
  const bin = atob(b64), len = bin.length, out = new Uint8Array(len);
  for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// --- minimal 4x4 matrix maths (column-major, as WebGL expects) -------------
function mat4() { const m = new Float32Array(16); m[0]=m[5]=m[10]=m[15]=1; return m; }
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far), m = new Float32Array(16);
  m[0]=f/aspect; m[5]=f; m[10]=(far+near)*nf; m[11]=-1; m[14]=2*far*near*nf; return m;
}
function lookAt(eye, target, up) {
  const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x), m = mat4();
  m[0]=x[0]; m[4]=x[1]; m[8]=x[2];
  m[1]=y[0]; m[5]=y[1]; m[9]=y[2];
  m[2]=z[0]; m[6]=z[1]; m[10]=z[2];
  m[12]=-dot(x,eye); m[13]=-dot(y,eye); m[14]=-dot(z,eye); m[15]=1;
  return m;
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+j] * b[i*4+k];
    o[i*4+j] = s;
  }
  return o;
}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function norm(v){const l=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/l,v[1]/l,v[2]/l];}

const VS = `
attribute vec3 aPos; attribute vec3 aCol;
uniform mat4 uMVP; uniform float uSize; uniform float uBright; uniform float uPxScale;
varying vec3 vCol;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  // Perspective-correct point size: a point of constant world size shrinks
  // with distance. uPxScale carries the viewport height so the result is in
  // pixels regardless of canvas resolution or device pixel ratio.
  gl_PointSize = clamp(uSize * uPxScale / max(gl_Position.w, 0.15), 1.0, 64.0);
  vCol = aCol * uBright;
}`;

const FS = `
precision mediump float; varying vec3 vCol;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;   // round points instead of squares
  gl_FragColor = vec4(vCol, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

(function main() {
  const positions = new Float32Array(b64ToBytes(POS_B64).buffer);
  const colorBytes = b64ToBytes(COL_B64);

  const canvas = document.getElementById('c');
  const gl = canvas.getContext('webgl', {antialias: true, alpha: false});
  if (!gl) { document.getElementById('loading').textContent = 'WebGL unavailable'; return; }

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog); gl.useProgram(prog);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

  const colBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, colorBytes, gl.STATIC_DRAW);
  const aCol = gl.getAttribLocation(prog, 'aCol');
  gl.enableVertexAttribArray(aCol);
  gl.vertexAttribPointer(aCol, 3, gl.UNSIGNED_BYTE, true, 0, 0);   // normalise 0-255 -> 0-1

  const uMVP = gl.getUniformLocation(prog, 'uMVP');
  const uSize = gl.getUniformLocation(prog, 'uSize');
  const uBright = gl.getUniformLocation(prog, 'uBright');
  const uPxScale = gl.getUniformLocation(prog, 'uPxScale');
  let pxScale = 1.0;

  gl.enable(gl.DEPTH_TEST);

  const home = {theta: 0.6, phi: 1.15, radius: 3.4, target: [0, 0, 0]};
  let cam = JSON.parse(JSON.stringify(home));
  let pointSize = parseFloat(document.getElementById('ps').value);
  let bright = 1.0, dark = true, spin = false;

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
    // A point of world-radius r projects to r * height / (2 w tan(fov/2)) px.
    // 0.0022 approximates the world spacing of a few-hundred-thousand-point
    // cloud normalised to a 2-unit box, so uSize reads as "pixels-ish" at the
    // default zoom.
    pxScale = canvas.height * 0.0022;
  }
  window.addEventListener('resize', resize); resize();

  let dragging = false, button = 0, lastX = 0, lastY = 0;
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('mousedown', e => {
    dragging = true; button = e.button; lastX = e.clientX; lastY = e.clientY;
    canvas.classList.add('dragging');
  });
  window.addEventListener('mouseup', () => { dragging = false; canvas.classList.remove('dragging'); });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (button === 2 || e.shiftKey) {
      // Pan in the camera's own screen plane, scaled by distance.
      const s = cam.radius * 0.0016;
      const st = Math.sin(cam.theta), ct = Math.cos(cam.theta);
      cam.target[0] -= (ct * dx - 0) * s;
      cam.target[2] -= (st * dx) * s;
      cam.target[1] += dy * s;
    } else {
      cam.theta -= dx * 0.006;
      cam.phi = Math.max(0.02, Math.min(Math.PI - 0.02, cam.phi - dy * 0.006));
    }
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    cam.radius = Math.max(0.15, Math.min(30, cam.radius * (1 + Math.sign(e.deltaY) * 0.11)));
  }, {passive: false});

  document.getElementById('ps').addEventListener('input', e => pointSize = parseFloat(e.target.value));
  document.getElementById('br').addEventListener('input', e => bright = parseFloat(e.target.value));
  document.getElementById('reset').onclick = () => { cam = JSON.parse(JSON.stringify(home)); };
  document.getElementById('spin').onclick = () => { spin = !spin; };
  document.getElementById('bg').onclick = () => {
    dark = !dark;
    document.documentElement.style.setProperty('--bg', dark ? '#0e1013' : '#f2f3f5');
    document.documentElement.style.setProperty('--fg', dark ? '#e8eaed' : '#1b1e22');
  };
  window.addEventListener('keydown', e => {
    if (e.key === 'r' || e.key === 'R') cam = JSON.parse(JSON.stringify(home));
  });

  document.getElementById('np').textContent = N.toLocaleString();
  document.getElementById('loading').remove();

  let frames = 0, last = performance.now();
  function draw() {
    if (spin) cam.theta += 0.0025;
    const eye = [
      cam.target[0] + cam.radius * Math.sin(cam.phi) * Math.sin(cam.theta),
      cam.target[1] + cam.radius * Math.cos(cam.phi),
      cam.target[2] + cam.radius * Math.sin(cam.phi) * Math.cos(cam.theta),
    ];
    const proj = perspective(50 * Math.PI / 180, canvas.width / canvas.height, 0.01, 100);
    const view = lookAt(eye, cam.target, [0, 1, 0]);
    gl.uniformMatrix4fv(uMVP, false, mul(proj, view));
    gl.uniform1f(uSize, pointSize);
    gl.uniform1f(uBright, bright);
    gl.uniform1f(uPxScale, pxScale);

    const c = dark ? 0.055 : 0.95;
    gl.clearColor(c, c, c + 0.012, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, N);

    frames++;
    const now = performance.now();
    if (now - last > 500) {
      document.getElementById('fps').textContent = Math.round(frames * 1000 / (now - last));
      frames = 0; last = now;
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = [f"scene.name={args.scene}"] if args.scene else []
    try:
        cfg = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2
    setup_logging(level="WARNING")

    import open3d as o3d

    try:
        path = _resolve_path(cfg, args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 3

    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        print(f"{path} contains no points", file=sys.stderr)
        return 3
    original = len(cloud.points)
    print(f"Loaded {original:,} points from {path.name}")

    if args.clean:
        from src.pointcloud_analysis import crop_to_percentile, remove_outliers
        cloud = crop_to_percentile(cloud, 1.0, 99.0)
        cloud, dropped = remove_outliers(cloud, method="statistical",
                                         nb_neighbors=20, std_ratio=2.0)
        print(f"  cleaned -> {len(cloud.points):,} points")

    points = np.asarray(cloud.points)
    if cloud.has_colors():
        colors = (np.asarray(cloud.colors) * 255.0).clip(0, 255)
    else:
        colors = np.full((len(points), 3), 180.0)

    if len(points) > args.max_points:
        # Even, deterministic thinning preserves the spatial distribution and
        # keeps the exported file reproducible.
        idx = np.linspace(0, len(points) - 1, args.max_points).astype(int)
        points, colors = points[idx], colors[idx]
        print(f"  subsampled -> {len(points):,} points (--max-points {args.max_points})")

    output = Path(args.output) if args.output else \
        ensure_dir(cfg.visualizations_dir) / f"{cfg.scene.name}_{args.what}_viewer.html"
    subtitle = (f"{len(points):,} of {original:,} points"
                f"{' · cleaned' if args.clean else ''} · {path.name}")
    html = build_html(points, colors, title=f"{cfg.scene.name} · {args.what}",
                      subtitle=subtitle, point_size=args.point_size)
    output.write_text(html, encoding="utf-8")

    print()
    print(f"  viewer written : {output}")
    print(f"  size           : {human_size(output.stat().st_size)}")
    print(f"  open it        : double-click the file, or drag it into a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
