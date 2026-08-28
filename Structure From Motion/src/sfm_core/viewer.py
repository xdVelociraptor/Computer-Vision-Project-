"""Self-contained interactive 3D viewer for a reconstruction.

Emits a single HTML file with the geometry embedded as base64 and a
hand-written WebGL renderer - no external libraries, no server, no network.
Layers (surface mesh, dense points, sparse points, camera poses) can be toggled
independently, and preset views snap the camera to canonical directions.

All layers are transformed by the *same* normalisation (centre on the combined
centroid, scale the largest dimension to 2 units) so they stay registered to
one another while letting the camera use fixed constants instead of depending
on the reconstruction's arbitrary units.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..utils import ensure_dir, get_logger

__all__ = ["build_viewer"]

LOGGER = get_logger()


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _quantize_unit(values: np.ndarray) -> np.ndarray:
    """Pack coordinates already normalised to [-1, 1] into int16.

    WebGL reads normalised SHORT attributes straight back to [-1, 1], so this
    halves the payload for free. At 16 bits the quantisation step is about
    3e-5 of the scene extent - far finer than the reconstruction's own
    accuracy, so nothing visible is lost.
    """
    return np.clip(np.round(np.asarray(values, np.float64) * 32767.0),
                   -32767, 32767).astype(np.int16)


def _quantize_normal(values: np.ndarray) -> np.ndarray:
    """Pack unit normals into int8; 8 bits per axis is ample for shading."""
    return np.clip(np.round(np.asarray(values, np.float64) * 127.0), -127, 127).astype(np.int8)


def estimate_up_axis(camera_centers: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
    """Estimate world "up" from the camera path.

    SfM fixes the world frame arbitrarily - the first camera defines it - so a
    reconstruction has no notion of vertical and renders upside down as often
    as not. A handheld capture is carried at roughly constant height, so the
    camera centres are approximately coplanar and that plane is close to
    horizontal; its normal is therefore a good estimate of up. The sign
    ambiguity is resolved by requiring the cameras to sit *above* the scene.
    """
    centres = np.asarray(camera_centers, dtype=np.float64)
    if len(centres) < 3:
        return np.array([0.0, -1.0, 0.0])
    _, _, Vt = np.linalg.svd(centres - centres.mean(axis=0))
    up = Vt[-1]
    if np.dot(centres.mean(axis=0) - np.asarray(scene_points).mean(axis=0), up) < 0:
        up = -up
    return up / max(np.linalg.norm(up), 1e-12)


def rotation_to_up(up: np.ndarray, target: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Rotation matrix taking ``up`` onto ``target`` (Rodrigues, minimal angle)."""
    a = np.asarray(up, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(target, dtype=np.float64)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _frustum_lines(centers: np.ndarray, rotations: np.ndarray, K: np.ndarray,
                   width: int, height: int, depth: float) -> np.ndarray:
    """Line-segment endpoints for camera frustums, as an (M, 3) array of pairs."""
    K_inv = np.linalg.inv(K)
    corners_px = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
                          dtype=np.float64)
    segments = []
    for C, R in zip(centers, rotations):
        rays = corners_px @ K_inv.T
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) * depth
        world = rays @ R + C          # R^T applied on the right
        for i in range(4):
            segments.append(C); segments.append(world[i])
        for i in range(4):
            segments.append(world[i]); segments.append(world[(i + 1) % 4])
    return np.array(segments, dtype=np.float64) if segments else np.zeros((0, 3))


def build_viewer(
    output: Path,
    *,
    title: str,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    dense_points: Optional[np.ndarray] = None,
    dense_colors: Optional[np.ndarray] = None,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    mesh_colors: Optional[np.ndarray] = None,
    camera_centers: Optional[np.ndarray] = None,
    camera_rotations: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (1600, 1200),
    align_up: bool = True,
    max_dense_points: int = 260000,
    max_mesh_faces: int = 220000,
    subtitle: str = "Sparse map, camera poses, and full-resolution geometry",
) -> Path:
    """Write the interactive viewer. Returns the path."""
    output = Path(output)
    ensure_dir(output.parent)

    # ---- shared normalisation ------------------------------------------
    reference = [sparse_points] if len(sparse_points) else []
    if dense_points is not None and len(dense_points):
        reference.append(dense_points)
    if not reference:
        raise ValueError("Nothing to display: no sparse or dense points")
    stacked = np.vstack(reference)
    lo = np.percentile(stacked, 1.0, axis=0)
    hi = np.percentile(stacked, 99.0, axis=0)
    centre = (lo + hi) / 2.0
    scale = float(np.max(hi - lo)) / 2.0
    if scale <= 0:
        scale = 1.0

    # Bring the scene upright before normalising, so every layer shares one
    # gravity-aligned frame and the viewer's +Y really is up.
    R_up = np.eye(3)
    if align_up and camera_centers is not None and len(camera_centers) >= 3:
        R_up = rotation_to_up(estimate_up_axis(np.asarray(camera_centers), stacked))
        LOGGER.info("Aligned reconstruction to gravity (rotated %.1f deg)",
                    np.degrees(np.arccos(np.clip((np.trace(R_up) - 1) / 2, -1, 1))))

    def norm(points: np.ndarray) -> np.ndarray:
        centred = (np.asarray(points, dtype=np.float64) - centre) @ R_up.T
        return (centred / scale).astype(np.float32)

    payload: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    sp = norm(sparse_points)
    payload["sparsePos"] = _b64(_quantize_unit(sp))
    payload["sparseCol"] = _b64(np.asarray(sparse_colors, np.uint8))
    counts["sparse"] = len(sp)

    if dense_points is not None and len(dense_points):
        pts, cols = np.asarray(dense_points), np.asarray(dense_colors, np.uint8)
        if len(pts) > max_dense_points:
            sel = np.linspace(0, len(pts) - 1, max_dense_points).astype(int)
            pts, cols = pts[sel], cols[sel]
        payload["densePos"] = _b64(_quantize_unit(norm(pts)))
        payload["denseCol"] = _b64(cols)
        counts["dense"] = len(pts)
    else:
        payload["densePos"] = payload["denseCol"] = ""
        counts["dense"] = 0

    if mesh_vertices is not None and mesh_faces is not None and len(mesh_faces):
        verts, faces = np.asarray(mesh_vertices), np.asarray(mesh_faces, np.int32)
        cols = (np.asarray(mesh_colors, np.uint8) if mesh_colors is not None
                else np.full((len(verts), 3), 190, np.uint8))
        if len(faces) > max_mesh_faces:
            LOGGER.warning("Mesh has %d faces; evenly sampling %d for the viewer "
                           "(surface.ply keeps the full mesh)",
                           len(faces), max_mesh_faces)
            # Evenly spaced, not the first N: marching-cubes output is ordered
            # by grid position, so taking a prefix would embed one slab of the
            # scene instead of a thinned version of all of it.
            faces = faces[np.linspace(0, len(faces) - 1, max_mesh_faces).astype(int)]
            used = np.unique(faces)
            remap = np.full(len(verts), -1, np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            faces = remap[faces]
            verts, cols = verts[used], cols[used]
        # Per-vertex normals from area-weighted face normals, so the surface
        # shades smoothly instead of showing every triangle.
        vn = np.zeros_like(verts, dtype=np.float64)
        tri = verts[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(vn, faces[:, k], fn)
        lens = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = vn / np.maximum(lens, 1e-12)
        payload["meshPos"] = _b64(_quantize_unit(norm(verts)))
        payload["meshNrm"] = _b64(_quantize_normal(vn))
        payload["meshCol"] = _b64(cols)
        payload["meshIdx"] = _b64(faces.astype(np.uint32))
        counts["meshFaces"] = len(faces)
    else:
        payload["meshPos"] = payload["meshNrm"] = payload["meshCol"] = payload["meshIdx"] = ""
        counts["meshFaces"] = 0

    if camera_centers is not None and len(camera_centers) and camera_rotations is not None:
        K = intrinsics if intrinsics is not None else np.array([[1000., 0, image_size[0] / 2],
                                                                [0, 1000., image_size[1] / 2],
                                                                [0, 0, 1.]])
        depth = float(np.max(hi - lo)) * 0.035
        segs = _frustum_lines(np.asarray(camera_centers), np.asarray(camera_rotations),
                              np.asarray(K), image_size[0], image_size[1], depth)
        payload["camPos"] = _b64(_quantize_unit(norm(segs)))
        counts["cameras"] = len(camera_centers)
        # A polyline through the centres, in capture order.
        path_pts = norm(np.asarray(camera_centers))
        pairs = np.empty((max(0, (len(path_pts) - 1) * 2), 3), np.float32)
        for i in range(len(path_pts) - 1):
            pairs[2 * i] = path_pts[i]
            pairs[2 * i + 1] = path_pts[i + 1]
        payload["pathPos"] = _b64(_quantize_unit(pairs))
        counts["pathSegs"] = len(pairs)
    else:
        payload["camPos"] = payload["pathPos"] = ""
        counts["cameras"] = counts["pathSegs"] = 0

    html = _TEMPLATE
    for key, value in payload.items():
        html = html.replace(f"__{key.upper()}__", value)
    html = (html.replace("__TITLE__", title)
                .replace("__SUBTITLE__", subtitle)
                .replace("__N_SPARSE__", str(counts["sparse"]))
                .replace("__N_DENSE__", str(counts["dense"]))
                .replace("__N_MESHFACES__", str(counts["meshFaces"]))
                .replace("__N_CAMERAS__", str(counts["cameras"]))
                .replace("__N_PATHSEGS__", str(counts["pathSegs"])))
    output.write_text(html, encoding="utf-8")
    LOGGER.info("Viewer written: %s (%d sparse, %d dense, %d faces, %d cameras)",
                output, counts["sparse"], counts["dense"], counts["meshFaces"],
                counts["cameras"])
    return output


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Interactive Reconstruction</title>
<style>
  :root{--bg:#12100c;--panel:#1b1813;--fg:#efe9dd;--muted:#8d8578;--accent:#e8a33d;--line:#2e2921;}
  *{box-sizing:border-box;}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
  .wrap{max-width:1180px;margin:0 auto;padding:26px 22px 14px;}
  h1{font-family:ui-serif,Georgia,"Times New Roman",serif;font-weight:600;
     font-size:24px;margin:0;display:inline;letter-spacing:.2px;}
  .sub{color:var(--muted);font-size:12.5px;margin-left:12px;}
  .views{margin:16px 0 10px;display:flex;gap:7px;}
  .views button{background:transparent;color:var(--fg);border:1px solid var(--line);
    border-radius:5px;padding:5px 13px;font-size:12px;cursor:pointer;transition:.12s;}
  .views button:hover{border-color:#4a4235;}
  .views button.active{background:var(--accent);border-color:var(--accent);color:#1a1409;font-weight:600;}
  .layers{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin-bottom:14px;}
  .layer{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--fg);cursor:pointer;
    user-select:none;}
  .sw{width:26px;height:14px;border-radius:8px;background:#3a3428;position:relative;transition:.15s;
    flex:none;}
  .sw::after{content:"";position:absolute;top:2px;left:2px;width:10px;height:10px;border-radius:50%;
    background:#8d8578;transition:.15s;}
  .layer.on .sw{background:var(--accent);}
  .layer.on .sw::after{left:14px;background:#1a1409;}
  .layer.disabled{opacity:.32;cursor:not-allowed;}
  #stage{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#100e0a;
    position:relative;}
  canvas{display:block;width:100%;height:560px;cursor:grab;}
  canvas.drag{cursor:grabbing;}
  .foot{color:var(--muted);font-size:11.5px;margin-top:11px;letter-spacing:.15px;}
  .foot b{color:#c9c0b0;font-weight:500;}
  #load{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);
    font-size:13px;background:#100e0a;z-index:5;}
  #err{position:absolute;left:12px;top:12px;color:#e07a5f;font-size:12px;display:none;}
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1><span class="sub">__SUBTITLE__</span>

  <div class="views">
    <button data-view="home" class="active">Home</button>
    <button data-view="top">Top</button>
    <button data-view="front">Front</button>
    <button data-view="side">Side</button>
  </div>

  <div class="layers">
    <div class="layer" id="L-mesh"><span class="sw"></span>Surface</div>
    <div class="layer on" id="L-dense"><span class="sw"></span>Dense points</div>
    <div class="layer on" id="L-sparse"><span class="sw"></span>Sparse points</div>
    <div class="layer" id="L-cam"><span class="sw"></span>Camera poses</div>
  </div>

  <div id="stage">
    <div id="load">decoding geometry…</div>
    <div id="err"></div>
    <canvas id="c"></canvas>
  </div>

  <div class="foot">
    <b>__N_CAMERAS__</b> camera poses &nbsp;·&nbsp; <b>__N_SPARSE__</b> sparse points
    &nbsp;·&nbsp; <b>__N_DENSE__</b> dense points &nbsp;·&nbsp; <b>__N_MESHFACES__</b> faces
    &nbsp;·&nbsp; Drag to orbit &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Right-drag to pan
  </div>
</div>

<script>
const DATA = {
  sparsePos:"__SPARSEPOS__", sparseCol:"__SPARSECOL__",
  densePos:"__DENSEPOS__",   denseCol:"__DENSECOL__",
  meshPos:"__MESHPOS__", meshNrm:"__MESHNRM__", meshCol:"__MESHCOL__", meshIdx:"__MESHIDX__",
  camPos:"__CAMPOS__", pathPos:"__PATHPOS__",
};
const N = {sparse:__N_SPARSE__, dense:__N_DENSE__, meshFaces:__N_MESHFACES__,
           cameras:__N_CAMERAS__, pathSegs:__N_PATHSEGS__};

function bytes(b64){ if(!b64) return new Uint8Array(0);
  const s=atob(b64), n=s.length, a=new Uint8Array(n);
  for(let i=0;i<n;i++) a[i]=s.charCodeAt(i); return a; }
const i16 = b64 => new Int16Array(bytes(b64).buffer);
const i8  = b64 => new Int8Array(bytes(b64).buffer);
const u32 = b64 => new Uint32Array(bytes(b64).buffer);

// ---- matrices (column-major) ---------------------------------------------
function perspective(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far);
  const m=new Float32Array(16); m[0]=f/aspect;m[5]=f;m[10]=(far+near)*nf;m[11]=-1;
  m[14]=2*far*near*nf; return m;}
function lookAt(e,t,u){const z=nrm(sub(e,t)),x=nrm(cr(u,z)),y=cr(z,x);const m=new Float32Array(16);
  m[0]=x[0];m[4]=x[1];m[8]=x[2];m[1]=y[0];m[5]=y[1];m[9]=y[2];m[2]=z[0];m[6]=z[1];m[10]=z[2];
  m[12]=-dt(x,e);m[13]=-dt(y,e);m[14]=-dt(z,e);m[15]=1;return m;}
function mul(a,b){const o=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}
  return o;}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dt=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cr=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=v=>{const l=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/l,v[1]/l,v[2]/l];};

const VS_PT=`attribute vec3 aPos;attribute vec3 aCol;uniform mat4 uMVP;
uniform float uSize,uPx,uBright;varying vec3 vCol;
void main(){gl_Position=uMVP*vec4(aPos,1.0);
gl_PointSize=clamp(uSize*uPx/max(gl_Position.w,0.12),1.0,48.0);vCol=aCol*uBright;}`;
const FS_PT=`precision mediump float;varying vec3 vCol;
void main(){vec2 d=gl_PointCoord-vec2(0.5);if(dot(d,d)>0.25)discard;gl_FragColor=vec4(vCol,1.0);}`;

const VS_MESH=`attribute vec3 aPos;attribute vec3 aNrm;attribute vec3 aCol;
uniform mat4 uMVP;varying vec3 vCol;varying vec3 vN;
void main(){gl_Position=uMVP*vec4(aPos,1.0);vN=aNrm;vCol=aCol;}`;
const FS_MESH=`precision mediump float;varying vec3 vCol;varying vec3 vN;
void main(){vec3 n=normalize(vN);
 float d=max(dot(n,normalize(vec3(0.4,0.75,0.55))),0.0);
 float b=max(dot(n,normalize(vec3(-0.5,-0.3,-0.6))),0.0);
 gl_FragColor=vec4(vCol*(0.34+0.72*d+0.18*b),1.0);}`;

const VS_LINE=`attribute vec3 aPos;uniform mat4 uMVP;void main(){gl_Position=uMVP*vec4(aPos,1.0);}`;
const FS_LINE=`precision mediump float;uniform vec3 uCol;void main(){gl_FragColor=vec4(uCol,1.0);}`;

function prog(gl,vs,fs){
  function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;}
  const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));
  gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
  return p;}
function buf(gl,data,target){const b=gl.createBuffer();
  target=target||gl.ARRAY_BUFFER;gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b;}

(function(){
const canvas=document.getElementById('c');
let gl;
try{ gl=canvas.getContext('webgl',{antialias:true,alpha:false,depth:true}); }catch(e){}
if(!gl){document.getElementById('load').textContent='WebGL is unavailable in this browser.';return;}

let P_PT,P_MESH,P_LINE;
try{ P_PT=prog(gl,VS_PT,FS_PT); P_MESH=prog(gl,VS_MESH,FS_MESH); P_LINE=prog(gl,VS_LINE,FS_LINE); }
catch(e){ document.getElementById('err').style.display='block';
          document.getElementById('err').textContent='Shader error: '+e.message; }

const G={};
G.sparse={pos:buf(gl,i16(DATA.sparsePos)),col:buf(gl,bytes(DATA.sparseCol)),n:N.sparse};
if(N.dense) G.dense={pos:buf(gl,i16(DATA.densePos)),col:buf(gl,bytes(DATA.denseCol)),n:N.dense};
if(N.meshFaces){const idx=u32(DATA.meshIdx);
  G.mesh={pos:buf(gl,i16(DATA.meshPos)),nrm:buf(gl,i8(DATA.meshNrm)),
          col:buf(gl,bytes(DATA.meshCol)),
          idx:buf(gl,idx,gl.ELEMENT_ARRAY_BUFFER),n:idx.length};
  G.mesh.uint = !!gl.getExtension('OES_element_index_uint');}
if(N.cameras){const cp=i16(DATA.camPos);
  G.cam={pos:buf(gl,cp),n:cp.length/3};
  const pp=i16(DATA.pathPos); G.path={pos:buf(gl,pp),n:pp.length/3};}

const show={mesh:false,dense:N.dense>0,sparse:true,cam:false};
const HOME={theta:0.62,phi:1.12,radius:3.3};
const PRESETS={home:HOME,top:{theta:0.0,phi:0.06,radius:3.1},
               front:{theta:0.0,phi:1.5708,radius:3.1},side:{theta:1.5708,phi:1.5708,radius:3.1}};
let cam={...HOME,target:[0,0,0]};

function setLayer(id,key,available){
  const el=document.getElementById(id);
  if(!available){el.classList.add('disabled');el.classList.remove('on');show[key]=false;return;}
  el.classList.toggle('on',show[key]);
  el.onclick=()=>{show[key]=!show[key];el.classList.toggle('on',show[key]);};
}
setLayer('L-mesh','mesh',N.meshFaces>0);
setLayer('L-dense','dense',N.dense>0);
setLayer('L-sparse','sparse',N.sparse>0);
setLayer('L-cam','cam',N.cameras>0);

document.querySelectorAll('.views button').forEach(b=>{
  b.onclick=()=>{document.querySelectorAll('.views button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');const p=PRESETS[b.dataset.view];
    cam={...cam,theta:p.theta,phi:p.phi,radius:p.radius,target:[0,0,0]};};});

let px=1;
function resize(){const dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=canvas.clientWidth*dpr;canvas.height=canvas.clientHeight*dpr;
  gl.viewport(0,0,canvas.width,canvas.height);px=canvas.height*0.0016;}
window.addEventListener('resize',resize);resize();

let drag=false,btn=0,lx=0,ly=0;
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('mousedown',e=>{drag=true;btn=e.button;lx=e.clientX;ly=e.clientY;
  canvas.classList.add('drag');});
window.addEventListener('mouseup',()=>{drag=false;canvas.classList.remove('drag');});
window.addEventListener('mousemove',e=>{if(!drag)return;
  const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
  if(btn===2||e.shiftKey){const s=cam.radius*0.0018,ct=Math.cos(cam.theta),st=Math.sin(cam.theta);
    cam.target[0]-=ct*dx*s;cam.target[2]+=st*dx*s;cam.target[1]+=dy*s;}
  else{cam.theta-=dx*0.0062;cam.phi=Math.max(0.03,Math.min(Math.PI-0.03,cam.phi-dy*0.0062));}});
canvas.addEventListener('wheel',e=>{e.preventDefault();
  cam.radius=Math.max(0.25,Math.min(18,cam.radius*(1+Math.sign(e.deltaY)*0.1)));},{passive:false});

function bindAttr(p,name,b,size,type,normalized){
  const loc=gl.getAttribLocation(p,name); if(loc<0)return;
  gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,size,type,!!normalized,0,0);}

function drawPoints(g,mvp,size,bright){
  gl.useProgram(P_PT);
  gl.uniformMatrix4fv(gl.getUniformLocation(P_PT,'uMVP'),false,mvp);
  gl.uniform1f(gl.getUniformLocation(P_PT,'uSize'),size);
  gl.uniform1f(gl.getUniformLocation(P_PT,'uPx'),px);
  gl.uniform1f(gl.getUniformLocation(P_PT,'uBright'),bright);
  bindAttr(P_PT,'aPos',g.pos,3,gl.SHORT,true);
  bindAttr(P_PT,'aCol',g.col,3,gl.UNSIGNED_BYTE,true);
  gl.drawArrays(gl.POINTS,0,g.n);}

function drawLines(g,mvp,col,width){
  gl.useProgram(P_LINE);
  gl.uniformMatrix4fv(gl.getUniformLocation(P_LINE,'uMVP'),false,mvp);
  gl.uniform3fv(gl.getUniformLocation(P_LINE,'uCol'),col);
  bindAttr(P_LINE,'aPos',g.pos,3,gl.SHORT,true);
  gl.lineWidth(width||1);
  gl.drawArrays(gl.LINES,0,g.n);}

document.getElementById('load').remove();

function frame(){
  const eye=[cam.target[0]+cam.radius*Math.sin(cam.phi)*Math.sin(cam.theta),
             cam.target[1]+cam.radius*Math.cos(cam.phi),
             cam.target[2]+cam.radius*Math.sin(cam.phi)*Math.cos(cam.theta)];
  const proj=perspective(48*Math.PI/180,canvas.width/canvas.height,0.01,80);
  const mvp=mul(proj,lookAt(eye,cam.target,[0,1,0]));

  gl.clearColor(0.063,0.055,0.039,1);
  gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  if(show.mesh&&G.mesh){
    gl.useProgram(P_MESH);
    gl.uniformMatrix4fv(gl.getUniformLocation(P_MESH,'uMVP'),false,mvp);
    bindAttr(P_MESH,'aPos',G.mesh.pos,3,gl.SHORT,true);
    bindAttr(P_MESH,'aNrm',G.mesh.nrm,3,gl.BYTE,true);
    bindAttr(P_MESH,'aCol',G.mesh.col,3,gl.UNSIGNED_BYTE,true);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,G.mesh.idx);
    if(G.mesh.uint) gl.drawElements(gl.TRIANGLES,G.mesh.n,gl.UNSIGNED_INT,0);
  }
  if(show.dense&&G.dense) drawPoints(G.dense,mvp,1.7,1.0);
  if(show.sparse&&G.sparse) drawPoints(G.sparse,mvp,2.6,1.25);
  if(show.cam&&G.cam){ drawLines(G.cam,mvp,[0.35,0.68,1.0],1);
                       if(G.path) drawLines(G.path,mvp,[0.91,0.64,0.24],2); }
  requestAnimationFrame(frame);
}
frame();
})();
</script>
</body>
</html>
"""
