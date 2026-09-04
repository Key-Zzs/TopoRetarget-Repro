"""Self-contained OakInk2 scene-graph HTML viewer (Viewer V2).

The browser receives Python-precomputed vertices, faces, joints, and object
poses. A single camera view matrix projects an immutable ``SCENE_WORLD``;
pointer orbit and wheel zoom never modify a scene-node model transform.
"""

# ruff: noqa: E501

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JOINT_PARENTS: tuple[int | None, ...] = (
    None,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
)


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return finite, unit, area-weighted normals for a fixed mesh topology."""
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if values.ndim != 3 or values.shape[2] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("OAKINK2_VIEWER_MESH_SHAPE_INVALID")
    normal = np.cross(
        values[:, triangles[:, 1]] - values[:, triangles[:, 0]],
        values[:, triangles[:, 2]] - values[:, triangles[:, 0]],
    )
    result = np.zeros_like(values)
    for column in range(3):
        np.add.at(result, (slice(None), triangles[:, column]), normal)  # type: ignore[arg-type]
    result /= np.maximum(np.linalg.norm(result, axis=2, keepdims=True), 1e-12)
    return result.astype(np.float32)


@dataclass(frozen=True)
class OakInk2HTMLViewerV2Data:
    """Python-precomputed arrays consumed by the V2 viewer."""

    frames: np.ndarray
    hand_vertices_world: np.ndarray
    hand_vertices_anatomy: np.ndarray
    hand_faces_closed: np.ndarray
    hand_faces_open: np.ndarray
    hand_joints_world: np.ndarray
    hand_joints_anatomy: np.ndarray
    object_vertices: np.ndarray
    object_faces: np.ndarray
    object_transforms: np.ndarray
    primary_frame: int
    record: dict[str, Any]
    camera_presets: dict[str, Any]


# Compatibility for the O1R2-C evidence generator. Both names render V2.
TrustedHTMLViewerData = OakInk2HTMLViewerV2Data


def _validate(data: OakInk2HTMLViewerV2Data) -> None:
    frame_count = len(data.frames)
    expected = (frame_count, 778, 3)
    for value in (data.hand_vertices_world, data.hand_vertices_anatomy):
        if np.asarray(value).shape != expected or not np.isfinite(value).all():
            raise ValueError("OAKINK2_VIEWER_HAND_GEOMETRY_INVALID")
    for value in (data.hand_joints_world, data.hand_joints_anatomy):
        if np.asarray(value).shape != (frame_count, 21, 3) or not np.isfinite(value).all():
            raise ValueError("OAKINK2_VIEWER_JOINT_GEOMETRY_INVALID")
    if int(data.primary_frame) not in set(np.asarray(data.frames, dtype=np.int64).tolist()):
        raise ValueError("OAKINK2_VIEWER_PRIMARY_FRAME_ABSENT")
    for faces in (data.hand_faces_closed, data.hand_faces_open, data.object_faces):
        array = np.asarray(faces)
        if array.ndim != 2 or array.shape[1] != 3 or array.min() < 0:
            raise ValueError("OAKINK2_VIEWER_FACE_TOPOLOGY_INVALID")
    if np.asarray(data.hand_faces_closed).max() >= 778:
        raise ValueError("OAKINK2_VIEWER_HAND_FACE_INDEX_INVALID")
    if np.asarray(data.object_faces).max() >= len(data.object_vertices):
        raise ValueError("OAKINK2_VIEWER_OBJECT_FACE_INDEX_INVALID")
    if len(data.object_vertices) > np.iinfo(np.uint16).max:
        raise ValueError("OAKINK2_VIEWER_OBJECT_UINT16_LIMIT")
    transforms = np.asarray(data.object_transforms)
    if transforms.shape != (frame_count, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("OAKINK2_VIEWER_OBJECT_TRACK_INVALID")
    for name in ("FRONT", "OBLIQUE", "SIDE"):
        preset = data.camera_presets.get(name)
        if not isinstance(preset, dict):
            raise ValueError(f"OAKINK2_VIEWER_CAMERA_PRESET_MISSING:{name}")
        for key in ("anatomy_camera_model_matrix", "anatomy_projection_matrix"):
            values = np.asarray(preset.get(key), dtype=np.float64)
            if values.shape != (16,) or not np.isfinite(values).all():
                raise ValueError(f"OAKINK2_VIEWER_CAMERA_MATRIX_INVALID:{name}:{key}")


def payload_for(data: OakInk2HTMLViewerV2Data) -> dict[str, Any]:
    """Serialize immutable scene nodes and their Python-computed frame data."""
    _validate(data)
    frames = np.asarray(data.frames, dtype=np.int32)
    primary_index = int(np.where(frames == int(data.primary_frame))[0][0])
    hand_world = np.asarray(data.hand_vertices_world, dtype=np.float32)
    hand_scene = np.asarray(data.hand_vertices_anatomy, dtype=np.float32)
    joints_scene = np.asarray(data.hand_joints_anatomy, dtype=np.float32)
    closed_faces = np.asarray(data.hand_faces_closed, dtype=np.uint16)
    open_faces = np.asarray(data.hand_faces_open, dtype=np.uint16)
    object_vertices = np.asarray(data.object_vertices, dtype=np.float32)
    object_faces = np.asarray(data.object_faces, dtype=np.uint16)
    transforms = np.asarray(data.object_transforms, dtype=np.float32)

    # SCENE_WORLD is MANO-root-relative. The browser only selects these
    # Python-computed per-frame object model matrices.
    hand_root_world = hand_world[:, 0, :] - hand_scene[:, 0, :]
    object_models_scene = transforms.copy()
    object_models_scene[:, :3, 3] -= hand_root_world
    object_center = np.append(object_vertices.mean(axis=0), 1.0)
    object_center_scene = (object_models_scene[primary_index] @ object_center)[:3]
    hand_center_scene = joints_scene[primary_index].mean(axis=0)
    interaction_center = (hand_center_scene + object_center_scene) * 0.5
    closed_normals = vertex_normals(hand_scene, closed_faces)
    open_normals = vertex_normals(hand_scene, open_faces)
    return {
        "schemaVersion": "OakInk2HTMLViewerV2",
        "sceneFrame": "SCENE_WORLD_MANO_ROOT_RELATIVE",
        "frames": frames.tolist(),
        "primaryFrameIndex": primary_index,
        "handScene": _b64(hand_scene),
        "handShape": list(hand_scene.shape),
        "handTrianglesClosedScene": _b64(hand_scene[:, closed_faces].reshape(len(frames), -1, 3)),
        "handTriangleNormalsClosedScene": _b64(
            closed_normals[:, closed_faces].reshape(len(frames), -1, 3)
        ),
        "handTrianglesOpenScene": _b64(hand_scene[:, open_faces].reshape(len(frames), -1, 3)),
        "handTriangleNormalsOpenScene": _b64(
            open_normals[:, open_faces].reshape(len(frames), -1, 3)
        ),
        "handFacesClosed": _b64(closed_faces),
        "handFacesOpen": _b64(open_faces),
        "object": _b64(object_vertices),
        "objectNormals": _b64(vertex_normals(object_vertices[None], object_faces)[0]),
        "objectFaces": _b64(object_faces),
        "objectModelsScene": _b64(object_models_scene),
        "jointsScene": _b64(joints_scene),
        "jointParents": [-1 if parent is None else parent for parent in JOINT_PARENTS],
        "focusPivots": {
            "FOCUS_HAND": joints_scene[primary_index, 0].astype(float).tolist(),
            "FOCUS_INTERACTION": interaction_center.astype(float).tolist(),
        },
        "record": data.record,
        "cameraPresets": data.camera_presets,
        "contracts": {
            "geometryAuthority": "PYTHON_PRECOMPUTED_ONLY",
            "cameraAuthority": "ViewerCameraStateV1",
            "orbitMutates": "CAMERA_VIEW_MATRIX_ONLY",
            "pan": "PAN_NOT_SUPPORTED_BY_DESIGN",
            "offline": True,
        },
    }


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OakInk2 HTML Viewer V2</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:16px;background:#111820;color:#e8eef4;font:14px system-ui,sans-serif}h1{font-size:20px;margin:0 0 10px}.toolbar{display:flex;flex-wrap:wrap;gap:6px;max-width:1000px;margin:6px 0}button{border:1px solid #536677;border-radius:5px;background:#23313d;color:#e8eef4;padding:6px 9px;cursor:pointer}button.active{background:#176b51;border-color:#35b889}label{display:flex;align-items:center;gap:5px}#frame{width:min(500px,90vw)}#stage{width:min(100%,640px);aspect-ratio:1}canvas{display:block;width:100%;height:100%;border:1px solid #586673;background:#0b1016;touch-action:none}pre{max-width:1000px;white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#9eb0bd}.status{font-variant-numeric:tabular-nums}@media print{button{display:none}}</style></head><body>
<h1>OakInk2 Ref2Dex-style HTML Viewer V2</h1><div id="camera" class="toolbar"><button data-preset="FRONT">FRONT</button><button data-preset="OBLIQUE">OBLIQUE</button><button data-preset="SIDE">SIDE</button><button id="reset">RESET CAMERA</button><button data-focus="FOCUS_HAND">FOCUS HAND</button><button data-focus="FOCUS_INTERACTION">FOCUS INTERACTION</button></div>
<div id="visibility" class="toolbar"><button data-mode="HAND_ONLY">HAND ONLY</button><button data-mode="HAND_OBJECT">HAND + OBJECT</button><button data-mode="SKELETON_ONLY">SKELETON ONLY</button><button data-mode="HAND_SKELETON_OBJECT">HAND + SKELETON + OBJECT</button></div><div class="toolbar"><button id="play">Play</button><input id="frame" type="range" min="0" step="1"><label><input id="closed" type="checkbox" checked>closed surface</label></div>
<div id="stage"><canvas id="c" width="640" height="640"></canvas></div><p id="state" class="status"></p><p class="muted">Drag to orbit · Wheel to zoom · SOURCE/CANONICAL FRAME STATUS=IDENTITY · PAN_NOT_SUPPORTED_BY_DESIGN</p><pre id="meta"></pre><pre id="certificate"></pre>
<script>"use strict";const D=__DATA__;
const bytes=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer,F32=s=>new Float32Array(bytes(s)),U16=s=>new Uint16Array(bytes(s));
const HS=F32(D.handScene),HTC=F32(D.handTrianglesClosedScene),HNC=F32(D.handTriangleNormalsClosedScene),HTO=F32(D.handTrianglesOpenScene),HNO=F32(D.handTriangleNormalsOpenScene),HC=U16(D.handFacesClosed),HO=U16(D.handFacesOpen),O=F32(D.object),ON=F32(D.objectNormals),OF=U16(D.objectFaces),OM=F32(D.objectModelsScene),J=F32(D.jointsScene);
const q=new URLSearchParams(location.search),capture=q.has("capture"),canvas=document.querySelector("#c"),slider=document.querySelector("#frame"),stateLabel=document.querySelector("#state"),certificateNode=document.querySelector("#certificate"),gl=canvas.getContext("webgl",{antialias:true,depth:true,alpha:false,preserveDrawingBuffer:true});if(!gl)throw Error("WEBGL_UNAVAILABLE");slider.max=String(D.frames.length-1);
let frame=Math.max(0,Math.min(D.frames.length-1,+(q.get("frameIndex")??D.primaryFrameIndex))),mode=(q.get("mode")??"HAND_OBJECT").toUpperCase(),closed=true,playing=false,drag=null;const camera={schema:"ViewerCameraStateV1",basePreset:(q.get("preset")??"OBLIQUE").toUpperCase(),focus:(q.get("focus")??"FOCUS_INTERACTION").toUpperCase(),yaw:0,pitch:0,distanceScale:1,free:false};if(!D.cameraPresets[camera.basePreset])camera.basePreset="OBLIQUE";if(!D.focusPivots[camera.focus])camera.focus="FOCUS_INTERACTION";
document.querySelector("#meta").textContent=JSON.stringify({record:D.record,contracts:D.contracts},null,2);if(capture){document.querySelectorAll("h1,.toolbar,p,#meta").forEach(x=>x.style.display="none");document.body.style.margin="0";document.body.style.background="#fff";canvas.style.border="0";canvas.style.background="#fff";}
const vs=`attribute vec3 p;attribute vec3 n;uniform mat4 m;uniform mat4 vp;uniform float pointSize;varying vec3 N;void main(){gl_Position=vp*m*vec4(p,1.);gl_PointSize=pointSize;N=mat3(m)*n;}`,fs=`precision mediump float;uniform vec3 color;uniform bool two;uniform bool unlit;varying vec3 N;void main(){float d=1.;if(!unlit){vec3 n=normalize(N);if(two&&!gl_FrontFacing)n=-n;d=.38+.62*max(dot(n,normalize(vec3(.2,.5,1.))),0.);}gl_FragColor=vec4(color*d,1.);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);const ap=gl.getAttribLocation(program,"p"),an=gl.getAttribLocation(program,"n"),um=gl.getUniformLocation(program,"m"),uvp=gl.getUniformLocation(program,"vp"),uc=gl.getUniformLocation(program,"color"),ut=gl.getUniformLocation(program,"two"),uu=gl.getUniformLocation(program,"unlit"),ups=gl.getUniformLocation(program,"pointSize");
const I=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}function T(x,y,z){const m=new Float32Array(I);m[12]=x;m[13]=y;m[14]=z;return m}function RX(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1])}function RY(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1])}function transformPoint(m,v){return[m[0]*v[0]+m[4]*v[1]+m[8]*v[2]+m[12],m[1]*v[0]+m[5]*v[1]+m[9]*v[2]+m[13],m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14]]}function rowModel(values,index){const r=values.subarray(index*16,index*16+16);return new Float32Array([r[0],r[4],r[8],r[12],r[1],r[5],r[9],r[13],r[2],r[6],r[10],r[14],r[3],r[7],r[11],r[15]])}
function cameraMatrices(){const p=D.cameraPresets[camera.basePreset],projection=new Float32Array(p.anatomy_projection_matrix),base=new Float32Array(p.anatomy_camera_model_matrix);let view=base;if(camera.free||camera.yaw!==0||camera.pitch!==0||camera.distanceScale!==1){const pivot=D.focusPivots[camera.focus],pc=transformPoint(base,pivot),orbit=mul(T(pc[0],pc[1],pc[2]),mul(RX(camera.pitch),mul(RY(camera.yaw),T(-pc[0],-pc[1],-pc[2]))));view=mul(T(0,0,(camera.distanceScale-1)*pc[2]),mul(orbit,base));}return{view,projection,viewProjection:mul(projection,view)}}
function buffer(target,data,usage=gl.STATIC_DRAW){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,usage);return b}const hp=buffer(gl.ARRAY_BUFFER,HTC.subarray(0,1),gl.DYNAMIC_DRAW),hn=buffer(gl.ARRAY_BUFFER,HNC.subarray(0,1),gl.DYNAMIC_DRAW),op=buffer(gl.ARRAY_BUFFER,O),on=buffer(gl.ARRAY_BUFFER,ON),oi=buffer(gl.ELEMENT_ARRAY_BUFFER,OF),sp=buffer(gl.ARRAY_BUFFER,new Float32Array(63),gl.DYNAMIC_DRAW),sn=buffer(gl.ARRAY_BUFFER,new Float32Array(63).map((_,i)=>i%3===2?1:0),gl.DYNAMIC_DRAW),slp=buffer(gl.ARRAY_BUFFER,new Float32Array(120),gl.DYNAMIC_DRAW),sln=buffer(gl.ARRAY_BUFFER,new Float32Array(120).map((_,i)=>i%3===2?1:0),gl.DYNAMIC_DRAW);
function bind(b,a){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0)}function draw(p,n,i,count,m,color,two,kind=gl.TRIANGLES,unlit=false,pointSize=1){bind(p,ap);bind(n,an);gl.uniformMatrix4fv(um,false,m);gl.uniform3fv(uc,color);gl.uniform1i(ut,two?1:0);gl.uniform1i(uu,unlit?1:0);gl.uniform1f(ups,pointSize);if(i===null)gl.drawArrays(kind,0,count);else{gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,i);gl.drawElements(kind,count,gl.UNSIGNED_SHORT,0)}}
function sceneFingerprint(values){let sum=0,sumsq=0,min=Infinity,max=-Infinity;for(const x of values){sum+=x;sumsq+=x*x;min=Math.min(min,x);max=Math.max(max,x)}return{count:values.length,sum,sumsq,min,max}}function readback(){const pixels=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);let h=2166136261,green=0,orange=0,foreground=0,x0=canvas.width,y0=canvas.height,x1=-1,y1=-1;for(let i=0;i<pixels.length;i+=4){h^=pixels[i];h=Math.imul(h,16777619);h^=pixels[i+1];h=Math.imul(h,16777619);h^=pixels[i+2];h=Math.imul(h,16777619);if(pixels[i]<245||pixels[i+1]<245||pixels[i+2]<245)foreground++;if(pixels[i+1]>pixels[i]+35&&pixels[i+1]>pixels[i+2]+15){green++;const n=i/4,x=n%canvas.width,y=Math.floor(n/canvas.width);x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y)}if(pixels[i]>pixels[i+1]+45&&pixels[i+1]>pixels[i+2]+15)orange++}return{fnv1a32:(h>>>0).toString(16),foreground_pixels:foreground,green_pixels:green,orange_pixels:orange,green_bbox_gl:green?[x0,y0,x1,y1]:null}}
const landmarkIndices=[0,1,5,9,13,17,4,8,12,16,20];function project(v,m){const x=v[0],y=v[1],z=v[2],X=m[0]*x+m[4]*y+m[8]*z+m[12],Y=m[1]*x+m[5]*y+m[9]*z+m[13],Z=m[2]*x+m[6]*y+m[10]*z+m[14],W=m[3]*x+m[7]*y+m[11]*z+m[15];return[(X/W+1)*canvas.width/2,(1-Y/W)*canvas.height/2,Z/W,W]}
function certify(matrices){const jb=frame*63,landmarksScene=landmarkIndices.map(i=>Array.from(J.subarray(jb+i*3,jb+i*3+3))),objectModel=rowModel(OM,frame),handFrame=HS.subarray(frame*778*3,(frame+1)*778*3),jointFrame=J.subarray(jb,jb+63);return{schema:"OakInk2BrowserCertificateV2",frame_index:frame,mocap_frame_id:D.frames[frame],mode,camera_state:{schema:camera.schema,base_preset:camera.basePreset,display_mode:camera.free?"FREE_ORBIT":camera.basePreset,focus:camera.focus,pivot:D.focusPivots[camera.focus],yaw_deg:camera.yaw*180/Math.PI,pitch_deg:camera.pitch*180/Math.PI,distance_scale:camera.distanceScale},projection_matrix:Array.from(matrices.projection),camera_view_matrix:Array.from(matrices.view),view_projection_matrix:Array.from(matrices.viewProjection),scene_nodes:{hand:{model_matrix:Array.from(I),fingerprint:sceneFingerprint(handFrame)},skeleton:{model_matrix:Array.from(I),fingerprint:sceneFingerprint(jointFrame)},object:{model_matrix:Array.from(objectModel),geometry_fingerprint:sceneFingerprint(O)}},landmark_indices:landmarkIndices,landmarks_scene:landmarksScene,landmarks_camera:landmarksScene.map(v=>transformPoint(matrices.view,v)),landmarks_px:landmarksScene.map(v=>project(v,matrices.viewProjection)),hand_object_anchor:{wrist_scene:landmarksScene[0],object_origin_scene:transformPoint(objectModel,[0,0,0])},framebuffer:readback()}}
function updateSkeleton(){const values=J.subarray(frame*63,frame*63+63),lines=new Float32Array(120);let k=0;for(let child=0;child<D.jointParents.length;child++){const parent=D.jointParents[child];if(parent<0)continue;lines.set(values.subarray(parent*3,parent*3+3),k);k+=3;lines.set(values.subarray(child*3,child*3+3),k);k+=3}gl.bindBuffer(gl.ARRAY_BUFFER,sp);gl.bufferData(gl.ARRAY_BUFFER,values,gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,slp);gl.bufferData(gl.ARRAY_BUFFER,lines,gl.DYNAMIC_DRAW)}
function paint(){const matrices=cameraMatrices(),faceCount=closed?HC.length:HO.length,triangles=closed?HTC:HTO,normals=closed?HNC:HNO,base=frame*faceCount*3;gl.bindBuffer(gl.ARRAY_BUFFER,hp);gl.bufferData(gl.ARRAY_BUFFER,triangles.subarray(base,base+faceCount*3),gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,hn);gl.bufferData(gl.ARRAY_BUFFER,normals.subarray(base,base+faceCount*3),gl.DYNAMIC_DRAW);updateSkeleton();gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(capture?1:.043,capture?1:.063,capture?1:.086,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniformMatrix4fv(uvp,false,matrices.viewProjection);const showHand=mode==="HAND_ONLY"||mode==="HAND_OBJECT"||mode==="HAND_SKELETON_OBJECT",showObject=mode==="HAND_OBJECT"||mode==="HAND_SKELETON_OBJECT",showSkeleton=mode==="SKELETON_ONLY"||mode==="HAND_SKELETON_OBJECT";if(showObject)draw(op,on,oi,OF.length,rowModel(OM,frame),[.95,.40,.08],false);if(showHand)draw(hp,hn,null,faceCount,I,[.12,.75,.48],true);if(showSkeleton){draw(slp,sln,null,40,I,[1,.82,.15],true,gl.LINES,true,1);draw(sp,sn,null,21,I,[1,.9,.25],true,gl.POINTS,true,6)}slider.value=String(frame);document.querySelectorAll("button.active").forEach(x=>x.classList.remove("active"));const pb=document.querySelector(`[data-preset="${camera.free?"":camera.basePreset}"]`);if(pb)pb.classList.add("active");const mb=document.querySelector(`[data-mode="${mode}"]`);if(mb)mb.classList.add("active");const fb=document.querySelector(`[data-focus="${camera.focus}"]`);if(fb)fb.classList.add("active");stateLabel.textContent=`TARGET=${D.record.target_object} · mocap frame ${D.frames[frame]} (${frame+1}/${D.frames.length}) · ${camera.free?"FREE_ORBIT":camera.basePreset} · ${camera.focus} · ${mode}`;const result=certify(matrices);window.__OAKINK2_BROWSER_CERTIFICATE__=result;if(q.has("certify"))certificateNode.textContent=JSON.stringify(result)}
function setPreset(value){if(!D.cameraPresets[value])throw Error(`UNKNOWN_PRESET:${value}`);camera.basePreset=value;camera.yaw=0;camera.pitch=0;camera.distanceScale=1;camera.free=false;paint()}function setFocus(value){if(!D.focusPivots[value])throw Error(`UNKNOWN_FOCUS:${value}`);camera.focus=value;camera.yaw=0;camera.pitch=0;camera.distanceScale=1;camera.basePreset="OBLIQUE";camera.free=false;paint()}function resetCamera(){setPreset("OBLIQUE")}function setMode(value){mode=value;paint()}function setFrame(value){frame=Math.max(0,Math.min(D.frames.length-1,Number(value)));paint()}function setOrbit(yawDeg,pitchDeg,distanceScale=1){camera.yaw=yawDeg*Math.PI/180;camera.pitch=Math.max(-75,Math.min(75,pitchDeg))*Math.PI/180;camera.distanceScale=Math.max(.35,Math.min(3,distanceScale));camera.free=true;paint()}function play(){playing=true;document.querySelector("#play").textContent="Pause"}function pause(){playing=false;document.querySelector("#play").textContent="Play"}
document.querySelectorAll("[data-preset]").forEach(x=>x.onclick=()=>setPreset(x.dataset.preset));document.querySelectorAll("[data-focus]").forEach(x=>x.onclick=()=>setFocus(x.dataset.focus));document.querySelectorAll("[data-mode]").forEach(x=>x.onclick=()=>setMode(x.dataset.mode));document.querySelector("#reset").onclick=resetCamera;document.querySelector("#play").onclick=()=>playing?pause():play();slider.oninput=e=>setFrame(e.target.value);document.querySelector("#closed").onchange=e=>{closed=e.target.checked;paint()};
canvas.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];try{canvas.setPointerCapture(e.pointerId)}catch(_){}});canvas.addEventListener("pointermove",e=>{if(!drag)return;camera.free=true;camera.yaw=(camera.yaw+(e.clientX-drag[0])*.01)%(Math.PI*2);camera.pitch=Math.max(-75*Math.PI/180,Math.min(75*Math.PI/180,camera.pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];paint()});function stopDrag(){drag=null}canvas.addEventListener("pointerup",stopDrag);canvas.addEventListener("pointercancel",stopDrag);canvas.addEventListener("wheel",e=>{e.preventDefault();camera.free=true;camera.distanceScale=Math.max(.35,Math.min(3,camera.distanceScale*Math.exp(e.deltaY*.001)));paint()},{passive:false});
setInterval(()=>{if(playing)setFrame((frame+1)%D.frames.length)},90);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);window.__OAKINK2_VIEWER_V2__={schema:"OakInk2HTMLViewerV2",setPreset,setFocus,setMode,setFrame,setOrbit,resetCamera,play,pause,paint,certificate:()=>window.__OAKINK2_BROWSER_CERTIFICATE__};paint();
</script></body></html>"""


def render_oakink2_html_viewer_v2(
    data: OakInk2HTMLViewerV2Data, destination: Path
) -> dict[str, Any]:
    """Write one offline V2 HTML whose only runtime input is embedded data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _HTML.replace("__DATA__", json.dumps(payload_for(data), separators=(",", ":"))),
        encoding="utf-8",
    )
    return {
        "schema_version": "OakInk2HTMLViewerV2",
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "self_contained": True,
        "cdn_required": False,
    }


def render_trusted_html_viewer(data: OakInk2HTMLViewerV2Data, destination: Path) -> dict[str, Any]:
    """Deprecated name retained for callers; production output is Viewer V2."""
    return render_oakink2_html_viewer_v2(data, destination)
