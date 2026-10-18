// PROTOTYPE, now the default entry (index.html loads it): a copy of main.js
// with a quality ladder, to answer "where does the frame time go, and what
// does each saving cost visually?" Run `npm run dev` (or `npm run proto:perf`).
// Switch tiers with ?tier=baseline|high|medium|low (the bottom bar does this;
// medium is the default); the toggles there cut one system at a time.
//
// Melted gold in an ice crown: a Position-Based Fluid (Three Blocks) held in a
// cavity inside a refractive ice crystal crown.
//
// The ice is the model itself (public/ice-crystal-crown.glb), rounded by Taubin
// smoothing as its sharpness drops. The gold's cavity is the model set back by
// the ice thickness, from its distance field (public/crystal-field.bin, baked
// by stream1018-tools/bake-sdf.mjs). The fluid starts filling the cavity in the
// crown's shape and melts down under gravity.
import * as THREE from 'three/webgpu';
import {
  Fn,
  vec2,
  vec4,
  uniform,
  mix,
  smoothstep,
  screenUV,
  screenCoordinate,
  interleavedGradientNoise,
  positionGeometry,
  pmremTexture,
  getViewPosition,
  viewZToPerspectiveDepth,
  cameraNear,
  cameraFar,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  pass,
  mrt,
  output,
  emissive,
  time,
  hash,
  uv,
  float,
  vec3,
  color,
  instanceIndex,
  instancedArray,
  varying,
  cameraViewMatrix,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { UltraHDRLoader } from 'three/addons/loaders/UltraHDRLoader.js';
import { MeshBVH } from 'three-mesh-bvh';
import GUI from 'lil-gui';
import Stats from 'stats-gl';
import { PBF } from 'three-blocks/pbf';
import { ComputeSDFGenerator, SDFVolumeConstraint } from 'three-blocks/sdf-raymarching';
import { WaterSurfaceRenderer } from 'three-blocks/water';
import { MeshTransmissionNodeMaterial } from 'three-blocks/transmission';
import { isosurface } from './marching.js';
import { weld, rounded } from './rounding.js';

const MODEL = `${import.meta.env.BASE_URL}ice-crystal-crown.glb`;
// what bake-sdf.mjs printed for crystal-field.bin:
//   node bake-sdf.mjs ice-crystal-crown.glb crystal-field.bin 200 --margin 0.03
const FIELD = {
  url: `${import.meta.env.BASE_URL}crystal-field.bin`,
  dims: [198, 186, 200],
  min: [-0.5406786456980657, -0.047216897551546455, -0.5461400612113403],
  voxelSize: 0.0054614006121134024,
  modelSize: 0.99951171875,
};
// The fluid's tuning (gravity, kernel, speeds) expects a scene tens of units
// across, as in the library's examples, so the simulation runs with the crown
// 40 units wide and the group scales it back down to 2 for display.
const SIM_SIZE = 40;
const SIM = SIM_SIZE / FIELD.modelSize; // field units -> simulation units
const DISPLAY = 2 / SIM_SIZE; // simulation units -> scene units

// --- PROTOTYPE: the quality ladder. Per the Three Blocks performance-ladder
// guidance, resolution, passes, samples, counts and DPR scale together.
//  dpr            renderer pixel ratio cap (transmission + gold are per-pixel)
//  antialias      MSAA on the main pass (the transmission snapshot resolves it)
//  particles      PBF count; spacing grows as it drops, so blobs get lumpier
//  solver         PBF density iterations per substep
//  maxSubsteps    cap on 1/120 s steps per frame; a slow frame slows the melt
//                 instead of queueing more work (the default 5 can spiral)
//  useDirection   PBF's smoothed-direction pass; nothing here reads it
//  samples        transmission taps per channel. The frost blur rotates its taps
//                 by a gradient noise that nothing averages over frames; at 1-2
//                 taps it shows as diagonal lines and a grid, at 6 it smooths out
//  prism, blur    chromatic aberration / anisotropic blur on the ice
//  surfaceScale   gold depth+normal targets vs the drawing buffer. Below 1 the
//                 composite upsamples them, which terraces the gold into flat
//                 blocks with stepped edges; the library's water example runs
//                 at 1 and measured it cleaner and faster (8.6 vs 16.6 ms)
//  depthSmoothing final range-limited depth smoothing in texels (default 2):
//                 rounds the plateaus the edge-preserving filter leaves
//  thicknessScale gold thickness target vs the drawing buffer
//  filter*        gold depth filter passes and width in texels
//  grid           how PBF rebuilds its neighbour grid every substep (see GRIDS)
//  stepRate       solver steps per simulated-and-scaled second; 60 halves the
//                 work at the same flow speed, with twice the motion per step
const TIERS = {
  baseline: { grid: 'bitonic sort (default)', stepRate: 120, dpr: 2, antialias: true, particles: 65536, solver: 3, maxSubsteps: 5, useDirection: true, samples: 3, prism: 0.04, blur: 0.05, surfaceScale: 0.5, depthSmoothing: 2, thicknessScale: 0.5, filterIterations: 2, filterRadius: 10 },
  high: { grid: 'atomic + workgroup', stepRate: 120, dpr: 1.5, antialias: true, particles: 65536, solver: 3, maxSubsteps: 2, useDirection: false, samples: 6, prism: 0.04, blur: 0.05, surfaceScale: 1, depthSmoothing: 4, thicknessScale: 0.5, filterIterations: 2, filterRadius: 10 },
  medium: { grid: 'atomic + workgroup', stepRate: 120, dpr: 1, antialias: true, particles: 32768, solver: 2, maxSubsteps: 2, useDirection: false, samples: 6, prism: 0.04, blur: 0.05, surfaceScale: 1, depthSmoothing: 4, thicknessScale: 0.35, filterIterations: 2, filterRadius: 10 },
  low: { grid: 'atomic + workgroup', stepRate: 120, dpr: 0.75, antialias: true, particles: 16384, solver: 2, maxSubsteps: 2, useDirection: false, samples: 1, prism: 0, blur: 0, surfaceScale: 1, depthSmoothing: 4, thicknessScale: 0.25, filterIterations: 1, filterRadius: 8 },
};
const tierName = new URLSearchParams(location.search).get('tier') ?? 'medium';
// #debug shows the panel, the timings, the tier bar and the leak count; without
// it the page is just the crown. GPU timestamps cost a little, so they are only
// asked for in debug, and switching in or out of #debug reloads to match.
const debug = location.hash === '#debug';
window.addEventListener('hashchange', () => location.reload());
const tier = TIERS[tierName] ?? TIERS.baseline;
// live isolation switches: each one removes a system so its cost shows up as the drop
const isolate = { sim: true, gold: true, transmission: true };
// particles outside the crown, for the readout (counted by countLeak, reset by rebuildFluid)
let leak = 'counting...';
// PBF's neighbour grid. By default it sorts the particles by cell with a
// bitonic sort: for 65k particles about 136 GPU passes, every substep. Radix
// sorts in a few passes; atomic skips sorting (count per cell, prefix sum,
// scatter), and only atomic allows the workgroup (shared-memory) neighbour lookup.
// Passed as the legacy spatialGridOptions: PBF's structured `neighbors` option
// switches the constructor to its structured path, which resets count, domain,
// kernel and time step to defaults when given alone (the gold then spills out).
const GRIDS = {
  'bitonic sort (default)': { buildAlgorithm: 'sort', sortAlgorithm: 'bitonic', lookupAlgorithm: 'global' },
  'radix sort': { buildAlgorithm: 'sort', sortAlgorithm: 'radix', lookupAlgorithm: 'global' },
  atomic: { buildAlgorithm: 'atomic', lookupAlgorithm: 'global' },
  'atomic + workgroup': { buildAlgorithm: 'atomic', lookupAlgorithm: 'workgroup' },
};
// the gold surface's knobs, live in the GUI; each change rebuilds the fluid
const goldSurfaceOptions = {
  surfaceScale: tier.surfaceScale,
  depthSmoothing: tier.depthSmoothing,
  filterIterations: tier.filterIterations,
  filterRadius: tier.filterRadius,
};
// PROTOTYPE: the cavity walls (SDFVolumeConstraint), live. Upside down the
// gold's weight lands on the thin spike tips and a few particles squeeze
// through: a stiffer wall pushes back harder, a bigger offset holds the gold
// further inside (in particle spacings).
const walls = { stiffness: 1e3, damping: 0.3, offset: 0.5 }; // tested: no leak upside down (was 5e3, 0.25)
// the simulation's knobs, live in the GUI; each change rebuilds the fluid
const sim = {
  grid: tier.grid,
  solver: tier.solver,
  stepRate: tier.stepRate,
  maxSubsteps: tier.maxSubsteps,
  useDirection: tier.useDirection,
};

// distances are in scene units (the crown is 2 across)
const settings = {
  // ice
  sharpness: 0.56,
  iceThickness: 0.012,
  frost: 0.22,
  ior: 1.31,
  iceTint: '#a9d4ff',
  clarity: 0.9,
  grain: 0.09,
  opacity: 0.01, // 0 the ice shows only what is behind it, 1 only its own colour
  iceColor: '#f2f8ff', // the ice's own colour: what milky ice shows, and a light tint on what it lets through
  prism: tier.prism,
  // gold
  fill: 0.35,
  particles: tier.particles,
  // thick and heavy: XSPH 0.9 (the library's gelLike preset is 0.25) makes
  // neighbours move together, and a slower clock makes it pour like syrup
  viscosity: 0.9,
  flowSpeed: 1,
  blobSize: 1.9, // wider splats read as rounder, fuller beads
  goldColor: '#ffd98a',
  goldRoughness: 0.18,
  heat: 0.12,
  heatColor: '#ff6a1a',
  remelt: () => rebuildFluid(),
};

const toField = (sceneDistance) => sceneDistance / (DISPLAY * SIM);

// --- renderer, scene, camera
const renderer = new THREE.WebGPURenderer({ antialias: tier.antialias, trackTimestamp: debug });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, tier.dpr));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
// the loading veil in index.html: its sigil fades away from the top down as
// the crown loads (--p, 0 to 1), and the veil lifts on the first frame
const veil = document.getElementById('veil');
const loaded = (progress) => veil?.style.setProperty('--p', progress.toFixed(3));
await renderer.init();
if (!renderer.backend.isWebGPUBackend) {
  veil?.remove();
  document.body.insertAdjacentHTML('beforeend', '<p class="note">This page needs WebGPU: the fluid runs in compute shaders.</p>');
  throw new Error('WebGPU is required');
}

const scene = new THREE.Scene();
// a cool lit backdrop: clear ice only shows what is behind it, and over black it reads as black glass
// the bottom colour runs into the top one across a band centred at `height`
// (0 the bottom of the screen, 1 the top) and `softness` wide
// a middle colour (at `middle` strength) swells in the centre of the band:
// between two far-apart hues it keeps the blend from going through mud
const backdropGradient = {
  bottom: uniform(new THREE.Color(0x101218)),
  top: uniform(new THREE.Color(0x6f7f96)),
  middle: uniform(new THREE.Color(0x0b0d14)),
  middleStrength: uniform(0),
  height: uniform(0.575),
  softness: uniform(0.85),
};
const GRADIENTS = {
  'cool steel': { top: '#6f7f96', bottom: '#101218', middle: '#0b0d14', middleStrength: 0, height: 0.575, softness: 0.85 },
  // icy blue above, embers below, a gold glow where they meet
  // near-black and almost flat: the crown carries all the colour
  charcoal: { top: '#0d0d0d', bottom: '#141414', middle: '#000000', middleStrength: 0.36, height: 0.5, softness: 0.99 },
  'ice & fire': { top: '#56b0f0', bottom: '#ff7847', middle: '#ffdc5c', middleStrength: 0.36, height: 0.5, softness: 0.99 },
};
// Drawn on a full-screen quad behind everything (first, without depth), so it
// stays put however the crown and camera turn, and the ice refracts it like
// any opaque thing behind it. Dithered: the scene renders to a half-float
// buffer and the output pass rounds it to 8 bits, which steps a slow dark
// gradient into bands. A one-step triangular noise, added in roughly display
// (gamma 2.2) space, breaks the steps into grain too fine to see.
const backdropMaterial = new THREE.MeshBasicNodeMaterial({ depthTest: false, depthWrite: false });
backdropMaterial.vertexNode = vec4(positionGeometry.xy, 0, 1);
backdropMaterial.colorNode = Fn(() => {
  const halfBand = backdropGradient.softness.mul(0.5).max(0.001);
  const t = smoothstep(backdropGradient.height.sub(halfBand), backdropGradient.height.add(halfBand), screenUV.y.oneMinus());
  const swell = t.mul(2).sub(1).abs().oneMinus().mul(backdropGradient.middleStrength); // 1 mid-band, 0 at the ends
  const linear = mix(mix(backdropGradient.bottom, backdropGradient.top, t), backdropGradient.middle, swell);
  const noise = interleavedGradientNoise(screenCoordinate.xy)
    .add(interleavedGradientNoise(screenCoordinate.xy.add(vec2(37, 17))))
    .sub(1); // triangular, -1..1
  return linear.pow(1 / 2.2).add(noise.div(255)).max(0).pow(2.2);
})();
const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), backdropMaterial);
backdrop.frustumCulled = false;
backdrop.renderOrder = -1;
scene.add(backdrop);
const pmrem = new THREE.PMREMGenerator(renderer);
const roomEnvironment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environment = roomEnvironment;
scene.environmentIntensity = 0.6;

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(0, 2.2, 5.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.8, 0);
// orbit and zoom only: the crown stays centred, between filling the view
// (the crown is 2 across; the view starts ~5.8 away) and a small object
controls.enablePan = false;
controls.minDistance = 3.2;
controls.maxDistance = 9;

const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(2, 4, 3);
scene.add(key);
const backLight = new THREE.PointLight(0xffd2a1, 12, 0, 2);
backLight.position.set(0, 1.6, -2.5);
scene.add(backLight);

// PROTOTYPE: no floor, so the crown can turn all the way round
// a dark stone disc to stand on, so the ice has something to refract
const floor = new THREE.Mesh(
  new THREE.CircleGeometry(6, 96),
  new THREE.MeshStandardNodeMaterial({ color: 0x23252c, roughness: 0.85 }),
);
floor.rotation.x = -Math.PI / 2;

// the crown: everything inside is in simulation units
const crown = new THREE.Group();
crown.scale.setScalar(DISPLAY);
scene.add(crown);


// --- the field
// streamed, so the veil can follow it: it is most of what the page downloads.
// Its decoded size is known from the grid (half floats), whatever the transfer encoding.
const field = await fetch(FIELD.url)
  .then(async (response) => {
    const bytes = new Uint8Array(FIELD.dims[0] * FIELD.dims[1] * FIELD.dims[2] * 2);
    const reader = response.body.getReader();
    for (let received = 0; ;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, received);
      received += value.length;
      loaded(0.75 * (received / bytes.length));
    }
    return bytes.buffer;
  })
  .then((buffer) => {
    const half = new Uint16Array(buffer);
    const data = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) data[i] = THREE.DataUtils.fromHalfFloat(half[i]);
    return { ...FIELD, data };
  });

// the model, welded once for rounding; its centre, in field units, puts the
// simulation round the origin
const model = await new GLTFLoader().loadAsync(MODEL);
loaded(0.8);
let modelMesh;
model.scene.traverse((o) => o.isMesh && (modelMesh = o));
model.scene.updateMatrixWorld(true);
const modelGeometry = modelMesh.geometry.clone().applyMatrix4(modelMesh.matrixWorld);
const welded = weld(modelGeometry);
modelGeometry.computeBoundingBox();
const centre = modelGeometry.boundingBox.getCenter(new THREE.Vector3());


function surface(iso) {
  const { positions, index } = isosurface(field, iso);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.translate(-centre.x, -centre.y, -centre.z);
  geometry.scale(SIM, SIM, SIM);
  // from the triangles, not the sampled gradient, which turns over in thin walls
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return geometry;
}

function volume(geometry) {
  const p = geometry.attributes.position.array;
  const idx = geometry.index.array;
  let v = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1])
      + p[a + 1] * (p[b + 2] * p[c] - p[b] * p[c + 2])
      + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return Math.abs(v / 6);
}

// --- the ice
const ice = new MeshTransmissionNodeMaterial({
  color: settings.iceColor,
  roughness: settings.frost,
  ior: settings.ior,
  thickness: 0.08, // optical thickness: about one crystal wall, in scene units
  chromaticAberration: settings.prism,
  anisotropicBlur: tier.blur,
  surfaceDistortion: settings.grain,
  distortionScale: 6,
  temporalDistortion: 0,
  attenuationColor: new THREE.Color(settings.iceTint),
  attenuationDistance: settings.clarity,
  backdropDistance: 0.6,
  samples: tier.samples,
});
// PROTOTYPE: the cheap stand-in for the ice when transmission is switched off
const plainIce = new THREE.MeshStandardNodeMaterial({ color: 0xcfe6ff, roughness: 0.1, metalness: 0, transparent: true, opacity: 0.35 });
const iceMesh = new THREE.Mesh(new THREE.BufferGeometry(), ice);
crown.add(iceMesh);

// sharpness 1 is the model as made, facets crisp; lower, its edges round off
// as if melting, and the shading creases soften with them
function rebuildIce() {
  const softness = 1 - settings.sharpness;
  const iterations = Math.round(softness * 40);
  const crease = THREE.MathUtils.degToRad(THREE.MathUtils.lerp(20, 180, softness));
  const geometry = rounded(welded, iterations, crease);
  geometry.translate(-centre.x, -centre.y, -centre.z);
  geometry.scale(SIM, SIM, SIM);
  geometry.computeBoundingBox();
  iceMesh.geometry.dispose();
  iceMesh.geometry = geometry;
  // stand the crown on the floor
  crown.position.y = -geometry.boundingBox.min.y * DISPLAY;
}

// --- the gold
// The particles are splatted and smoothed into one surface on screen
// (WaterSurfaceRenderer, fed plain positions), then drawn as an opaque
// full-screen layer that writes the surface's depth. Being opaque, it is part
// of what the ice's transmission refracts, like any other object behind it.
const goldUniforms = {
  color: uniform(new THREE.Color(settings.goldColor)),
  roughness: uniform(settings.goldRoughness),
  heat: uniform(settings.heat),
  heatColor: uniform(new THREE.Color(settings.heatColor)),
  bloomHeat: uniform(1), // how much heat colour the gold's motion sends to the bloom
  keyDirection: uniform(new THREE.Vector3()), // towards the key light, view space
  keyColor: uniform(new THREE.Color()),
  backDirection: uniform(new THREE.Vector3()), // towards the back light, view space
  backColor: uniform(new THREE.Color()),
  crownMatrix: uniform(new THREE.Matrix4()),
};

// PROTOTYPE: the gold's bloom follows its motion. Each particle glows by how
// fast it moves plus how hard it speeds up or slows down (its velocity's change
// since the last frame), smoothed over a few frames so a splash flares and
// fades rather than flickering. A compute pass keeps last frame's velocity and
// the smoothed glow per particle; a sprite per particle adds that glow to the
// bloom's `emissive` buffer only (its colour is zero, so without the bloom it
// shows nothing). The sprites skip the depth test, so the ice in front can't
// hide them, and drop themselves where they sit behind the gold's own
// reconstructed surface instead.
const motionGlow = {
  speedGain: uniform(1 / 6), // glow per simulation unit/s of speed
  accelGain: uniform(1 / 80), // glow per simulation unit/s² of acceleration
  smoothing: uniform(0.25), // share of the new value taken each frame
  size: uniform(1.4), // sprite size against the gold's splat diameter
  frameDelta: uniform(1 / 60),
};
let motionGlowParts = null;
function buildMotionGlow(count, splatRadius) {
  const velocities = fluid.buffers.velocities;
  const positions = fluid.buffers.positions;
  const previous = instancedArray(count, 'vec3');
  const glow = instancedArray(count, 'float');
  const compute = Fn(() => {
    const v = velocities.element(instanceIndex);
    const acceleration = v.sub(previous.element(instanceIndex)).length().div(motionGlow.frameDelta.max(1e-4));
    const target = v.length().mul(motionGlow.speedGain).add(acceleration.mul(motionGlow.accelGain)).min(3);
    glow.element(instanceIndex).assign(mix(glow.element(instanceIndex), target, motionGlow.smoothing));
    previous.element(instanceIndex).assign(v);
  })().compute(count);

  const material = new THREE.SpriteNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending });
  const world = goldUniforms.crownMatrix.mul(vec4(positions.element(instanceIndex), 1)).xyz;
  material.positionNode = world;
  material.scaleNode = motionGlow.size.mul(splatRadius * 2);
  const amount = varying(glow.element(instanceIndex));
  const distance = varying(cameraViewMatrix.mul(vec4(world, 1)).z.negate()); // the particle's centre, from the camera
  const front = goldSurface.surfaceDataNode;
  material.colorNode = Fn(() => {
    const surface = front.sample(screenUV).x; // the gold's front surface here, 0 where there is none
    surface.lessThanEqual(0).discard();
    distance.greaterThan(surface.add(splatRadius * 2)).discard();
    const spot = smoothstep(0, 0.5, uv().sub(0.5).length()).oneMinus();
    emissive.assign(goldUniforms.heatColor.mul(goldUniforms.bloomHeat).mul(amount).mul(spot));
    return vec3(0);
  })();
  const sprite = new THREE.Sprite(material);
  sprite.count = count;
  sprite.frustumCulled = false;
  scene.add(sprite);
  motionGlowParts = { compute, sprite, material };
}
function disposeMotionGlow() {
  if (!motionGlowParts) return;
  scene.remove(motionGlowParts.sprite);
  motionGlowParts.material.dispose();
  motionGlowParts.compute.dispose?.();
  motionGlowParts = null;
}
const gold = new THREE.MeshBasicNodeMaterial();
gold.vertexNode = vec4(positionGeometry.xy, 0, 1); // a full-screen quad
const goldLayer = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), gold);
goldLayer.frustumCulled = false;
goldLayer.visible = false;
scene.add(goldLayer);
let goldSurface = null;

const ggx = Fn(([n, v, l, rough]) => {
  const h = v.add(l).normalize();
  const a = rough.mul(rough).max(0.002);
  const a2 = a.mul(a);
  const nh = n.dot(h).max(0);
  const d = nh.mul(nh).mul(a2.sub(1)).add(1);
  return a2.div(d.mul(d).mul(Math.PI)).mul(n.dot(l).max(0)).mul(0.25);
});

function shadeGold(surfaceRenderer) {
  const data = surfaceRenderer.surfaceDataNode.sample(screenUV); // view depth, view normal
  const thickness = surfaceRenderer.thicknessNode.sample(screenUV).r;
  gold.colorNode = Fn(() => {
    data.x.lessThanEqual(0).discard();
    const viewZ = data.x.negate();
    const depth = viewZToPerspectiveDepth(viewZ, cameraNear, cameraFar);
    const p = getViewPosition(screenUV, depth, cameraProjectionMatrixInverse);
    const v = p.negate().normalize();
    const n = data.yzw.normalize().toVar();
    n.assign(n.dot(v).lessThan(0).select(n.negate(), n)); // face the camera
    const nv = n.dot(v).max(0);
    const f0 = goldUniforms.color;
    const fresnel = f0.add(f0.oneMinus().mul(nv.oneMinus().pow(5)));
    // reflections of the room, through the gold's Fresnel
    const r = cameraWorldMatrix.mul(vec4(v.negate().reflect(n), 0)).xyz;
    const env = pmremTexture(scene.environment, r, goldUniforms.roughness).mul(scene.environmentIntensity);
    const lit = env
      .add(goldUniforms.keyColor.mul(ggx(n, v, goldUniforms.keyDirection, goldUniforms.roughness)))
      .add(goldUniforms.backColor.mul(ggx(n, v, goldUniforms.backDirection, goldUniforms.roughness)));
    // molten gold glows from within where it runs deep
    const deep = smoothstep(0.02, 0.6, thickness);
    const glow = goldUniforms.heatColor.mul(goldUniforms.heat).mul(deep);
    return fresnel.mul(lit).add(glow);
  })();
  gold.depthNode = viewZToPerspectiveDepth(data.x.negate(), cameraNear, cameraFar);
  gold.needsUpdate = true;
}

let fluid = null;
let sdf = null;
let constraint = null;
// a rebuild waits for the frame in flight and holds the next ones off
let frame = Promise.resolve();
let rebuilding = Promise.resolve();
let busy = false;

async function buildFluid() {
  const cavityGeometry = surface(-toField(settings.iceThickness));
  cavityGeometry.boundsTree = new MeshBVH(cavityGeometry);
  const count = settings.particles;
  const spacing = Math.cbrt((volume(cavityGeometry) * settings.fill) / count);
  const particleRadius = spacing * 0.3;

  // the crystal walls are thin: the cavity needs several voxels across to hold the fluid
  sdf = new ComputeSDFGenerator({ resolution: 256, margin: 4 * spacing });
  await sdf.generate(cavityGeometry, cavityGeometry.boundsTree, renderer);
  const response = { stiffness: walls.stiffness, damping: walls.damping, threshold: -walls.offset * spacing };
  constraint = new SDFVolumeConstraint(sdf, response);
  // seeded here rather than by ComputeBVHSampler, whose 64 tries per particle
  // mostly miss a cavity this thin and leave the particle outside it
  const seeds = seedCavity(count, -toField(settings.iceThickness) - (particleRadius + 0.25 * spacing) / SIM);
  if (location.search.includes('debug')) await probeSeeds(seeds, count, spacing);

  const size = cavityGeometry.boundingBox.getSize(new THREE.Vector3()).addScalar(8 * spacing);
  fluid = new PBF({
    count,
    initialPositions: seeds,
    is3D: true,
    domainDimensions: size,
    mass: 1,
    useDirection: sim.useDirection,
    useMatrices: false,
    useSpatialGrid: true,
    spatialGridOptions: GRIDS[sim.grid],
    scaleKernelWithDomain: false,
    fixedTimeStep: 1 / sim.stepRate,
    sdfVolumeConstraint: constraint,
    h: spacing * 2,
    spacing,
    particleRadius,
    restDensity: null,
    solverIterations: sim.solver,
    maxSubsteps: sim.maxSubsteps,
    lambdaEpsilon: 5,
    xsphViscosity: settings.viscosity,
    corrK: 0,
    corrN: 4,
  });
  fluid.ubos.timeScale.value = settings.flowSpeed;
  await fluid.step(renderer, 1 / sim.stepRate);
  cavityGeometry.dispose();

  // the surface renderer reads world positions: the particles live in the crown's frame
  const positions = fluid.buffers.positions;
  const worldPositions = {
    element: (i) => goldUniforms.crownMatrix.mul(vec4(positions.element(i), 1)),
  };
  goldSurface = new WaterSurfaceRenderer(renderer, { positions: worldPositions, count }, {
    worldRadius: spacing * settings.blobSize * DISPLAY,
    temporalBlend: 0, // the crown turns, and history would smear it
    filterIterations: goldSurfaceOptions.filterIterations,
    filterRadius: goldSurfaceOptions.filterRadius,
    resolutionScale: goldSurfaceOptions.surfaceScale,
    depthSmoothingRadius: goldSurfaceOptions.depthSmoothing,
    thicknessScale: tier.thicknessScale,
  });
  goldLayer.userData.spacing = spacing;
  shadeGold(goldSurface);
  buildMotionGlow(count, spacing * settings.blobSize * DISPLAY);
}

// the field, trilinear, at a point in field units
function sampleField(x, y, z) {
  const [nx, ny, nz] = field.dims;
  const gx = Math.min(Math.max((x - field.min[0]) / field.voxelSize - 0.5, 0), nx - 1.001);
  const gy = Math.min(Math.max((y - field.min[1]) / field.voxelSize - 0.5, 0), ny - 1.001);
  const gz = Math.min(Math.max((z - field.min[2]) / field.voxelSize - 0.5, 0), nz - 1.001);
  const i = Math.floor(gx), j = Math.floor(gy), k = Math.floor(gz);
  const fx = gx - i, fy = gy - j, fz = gz - k;
  const d = field.data;
  const at = (a, b, c) => d[i + a + nx * (j + b + ny * (k + c))];
  const lerp = (a, b, t) => a + (b - a) * t;
  return lerp(
    lerp(lerp(at(0, 0, 0), at(1, 0, 0), fx), lerp(at(0, 1, 0), at(1, 1, 0), fx), fy),
    lerp(lerp(at(0, 0, 1), at(1, 0, 1), fx), lerp(at(0, 1, 1), at(1, 1, 1), fx), fy),
    fz,
  );
}

// `count` points spread evenly through the part of the crown deeper than
// `depth` (field units, negative inside), in simulation units
function seedCavity(count, depth) {
  const box = modelGeometry.boundingBox;
  const lo = box.min, size = box.getSize(new THREE.Vector3());
  const out = new Float32Array(count * 4);
  let n = 0;
  for (let tries = 0; n < count && tries < count * 2000; tries++) {
    const x = lo.x + Math.random() * size.x;
    const y = lo.y + Math.random() * size.y;
    const z = lo.z + Math.random() * size.z;
    if (sampleField(x, y, z) >= depth) continue;
    out[n * 4] = (x - centre.x) * SIM;
    out[n * 4 + 1] = (y - centre.y) * SIM;
    out[n * 4 + 2] = (z - centre.z) * SIM;
    out[n * 4 + 3] = 1;
    n++;
  }
  if (n < count) console.warn(`seeded ${n} of ${count}: the cavity is too thin at this ice thickness`);
  return new THREE.StorageInstancedBufferAttribute(out, 4);
}

// ?debug: where the seeded particles sit in the crown's own field
async function probeSeeds(buffer, count, spacing) {
  const data = buffer.isBufferAttribute ? buffer.array : new Float32Array(await renderer.getArrayBufferAsync(buffer));
  const stride = data.length / count;
  const [nx, ny] = field.dims;
  const inset = -toField(settings.iceThickness);
  let outside = 0, outOfGrid = 0;
  const yHist = new Array(10).fill(0);
  let minY = Infinity, maxY = -Infinity;
  for (let n = 0; n < count; n++) {
    const p = [0, 1, 2].map((a) => data[n * stride + a] / SIM + centre.getComponent(a));
    const g = p.map((c, a) => Math.round((c - field.min[a]) / field.voxelSize - 0.5));
    if (g.some((c, a) => c < 0 || c >= field.dims[a])) { outOfGrid++; continue; }
    const d = field.data[g[0] + nx * (g[1] + ny * g[2])];
    minY = Math.min(minY, p[1]);
    maxY = Math.max(maxY, p[1]);
    if (d > inset + field.voxelSize) {
      outside++;
      yHist[Math.min(9, Math.max(0, Math.floor(p[1] / 0.1)))]++;
    }
  }
  console.log(`[probe] seeds ${count}, stride ${stride}, spacing ${spacing.toFixed(3)}, out of grid ${outOfGrid}, outside the cavity ${outside}, y ${minY.toFixed(3)}..${maxY.toFixed(3)}, outside by y/0.1: ${yHist.join(' ')}`);
}

function disposeFluid() {
  disposeMotionGlow();
  goldSurface?.dispose();
  fluid?.dispose();
  constraint?.dispose();
  sdf?.dispose();
  fluid = constraint = sdf = goldSurface = null;
}

function rebuildFluid() {
  rebuilding = rebuilding.then(async () => {
    busy = true;
    leak = 'counting...';
    await frame;
    disposeFluid();
    try {
      await buildFluid();
    } finally {
      busy = false;
    }
  });
  return rebuilding;
}

rebuildIce();
await rebuildFluid();
loaded(0.95);

// --- controls
const gui = new GUI({ title: 'Melted gold in ice' });
if (!debug) gui.hide();
const iceFolder = gui.addFolder('Ice');
iceFolder.add(settings, 'sharpness', 0, 1, 0.01).onFinishChange(rebuildIce);
// the ice between the crown's surface and the gold
iceFolder.add(settings, 'iceThickness', 0.004, 0.03, 0.001).name('thickness').onFinishChange(rebuildFluid);
iceFolder.add(settings, 'frost', 0, 1, 0.01).onChange(() => applyIce());
iceFolder.add(settings, 'opacity', 0, 1, 0.01).name('opacity (milky)').onChange(() => applyIce());
iceFolder.addColor(settings, 'iceColor').name('ice color').onChange((v) => ice.color.set(v));
iceFolder.add(settings, 'ior', 1, 1.6, 0.01).name('IOR').onChange(() => applyIce());
iceFolder.addColor(settings, 'iceTint').name('tint').onChange((v) => ice.attenuationColor.set(v));
iceFolder.add(settings, 'clarity', 0.05, 5, 0.05).onChange(() => applyIce());
iceFolder.add(settings, 'grain', 0, 0.1, 0.001).onChange(() => applyIce());
iceFolder.add(settings, 'prism', 0, 0.5, 0.01).onChange(() => applyIce());

// PROTOTYPE: the ice's pixel noise. The transmission blurs by roughness with a
// world-anchored dither pattern (scale 128 by default): each pixel picks one of
// two blur mips and rotates its few taps by it, whatever ditherStrength says,
// and nothing averages it over frames. At roughness 0 there is no blur, so no
// noise. `on` off zeroes frost, dither and grain, and keeps their values for when it is back on
const noise = { on: true, dither: 0.52 };
// Every ice control lands here. `look.strength` scales the whole ice look at
// once: 0 is a thin, clear, untinted shell, 1 the tuned look, 2 twice as heavy.
// It scales glass depth, frost, grain, dither, prism and how far the IOR bends
// from 1, and tightens the tint distance as it grows.
function applyIce() {
  const k = look.strength;
  const noisy = noise.on ? k : 0;
  ice.thickness = (look.thickness / DISPLAY) * k;
  ice.ior = 1 + (settings.ior - 1) * k;
  ice.attenuationDistance = settings.clarity / Math.max(k, 0.05);
  ice.chromaticAberration = settings.prism * k;
  ice.transmission = 1 - settings.opacity;
  ice.roughness = settings.frost * noisy;
  ice.ditherStrength = noise.dither * noisy;
  ice.surfaceDistortion = settings.grain * noisy;
}
const applyNoise = () => applyIce();
const noiseFolder = gui.addFolder('Ice noise (prototype)');
noiseFolder.add(noise, 'on').name('ice noise').onChange(applyNoise);
noiseFolder.add(noise, 'dither', 0, 1, 0.01).name('dither strength').onChange(applyNoise);
noiseFolder.add(ice, 'ditherScale', 16, 2048, 1).name('dither scale').listen();
noiseFolder.add(ice, 'ditherAnchor', ['world', 'screen']).name('dither anchor (recompiles)');
noiseFolder.add(ice, 'samples', 1, 8, 1).name('taps (recompiles)');

// PROTOTYPE: the ice's look, scale-corrected. The crown group is scaled by
// DISPLAY (0.05), and the transmission measures in different spaces:
//  thickness        local units x the model's world scale, so 0.08 was 0.004
//                   scene units of glass: next to no refraction or absorption
//  distortionScale  local units (the crown is 40 across): 6 put ~240 noise
//                   cells across it, under a pixel each, so they alias and the
//                   material half-fades them
//  ditherScale      world units (the crown is 2 across): 128 made ~256 dither
//                   cells across it, 2-4 pixels each, the blocky grain
// 'scale-corrected' carries the library example's proportions (a 10-unit cube
// at distortionScale 8, ditherScale 128, roughness 0.16) over to the crown.
const LOOKS = {
  'main.js': { thickness: 0.08, distortionScale: 6, ditherScale: 128, frost: 0.22 },
  'scale-corrected': { thickness: 0.08 / DISPLAY, distortionScale: 1.2, ditherScale: 640, frost: 0.16 },
  // tuned by eye on the scale-corrected look (2026-10-18)
  frosted: { thickness: 0.09 / DISPLAY, distortionScale: 0.7, ditherScale: 1359, frost: 0.25 },
};
const look = { preset: 'frosted', strength: 0.65, thickness: 0.09, environment: 'royal_esplanade_2k.hdr.jpg', backdrop: 'gradient', cssColor: '#15161a', gradient: 'charcoal', gradientBottom: '#101218', gradientTop: '#6f7f96', gradientMiddle: '#0b0d14', backdropBrightness: 0.35, backdropBlur: 0.3 };
function applyLook() {
  const preset = LOOKS[look.preset];
  look.thickness = preset.thickness * DISPLAY;
  ice.distortionScale = preset.distortionScale;
  ice.ditherScale = preset.ditherScale;
  settings.frost = preset.frost;
  applyNoise();
  gui.controllersRecursive().forEach((c) => c.updateDisplay());
}
const lookFolder = gui.addFolder('Ice look (prototype)');
lookFolder.add(look, 'preset', Object.keys(LOOKS)).name('look').onChange(applyLook);
lookFolder.add(look, 'strength', 0, 2, 0.01).name('ice strength').onChange(() => applyIce());
lookFolder.add(look, 'thickness', 0, 0.5, 0.005).name('glass depth (scene)').listen().onChange(() => applyIce());
lookFolder.add(ice, 'distortionScale', 0.1, 8, 0.05).name('grain size (smaller = bigger)').listen();
// the docs: refraction over a flat backdrop can make a correct material look broken
// The environment lights the ice and gold and fills their reflections: the
// procedural room (what the scene always had) or one of three.js's example
// HDRs, copied from three.js/examples/textures into public/hdr. Each loads on
// first pick and is kept.
const ENVIRONMENTS = {
  'room (procedural)': null,
  ...Object.fromEntries([
    'lobe.hdr',
    'pedestrian_overpass_1k.hdr',
    'royal_esplanade_2k.hdr.jpg',
    'san_giuseppe_bridge_2k.hdr',
  ].map((file) => [file, `${import.meta.env.BASE_URL}hdr/${file}`])),
};
const loadedEnvironments = new Map([['room (procedural)', { background: roomEnvironment, environment: roomEnvironment }]]);
async function loadEnvironment(name) {
  if (loadedEnvironments.has(name)) return loadedEnvironments.get(name);
  const source = ENVIRONMENTS[name];
  let background, environment;
  const loader = source.endsWith('.jpg') ? new UltraHDRLoader() : new HDRLoader();
  background = await loader.loadAsync(source);
  background.mapping = THREE.EquirectangularReflectionMapping;
  environment = pmrem.fromEquirectangular(background).texture;
  const loaded = { background, environment };
  loadedEnvironments.set(name, loaded);
  return loaded;
}
async function applyEnvironment(name) {
  const loaded = await loadEnvironment(name);
  if (look.environment !== name) return; // another pick landed first
  scene.environment = loaded.environment;
  if (goldSurface) shadeGold(goldSurface); // its reflections sample the environment it was built with
  applyBackdrop(look.backdrop);
}

// What is behind the crown. 'CSS color' clears the canvas to transparent so
// the page's background (the colour below) shows through, but the ice refracts
// only what the renderer drew: through it the CSS is invisible, and the ice
// shows the gold and its own back faces over black. 'environment' shows the
// picked environment, which the ice does refract.
function applyBackdrop(kind) {
  const loaded = loadedEnvironments.get(look.environment);
  backdrop.visible = kind === 'gradient';
  scene.background = kind === 'environment' && loaded ? loaded.background : null;
  scene.backgroundBlurriness = look.backdropBlur;
  scene.backgroundIntensity = look.backdropBrightness;
  renderer.setClearColor(0x000000, kind === 'CSS color' ? 0 : 1);
}
const applyCssColor = (v) => (document.body.style.background = v);
lookFolder.add(look, 'environment', Object.keys(ENVIRONMENTS)).name('environment').onChange(applyEnvironment);
lookFolder.add(look, 'backdrop', ['CSS color', 'environment', 'gradient']).name('behind').onChange(applyBackdrop);
function applyGradient(name) {
  const g = GRADIENTS[name];
  look.gradientTop = g.top;
  look.gradientBottom = g.bottom;
  look.gradientMiddle = g.middle;
  backdropGradient.top.value.set(g.top);
  backdropGradient.bottom.value.set(g.bottom);
  backdropGradient.middle.value.set(g.middle);
  backdropGradient.middleStrength.value = g.middleStrength;
  backdropGradient.height.value = g.height;
  backdropGradient.softness.value = g.softness;
  gui.controllersRecursive().forEach((c) => c.updateDisplay());
}
lookFolder.add(look, 'gradient', Object.keys(GRADIENTS)).name('gradient preset').onChange(applyGradient);
lookFolder.addColor(look, 'gradientTop').name('gradient top').onChange((v) => backdropGradient.top.value.set(v));
lookFolder.addColor(look, 'gradientMiddle').name('gradient middle').onChange((v) => backdropGradient.middle.value.set(v));
lookFolder.add(backdropGradient.middleStrength, 'value', 0, 1, 0.01).name('gradient middle strength');
lookFolder.addColor(look, 'gradientBottom').name('gradient bottom').onChange((v) => backdropGradient.bottom.value.set(v));
lookFolder.add(backdropGradient.height, 'value', 0, 1, 0.01).name('gradient mix height');
lookFolder.add(backdropGradient.softness, 'value', 0, 2, 0.01).name('gradient softness');
applyGradient(look.gradient);
lookFolder.addColor(look, 'cssColor').name('CSS color').onChange(applyCssColor);
lookFolder.add(look, 'backdropBrightness', 0, 1.5, 0.01).name('environment brightness').onChange((v) => (scene.backgroundIntensity = v));
lookFolder.add(look, 'backdropBlur', 0, 1, 0.01).name('environment blur').onChange((v) => (scene.backgroundBlurriness = v));
// the gold's reflections bake the intensity in, so a change re-shades it
lookFolder.add(scene, 'environmentIntensity', 0, 2, 0.01).name('lighting').onFinishChange(() => goldSurface && shadeGold(goldSurface));
applyLook();
applyCssColor(look.cssColor);
await applyEnvironment(look.environment);
loaded(1);

// PROTOTYPE: resolution, live. MSAA can only change on reload (the tier sets it).
const view = { pixelRatio: renderer.getPixelRatio() };
const renderFolder = gui.addFolder('Render (prototype)');
renderFolder.add(view, 'pixelRatio', 0.5, Math.max(2, window.devicePixelRatio), 0.05).name(`pixel ratio (screen ${window.devicePixelRatio})`).onFinishChange((v) => {
  renderer.setPixelRatio(v);
  renderer.setSize(window.innerWidth, window.innerHeight);
  showState();
});

const surfaceFolder = gui.addFolder('Gold surface (prototype)');
surfaceFolder.add(goldSurfaceOptions, 'surfaceScale', [0.5, 0.75, 1]).name('resolution').onFinishChange(rebuildFluid);
surfaceFolder.add(goldSurfaceOptions, 'depthSmoothing', 0, 8, 1).name('depth smoothing').onFinishChange(rebuildFluid);
surfaceFolder.add(goldSurfaceOptions, 'filterIterations', 0, 4, 1).name('filter passes').onFinishChange(rebuildFluid);
surfaceFolder.add(goldSurfaceOptions, 'filterRadius', 2, 20, 1).name('filter width').onFinishChange(rebuildFluid);

const wallsFolder = gui.addFolder('Walls (prototype)');
wallsFolder.add(walls, 'stiffness', 1e3, 1e5, 1e3).onChange((v) => constraint && (constraint.stiffness = v));
wallsFolder.add(walls, 'damping', 0, 1, 0.01).onChange((v) => constraint && (constraint.damping = v));
wallsFolder.add(walls, 'offset', 0, 1, 0.01).name('offset (spacings)').onChange((v) => {
  if (constraint) constraint.threshold = -v * goldLayer.userData.spacing;
});

const simFolder = gui.addFolder('Simulation (prototype)');
simFolder.add(sim, 'grid', Object.keys(GRIDS)).name('neighbour grid').onFinishChange(rebuildFluid);
simFolder.add(sim, 'solver', 1, 4, 1).name('solver iterations').onFinishChange(rebuildFluid);
simFolder.add(sim, 'stepRate', [60, 90, 120]).name('steps / s').onFinishChange(rebuildFluid);
simFolder.add(sim, 'maxSubsteps', 1, 5, 1).name('max substeps').onFinishChange(rebuildFluid);
simFolder.add(sim, 'useDirection').name('direction pass').onFinishChange(rebuildFluid);

const goldFolder = gui.addFolder('Liquid gold');
goldFolder.add(settings, 'fill', 0.1, 1, 0.01).name('amount').onFinishChange(rebuildFluid);
goldFolder.add(settings, 'particles', [16384, 32768, 65536, 131072]).onFinishChange(rebuildFluid);
goldFolder.add(settings, 'viscosity', 0, 1, 0.01).onChange((v) => fluid && (fluid.ubos.xsphViscosity.value = v));
goldFolder.add(settings, 'flowSpeed', 0, 5, 0.05).name('flow speed').onChange((v) => fluid && (fluid.ubos.timeScale.value = v));
goldFolder.add(settings, 'blobSize', 0.5, 3, 0.01).name('smoothness').onChange((v) => {
  goldSurface?.setWorldRadius((goldLayer.userData.spacing ?? 1) * v * DISPLAY);
});
goldFolder.addColor(settings, 'goldColor').name('color').onChange((v) => goldUniforms.color.value.set(v));
goldFolder.add(settings, 'goldRoughness', 0.02, 1, 0.01).name('roughness').onChange((v) => (goldUniforms.roughness.value = v));
goldFolder.add(settings, 'heat', 0, 5, 0.05).name('heat glow').onChange((v) => (goldUniforms.heat.value = v));
goldFolder.addColor(settings, 'heatColor').name('glow color').onChange((v) => goldUniforms.heatColor.value.set(v));
goldFolder.add(settings, 'remelt').name('melt again');


// --- loop
const timer = new THREE.Timer();
const gravity = new THREE.Vector3();
const inverse = new THREE.Quaternion();

// PROTOTYPE: the crown's motion, as a spin it always carries. On its own the
// spin turns it round the vertical (clockwise seen from above) while a tumble
// whose axis runs a 1:2 sine figure (sin u, sin 2u = 2 sin u cos u) in the
// horizontal plane rolls it over, so it goes upside down and its top crosses
// back over its path. Grab it (a drag that starts on the crown) and it follows
// the mouse about the camera's axes; let go and the throw's spin carries on
// on top of its own spin and dies away (inertia), while its own spin lerps
// back in from rest (resume), so it carries on from wherever it is without
// snapping back. Its own clock pauses while held. A drag anywhere else
// orbits the camera. The capture listener runs before OrbitControls' own
// pointerdown, so it can switch the controls off.
const motion = {
  speed: 1, // multiplies its own spin and tumble
  spin: 0.3, // rad/s round the vertical
  tumble: 0.35, // rad/s rolling it over
  pace: 0.25, // rad/s, how fast the tumble's axis runs the figure
  inertia: 0.25, // s, how long a throw takes to die to a third
  paused: false, // stops its own motion; a drag and a throw still work
  resume: 1.5, // s, how long its own spin takes to lerp back in (to two thirds)
};
const motionFolder = gui.addFolder('Crown motion (prototype)');
motionFolder.add(motion, 'paused').name('pause motion');
motionFolder.add(motion, 'speed', 0, 3, 0.01).name('rotation speed');
motionFolder.add(motion, 'spin', 0, 1, 0.01);
motionFolder.add(motion, 'tumble', 0, 1.5, 0.01);
motionFolder.add(motion, 'pace', 0, 1, 0.01).name('tumble pace');
motionFolder.add(motion, 'inertia', 0.1, 6, 0.05).name('inertia (s)');
motionFolder.add(motion, 'resume', 0.1, 6, 0.05).name('resume own spin (s)');

const orientation = new THREE.Quaternion();
const throwVelocity = new THREE.Vector3(); // the throw's angular velocity, rad/s about world axes
const own = new THREE.Vector3();
let ownWeight = 0; // how much of its own spin is back, 0 just after a grab, 1 in full
const step = new THREE.Vector3();
let motionTime = 0; // the own spin's clock, paused while held
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const turn = new THREE.Quaternion();
const axis = new THREE.Vector3();
let grab = null;

// its own spin at time t: clockwise round the vertical, plus the tumble
function ownSpin(t, out) {
  const u = t * motion.pace;
  return out.set(Math.sin(u) * motion.tumble, -motion.spin, Math.sin(2 * u) * motion.tumble).multiplyScalar(motion.speed);
}

// turn the crown by a small rotation vector (axis x angle, world axes)
function turnBy(rotation) {
  const angle = rotation.length();
  if (angle < 1e-6) return;
  orientation.premultiply(turn.setFromAxisAngle(axis.copy(rotation).divideScalar(angle), angle)).normalize();
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  pointer.set((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  if (raycaster.intersectObject(iceMesh, false).length === 0) return;
  controls.enabled = false;
  throwVelocity.set(0, 0, 0);
  ownWeight = 0;
  grab = { x: event.clientX, y: event.clientY, time: performance.now() };
  renderer.domElement.setPointerCapture(event.pointerId);
}, { capture: true });
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!grab) return;
  const now = performance.now();
  const dx = event.clientX - grab.x, dy = event.clientY - grab.y;
  const dt = Math.max((now - grab.time) / 1000, 1 / 240);
  grab.x = event.clientX;
  grab.y = event.clientY;
  grab.time = now;
  // left/right round the screen's vertical, up/down round its horizontal
  step.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(dx * 0.01);
  step.add(axis.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(dy * 0.01));
  turnBy(step);
  // the throw: a smoothed velocity from the last few moves
  throwVelocity.lerp(step.divideScalar(dt), 0.5);
});
const release = () => {
  if (!grab) return;
  // held still before letting go: no throw
  if (performance.now() - grab.time > 80) throwVelocity.set(0, 0, 0);
  grab = null;
  controls.enabled = true;
};
renderer.domElement.addEventListener('pointerup', release);
renderer.domElement.addEventListener('pointercancel', release);

function updateCrownMotion(dt) {
  if (!grab) {
    throwVelocity.multiplyScalar(Math.exp(-dt / motion.inertia));
    if (motion.paused) {
      ownWeight = 0; // so unpausing lerps its own motion back in
    } else {
      motionTime += dt;
      ownWeight += (1 - ownWeight) * (1 - Math.exp(-dt / motion.resume));
    }
    ownSpin(motionTime, own).multiplyScalar(ownWeight);
    turnBy(step.copy(throwVelocity).add(own).multiplyScalar(dt));
  }
  crown.quaternion.copy(orientation);
}

// the direction from the crown towards a light, in view space
const towards = (light, out) =>
  out.copy(light.position).sub(crown.position).normalize().transformDirection(camera.matrixWorldInverse);

// --- PROTOTYPE: timing readout and the bottom bar
// stats-gl: FPS, CPU ms, GPU ms (render passes) and CPT ms (compute passes)
const stats = debug ? new Stats({ trackGPU: true, trackCPT: true, horizontal: true }) : null;
if (stats) {
  document.body.appendChild(stats.dom);
  await stats.init(renderer);
}

const bar = document.createElement('div');
bar.style.cssText = 'position:fixed;left:50%;bottom:12px;transform:translateX(-50%);display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:center;max-width:96vw;padding:8px 10px;background:#000c;color:#ddd;font:12px system-ui,sans-serif;border-radius:8px;z-index:10';
if (debug) document.body.appendChild(bar);
const state = document.createElement('div');
state.style.cssText = 'position:fixed;right:12px;bottom:70px;padding:8px 10px;background:#000c;color:#bbb;font:11px ui-monospace,monospace;white-space:pre;border-radius:8px;z-index:10';
if (debug) document.body.appendChild(state);

for (const name of Object.keys(TIERS)) {
  const a = document.createElement('a');
  a.textContent = name;
  a.href = `?tier=${name}#debug`; // the bar only shows in debug: stay there
  a.style.cssText = `padding:4px 10px;border-radius:5px;text-decoration:none;color:#fff;background:${name === tierName ? '#c8901e' : '#333'}`;
  bar.appendChild(a);
}
bar.appendChild(Object.assign(document.createElement('span'), { textContent: '|', style: 'opacity:.4' }));
const toggles = {
  sim: 'fluid sim',
  gold: 'gold surface',
  transmission: 'ice transmission',
};
for (const [key, label] of Object.entries(toggles)) {
  const l = document.createElement('label');
  l.style.cssText = 'display:flex;gap:4px;align-items:center;cursor:pointer';
  const box = Object.assign(document.createElement('input'), { type: 'checkbox', checked: isolate[key] });
  box.onchange = () => {
    isolate[key] = box.checked;
    if (key === 'transmission') iceMesh.material = box.checked ? ice : plainIce;
    showState();
  };
  l.append(box, label);
  bar.appendChild(l);
}

// PROTOTYPE: how many particles have got out through the ice, read back every
// 2 s (the readback costs a little, so timings wobble when it lands)
let leakBusy = false;
async function countLeak() {
  if (leakBusy || busy || !fluid) return;
  leakBusy = true;
  try {
    const current = fluid;
    const count = current.particleCount;
    const data = new Float32Array(await renderer.getArrayBufferAsync(current.buffers.positions.value));
    if (current !== fluid) return;
    const stride = data.length / count;
    let out = 0;
    for (let n = 0; n < count; n++) {
      const x = data[n * stride] / SIM + centre.x;
      const y = data[n * stride + 1] / SIM + centre.y;
      const z = data[n * stride + 2] / SIM + centre.z;
      const inGrid = [x, y, z].every((c, a) => c > field.min[a] && c < field.min[a] + field.dims[a] * field.voxelSize);
      if (!inGrid || sampleField(x, y, z) > 0) out++;
    }
    leak = `${out} of ${count} (${((100 * out) / count).toFixed(2)}%) outside the crown`;
  } catch {
    // the fluid was rebuilt under the readback; the next count catches up
  } finally {
    leakBusy = false;
  }
}
if (debug) setInterval(countLeak, 2000);

function showState() {
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const lines = [
    `tier         ${tierName}`,
    `buffer       ${size.x}x${size.y} @ dpr ${renderer.getPixelRatio().toFixed(2)}${tier.antialias ? ' + MSAA' : ''}`,
    `particles    ${fluid?.particleCount ?? settings.particles}`,
    `grid         ${sim.grid}`,
    `solver       ${sim.solver} it x <=${sim.maxSubsteps} substeps of 1/${sim.stepRate} s`,
    `useDirection ${sim.useDirection}`,
    `ice          ${isolate.transmission ? `transmission, ${tier.samples} taps, prism ${tier.prism}` : 'plain (no transmission)'}`,
    `gold         ${isolate.gold ? `surface ${goldSurface?.width}x${goldSurface?.height} (x${goldSurfaceOptions.surfaceScale}, smooth ${goldSurfaceOptions.depthSmoothing}), thickness x${tier.thicknessScale}, filter ${goldSurfaceOptions.filterIterations}x${goldSurfaceOptions.filterRadius}` : 'off'}`,
    `sim          ${isolate.sim ? 'running' : 'frozen'}`,
    `leaked       ${leak}`,
  ];
  state.textContent = lines.join('\n');
}

// --- PROTOTYPE: post-processing: bloom on what glows. The scene renders
// through a pass with a second output, `emissive`, which only the gold's motion
// glow and the embers write; the bloom spreads that buffer at quarter
// resolution and adds it back before the tone mapping. The emissive output
// blends additively, so the ice drawn over the gold adds its zero instead of
// erasing the glow underneath it. With post-processing off, the scene renders
// straight to the screen as before.
const fx = { post: true, bloom: true, strength: 0.1, radius: 0 };
const scenePass = pass(scene, camera);
const sceneOutputs = mrt({ output, emissive });
sceneOutputs.setBlendMode('emissive', new THREE.BlendMode(THREE.AdditiveBlending));
scenePass.setMRT(sceneOutputs);
const goldBloom = bloom(scenePass.getTextureNode('emissive'), fx.strength, fx.radius, 0);
goldBloom.setResolutionScale(0.25);
const pipeline = new THREE.RenderPipeline(renderer, scenePass.getTextureNode('output').add(goldBloom));
const bloomFolder = gui.addFolder('Bloom (prototype)');
bloomFolder.add(fx, 'post').name('post-processing');
bloomFolder.add(fx, 'bloom').name('bloom').onChange((on) => (goldBloom.strength.value = on ? fx.strength : 0));
bloomFolder.add(fx, 'strength', 0, 3, 0.01).name('bloom strength').onChange((v) => fx.bloom && (goldBloom.strength.value = v));
bloomFolder.add(fx, 'radius', 0, 1, 0.01).name('bloom radius').onChange((v) => (goldBloom.radius.value = v));
bloomFolder.add(goldUniforms.bloomHeat, 'value', 0, 8, 0.01).name('gold glow');
bloomFolder.add(motionGlow.speedGain, 'value', 0, 1, 0.005).name('glow from speed');
bloomFolder.add(motionGlow.accelGain, 'value', 0, 0.1, 0.0005).name('glow from acceleration');
bloomFolder.add(motionGlow.smoothing, 'value', 0.02, 1, 0.01).name('glow response');
bloomFolder.add(motionGlow.size, 'value', 0.3, 4, 0.05).name('glow spot size');

// --- PROTOTYPE: embers. Instanced sprites whose whole life is worked out in
// the vertex stage from their index and the clock, so they cost no simulation.
// Each rises from somewhere round the crown, drifts, flickers and fades, then
// starts again from a new spot (its seeds change every cycle). They write their
// glow to `emissive` too, so the bloom catches them.
const atmosphere = { embers: true, emberCount: 50 };
const crownCentre = uniform(new THREE.Vector3()); // the crown group's origin is its centre
const embersUniforms = {
  rise: uniform(1.5), // how far an ember climbs over its life, scene units
  spread: uniform(0.85), // radius round the crown they start within
  size: uniform(0.018),
  brightness: uniform(2.5),
  speed: uniform(1), // multiplies their rate of rise and their drift
};
const MAX_EMBERS = 1000;
const emberMaterial = new THREE.SpriteNodeMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
const emberRate = mix(0.12, 0.28, hash(instanceIndex.add(7919))).mul(embersUniforms.speed); // cycles a second
const emberClock = time.mul(emberRate).add(hash(instanceIndex));
const emberLife = emberClock.fract(); // 0 born, 1 gone
const emberSeed = float(instanceIndex).add(emberClock.floor().mul(MAX_EMBERS)); // a new spot each cycle
const emberAngle = hash(emberSeed.add(1)).mul(Math.PI * 2);
const emberRadius = hash(emberSeed.add(2)).sqrt().mul(embersUniforms.spread);
const emberWobble = time.mul(embersUniforms.speed).mul(1.3).add(hash(emberSeed.add(3)).mul(6.283));
emberMaterial.positionNode = crownCentre.add(vec3(
  emberAngle.cos().mul(emberRadius).add(emberWobble.sin().mul(0.09).mul(emberLife)),
  hash(emberSeed.add(4)).sub(0.6).add(emberLife.mul(embersUniforms.rise)),
  emberAngle.sin().mul(emberRadius).add(emberWobble.mul(0.8).cos().mul(0.09).mul(emberLife)),
));
emberMaterial.scaleNode = embersUniforms.size.mul(mix(0.5, 1.3, hash(emberSeed.add(5)))).mul(emberLife.mul(0.6).oneMinus());
emberMaterial.colorNode = Fn(() => {
  const spark = smoothstep(0, 0.5, uv().sub(0.5).length()).oneMinus(); // soft round dot
  const flicker = time.mul(14).add(hash(instanceIndex.add(11)).mul(40)).sin().mul(0.25).add(0.75);
  const fade = smoothstep(0, 0.08, emberLife).mul(smoothstep(0.5, 1, emberLife).oneMinus());
  const heat = mix(color(0xff5a14), color(0xffcf73), hash(emberSeed.add(6)));
  const glow = heat.mul(embersUniforms.brightness).mul(spark).mul(flicker).mul(fade);
  emissive.assign(glow);
  return glow;
})();
const embers = new THREE.Sprite(emberMaterial);
embers.count = atmosphere.emberCount;
embers.frustumCulled = false;
scene.add(embers);

const atmosphereFolder = gui.addFolder('Embers (prototype)');
atmosphereFolder.add(atmosphere, 'embers').onChange((on) => (embers.visible = on));
atmosphereFolder.add(atmosphere, 'emberCount', 0, MAX_EMBERS, 1).name('ember count').onChange((v) => (embers.count = v));
atmosphereFolder.add(embersUniforms.brightness, 'value', 0, 8, 0.05).name('ember brightness');
atmosphereFolder.add(embersUniforms.size, 'value', 0.004, 0.06, 0.001).name('ember size');
atmosphereFolder.add(embersUniforms.rise, 'value', 0.2, 4, 0.05).name('ember rise');
atmosphereFolder.add(embersUniforms.spread, 'value', 0.1, 2, 0.01).name('ember spread');
atmosphereFolder.add(embersUniforms.speed, 'value', 0, 3, 0.01).name('ember speed');

let inFlight = false;
let frames = 0;
renderer.setAnimationLoop(() => {
  if (inFlight) return;
  inFlight = true;
  frame = (async () => {
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.1);
    updateCrownMotion(dt);
    crown.updateMatrixWorld();

    controls.update();
    camera.updateMatrixWorld();
    goldLayer.visible = !busy && fluid !== null && isolate.gold;
    if (!busy && fluid !== null && isolate.sim) {
      // gravity points down in the world, so in the crown's own frame it turns as it tilts
      crown.getWorldQuaternion(inverse).invert();
      gravity.set(0, -9.81, 0).applyQuaternion(inverse);
      fluid.ubos.gravity.value.copy(gravity);
      await fluid.step(renderer, dt);
      if (motionGlowParts) {
        motionGlow.frameDelta.value = dt;
        renderer.compute(motionGlowParts.compute);
      }
    }
    if (goldLayer.visible) {
      goldUniforms.crownMatrix.value.copy(crown.matrixWorld);
      towards(key, goldUniforms.keyDirection.value);
      goldUniforms.keyColor.value.copy(key.color).multiplyScalar(key.intensity);
      towards(backLight, goldUniforms.backDirection.value);
      goldUniforms.backColor.value.copy(backLight.color).multiplyScalar(backLight.intensity * 0.15);
      goldSurface.update(renderer, camera);
    }
    crownCentre.value.copy(crown.position);
    if (fx.post) pipeline.render();
    else renderer.render(scene, camera);
    if (veil && !veil.classList.contains('lifted')) {
      veil.classList.add('lifted');
      veil.addEventListener('transitionend', () => veil.remove(), { once: true });
    }
    // stats-gl reads renderer.info's timestamps; three fills them only when asked
    if (debug) {
      renderer.resolveTimestampsAsync('render');
      renderer.resolveTimestampsAsync('compute');
      stats.update();
      if (frames++ % 30 === 0) showState();
    }
  })().finally(() => (inFlight = false));
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
