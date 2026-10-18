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
  vec4,
  uniform,
  color,
  mix,
  smoothstep,
  screenUV,
  positionGeometry,
  pmremTexture,
  getViewPosition,
  viewZToPerspectiveDepth,
  cameraNear,
  cameraFar,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshBVH } from 'three-mesh-bvh';
import GUI from 'lil-gui';
import { PBF } from 'three-blocks/pbf';
import { ComputeSDFGenerator, SDFVolumeConstraint } from 'three-blocks/sdf-raymarching';
import { WaterSurfaceRenderer } from 'three-blocks/water';
import { MeshTransmissionNodeMaterial } from 'three-blocks/transmission';
import { isosurface } from './marching.js';
import { weld, rounded } from './rounding.js';

const MODEL = '/ice-crystal-crown.glb';
// what bake-sdf.mjs printed for crystal-field.bin:
//   node bake-sdf.mjs ice-crystal-crown.glb crystal-field.bin 200 --margin 0.03
const FIELD = {
  url: '/crystal-field.bin',
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
const FIXED_STEP = 1 / 120;

// distances are in scene units (the crown is 2 across)
const settings = {
  // ice
  sharpness: 1,
  iceThickness: 0.012,
  frost: 0.35, // surface roughness: how soft the reflections and the blur behind are
  blur: 1.8, // extra blur on what shows through, beyond what the roughness gives
  milkiness: 0.12, // how much of the ice's own white shows instead of what is behind
  ior: 1.31,
  iceTint: '#bfe0ff',
  clarity: 0.6,
  grain: 0.06,
  grainSize: 2.5,
  prism: 0.02,
  // gold
  fill: 0.6,
  particles: 32768,
  viscosity: 0.35,
  flowSpeed: 2.5,
  blobSize: 1.6,
  goldColor: '#ffd98a',
  goldRoughness: 0.18,
  heat: 0.12,
  heatColor: '#ff6a1a',
  // crown
  tilt: 0,
  rock: false,
  rockAmount: 0.35,
  spin: 0,
  remelt: () => rebuildFluid(),
};

const toField = (sceneDistance) => sceneDistance / (DISPLAY * SIM);

// --- renderer, scene, camera
const renderer = new THREE.WebGPURenderer({ antialias: true });
// the ice's refraction and the gold are both per-pixel work: 1x keeps them affordable
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
await renderer.init();
if (!renderer.backend.isWebGPUBackend) {
  document.body.insertAdjacentHTML('beforeend', '<p class="note">This page needs WebGPU: the fluid runs in compute shaders.</p>');
  throw new Error('WebGPU is required');
}

const scene = new THREE.Scene();
// a cool lit backdrop: clear ice only shows what is behind it, and over black it reads as black glass
scene.backgroundNode = mix(color(0x101218), color(0x6f7f96), smoothstep(0.15, 1.0, screenUV.y.oneMinus()));
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(0, 2.2, 5.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.8, 0);

const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(2, 4, 3);
scene.add(key);
const backLight = new THREE.PointLight(0xffd2a1, 12, 0, 2);
backLight.position.set(0, 1.6, -2.5);
scene.add(backLight);

// the crown: everything inside is in simulation units
const crown = new THREE.Group();
crown.scale.setScalar(DISPLAY);
scene.add(crown);

// --- the field
const field = await fetch(FIELD.url)
  .then((r) => r.arrayBuffer())
  .then((buffer) => {
    const half = new Uint16Array(buffer);
    const data = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) data[i] = THREE.DataUtils.fromHalfFloat(half[i]);
    return { ...FIELD, data };
  });

// the model, welded once for rounding; its centre, in field units, puts the
// simulation round the origin
const model = await new GLTFLoader().loadAsync(MODEL);
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
  color: 0xf2f8ff,
  roughness: settings.frost,
  ior: settings.ior,
  // The material measures in different spaces, and the crown group is scaled
  // by DISPLAY: thickness is in local units times the model's world scale (so
  // 0.08 / DISPLAY is 0.08 scene units, about one crystal wall), distortionScale
  // in local units (the crown is 40 across), ditherScale in world units (the
  // crown is 2 across). At the library example's proportions the grain stays
  // above a pixel and the dither below one, so neither reads as blocky noise.
  thickness: 0.08 / DISPLAY,
  transmission: 1 - settings.milkiness,
  blurScale: settings.blur,
  chromaticAberration: settings.prism,
  anisotropicBlur: 0.05,
  surfaceDistortion: settings.grain,
  distortionScale: settings.grainSize,
  ditherScale: 640,
  temporalDistortion: 0,
  attenuationColor: new THREE.Color(settings.iceTint),
  attenuationDistance: settings.clarity,
  backdropDistance: 0.6,
  samples: 6, // as drei's MeshTransmissionMaterial: frost blurs wide, and fewer taps across it read as grain
});
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
  // lift the crown so its base sits at y = 0, round the camera's target
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
  keyDirection: uniform(new THREE.Vector3()), // towards the key light, view space
  keyColor: uniform(new THREE.Color()),
  backDirection: uniform(new THREE.Vector3()), // towards the back light, view space
  backColor: uniform(new THREE.Color()),
  crownMatrix: uniform(new THREE.Matrix4()),
};
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
    const glow = goldUniforms.heatColor.mul(goldUniforms.heat).mul(smoothstep(0.02, 0.6, thickness));
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
  // soft and held half a spacing in: upside down the gold's weight lands on the
  // thin spike tips, and a stiffer wall let particles squeeze through them
  const response = { stiffness: 1e3, damping: 0.3, threshold: -0.5 * spacing };
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
    useDirection: false, // the smoothed directions are for instanced meshes; the splats don't read them
    useMatrices: false,
    useSpatialGrid: true,
    // build the neighbour grid by counting into cells rather than sorting
    // (the default bitonic sort is ~136 passes a substep), which also allows
    // the shared-memory lookup. Given as the legacy spatialGridOptions: the
    // structured `neighbors` option resets count, domain and kernel to defaults.
    spatialGridOptions: { buildAlgorithm: 'atomic', lookupAlgorithm: 'workgroup' },
    scaleKernelWithDomain: false,
    fixedTimeStep: FIXED_STEP,
    sdfVolumeConstraint: constraint,
    h: spacing * 2,
    spacing,
    particleRadius,
    restDensity: null,
    solverIterations: 2,
    // a slow frame slows the melt rather than queueing more steps (default 5).
    // The step itself stays 1/120: at 1/60 the gold crosses the thin walls.
    maxSubsteps: 2,
    lambdaEpsilon: 5,
    xsphViscosity: settings.viscosity,
    corrK: 0,
    corrN: 4,
  });
  fluid.ubos.timeScale.value = settings.flowSpeed;
  await fluid.step(renderer, FIXED_STEP);
  cavityGeometry.dispose();

  // the surface renderer reads world positions: the particles live in the crown's frame
  const positions = fluid.buffers.positions;
  const worldPositions = {
    element: (i) => goldUniforms.crownMatrix.mul(vec4(positions.element(i), 1)),
  };
  goldSurface = new WaterSurfaceRenderer(renderer, { positions: worldPositions, count }, {
    worldRadius: spacing * settings.blobSize * DISPLAY,
    temporalBlend: 0, // the crown turns, and history would smear it
    filterIterations: 2,
    filterRadius: 10,
    // full resolution: at the default half, upsampling terraces the gold into
    // flat blocks with stepped edges (MSAA can't reach this full-screen layer)
    resolutionScale: 1,
    depthSmoothingRadius: 4,
    thicknessScale: 0.35,
  });
  goldLayer.userData.spacing = spacing;
  shadeGold(goldSurface);
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
  goldSurface?.dispose();
  fluid?.dispose();
  constraint?.dispose();
  sdf?.dispose();
  fluid = constraint = sdf = goldSurface = null;
}

function rebuildFluid() {
  rebuilding = rebuilding.then(async () => {
    busy = true;
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

// --- controls
const gui = new GUI({ title: 'Melted gold in ice' });
const iceFolder = gui.addFolder('Ice');
iceFolder.add(settings, 'sharpness', 0, 1, 0.01).onFinishChange(rebuildIce);
// the ice between the crown's surface and the gold
iceFolder.add(settings, 'iceThickness', 0.004, 0.03, 0.001).name('thickness').onFinishChange(rebuildFluid);
iceFolder.add(settings, 'frost', 0, 1, 0.01).onChange((v) => (ice.roughness = v));
iceFolder.add(settings, 'blur', 0, 4, 0.05).name('frost blur').onChange((v) => (ice.blurScale = v));
iceFolder.add(settings, 'milkiness', 0, 0.6, 0.01).onChange((v) => (ice.transmission = 1 - v));
iceFolder.add(settings, 'ior', 1, 1.6, 0.01).name('IOR').onChange((v) => (ice.ior = v));
iceFolder.addColor(settings, 'iceTint').name('tint').onChange((v) => ice.attenuationColor.set(v));
iceFolder.add(settings, 'clarity', 0.05, 5, 0.05).onChange((v) => (ice.attenuationDistance = v));
iceFolder.add(settings, 'grain', 0, 0.2, 0.001).onChange((v) => (ice.surfaceDistortion = v));
// in the crown's local units (40 across): higher is finer; past ~4 the grain
// drops under a pixel and the material fades it out
iceFolder.add(settings, 'grainSize', 0.3, 6, 0.05).name('grain fineness').onChange((v) => (ice.distortionScale = v));
iceFolder.add(settings, 'prism', 0, 0.5, 0.01).onChange((v) => (ice.chromaticAberration = v));

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

const crownFolder = gui.addFolder('Crown');
crownFolder.add(settings, 'tilt', -0.8, 0.8, 0.01).listen().onChange((v) => !settings.rock && orientation.setFromEuler(new THREE.Euler(v, 0, 0)));
crownFolder.add(settings, 'rock').name('rock it').listen();
crownFolder.add(settings, 'rockAmount', 0, 0.8, 0.01).name('rock amount');
crownFolder.add(settings, 'spin', 0, 1, 0.01);

// --- loop
const timer = new THREE.Timer();
const gravity = new THREE.Vector3();
const inverse = new THREE.Quaternion();
let spinAngle = 0;

// Grab the crown to turn it, all the way round. A drag that starts on the
// crown turns it about the camera's axes (left/right round the screen's
// vertical, up/down round its horizontal), so it rolls the way the mouse goes
// at any orientation, and the gold sloshes with it; a drag anywhere else orbits
// the camera. The capture listener runs before OrbitControls' own pointerdown,
// so it can switch the controls off first.
const orientation = new THREE.Quaternion(); // the crown's turn when not rocking
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const turn = new THREE.Quaternion();
const axis = new THREE.Vector3();
let grab = null;
renderer.domElement.addEventListener('pointerdown', (event) => {
  pointer.set((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  if (raycaster.intersectObject(iceMesh, false).length === 0) return;
  controls.enabled = false;
  if (settings.rock) orientation.copy(crown.quaternion);
  settings.rock = false;
  grab = { x: event.clientX, y: event.clientY };
  renderer.domElement.setPointerCapture(event.pointerId);
}, { capture: true });
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!grab) return;
  const dx = event.clientX - grab.x, dy = event.clientY - grab.y;
  grab.x = event.clientX;
  grab.y = event.clientY;
  axis.setFromMatrixColumn(camera.matrixWorld, 1); // the screen's vertical
  orientation.premultiply(turn.setFromAxisAngle(axis, dx * 0.01));
  axis.setFromMatrixColumn(camera.matrixWorld, 0); // the screen's horizontal
  orientation.premultiply(turn.setFromAxisAngle(axis, dy * 0.01));
});
const release = () => {
  grab = null;
  controls.enabled = true;
};
renderer.domElement.addEventListener('pointerup', release);
renderer.domElement.addEventListener('pointercancel', release);

// the direction from the crown towards a light, in view space
const towards = (light, out) =>
  out.copy(light.position).sub(crown.position).normalize().transformDirection(camera.matrixWorldInverse);

let inFlight = false;
renderer.setAnimationLoop(() => {
  if (inFlight) return;
  inFlight = true;
  frame = (async () => {
    timer.update();
    const dt = Math.min(timer.getDelta(), 0.1);
    const t = timer.getElapsed();
    spinAngle += dt * settings.spin;
    if (settings.rock) {
      settings.tilt = Math.sin(t * 0.9) * settings.rockAmount;
      crown.rotation.set(settings.tilt, spinAngle, Math.sin(t * 0.6) * settings.tilt * 0.5);
    } else {
      // spin turns the grabbed orientation about the world's vertical
      orientation.premultiply(turn.setFromAxisAngle(axis.set(0, 1, 0), dt * settings.spin));
      crown.quaternion.copy(orientation);
    }
    crown.updateMatrixWorld();

    controls.update();
    camera.updateMatrixWorld();
    goldLayer.visible = !busy && fluid !== null;
    if (goldLayer.visible) {
      // gravity points down in the world, so in the crown's own frame it turns as it tilts
      crown.getWorldQuaternion(inverse).invert();
      gravity.set(0, -9.81, 0).applyQuaternion(inverse);
      fluid.ubos.gravity.value.copy(gravity);
      await fluid.step(renderer, dt);

      goldUniforms.crownMatrix.value.copy(crown.matrixWorld);
      towards(key, goldUniforms.keyDirection.value);
      goldUniforms.keyColor.value.copy(key.color).multiplyScalar(key.intensity);
      towards(backLight, goldUniforms.backDirection.value);
      goldUniforms.backColor.value.copy(backLight.color).multiplyScalar(backLight.intensity * 0.15);
      goldSurface.update(renderer, camera);
    }
    renderer.render(scene, camera);
  })().finally(() => (inFlight = false));
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
