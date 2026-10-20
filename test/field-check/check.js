// Computes a model's field live (src/field.js) and prints its stats; if a
// baked file (stream1018-tools/bake-sdf.mjs, 200 or 256 samples, --margin
// 0.03) is put back in public/, compares the two cell by cell.
//   https://localhost:5190/test/field-check/   (npm run dev)
//   node test/field-check/run.mjs   (headless)
import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { distanceField } from '../../src/field.js';

const MODELS = {
  crown: { url: '/ice-crystal-crown.glb', samples: 200, baked: '/crystal-field.bin' },
};
const choice = MODELS[new URLSearchParams(location.search).get('model')] ?? MODELS.crown;
const out = document.getElementById('out');
const log = (line) => {
  out.textContent += `\n${line}`;
  console.log(line);
};

const renderer = new THREE.WebGPURenderer();
await renderer.init();

const model = await new GLTFLoader().loadAsync(choice.url);
let mesh;
model.scene.traverse((o) => o.isMesh && (mesh = o));
model.scene.updateMatrixWorld(true);
const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);

const t0 = performance.now();
const field = await distanceField(geometry, renderer, { samples: choice.samples, margin: 0.03 });
const ms = performance.now() - t0;
const { data, dims, min, voxelSize, modelSize } = field;
let inside = 0, deepest = 0;
for (const d of data) {
  if (d < 0) inside++;
  deepest = Math.min(deepest, d);
}
log(`live: ${ms.toFixed(0)} ms, dims ${dims}, min ${min.map((x) => x.toFixed(6))}, voxel ${voxelSize.toFixed(8)}, modelSize ${modelSize}`);
log(`live: ${((100 * inside) / data.length).toFixed(2)}% inside, deepest ${deepest.toFixed(4)}`);

const response = await fetch(choice.baked);
if (!response.ok) {
  log('no baked file to compare with');
} else {
  const half = new Uint16Array(await response.arrayBuffer());
  if (half.length !== data.length) {
    log(`baked has ${half.length} cells, live ${data.length}: grids differ`);
  } else {
    const baked = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) baked[i] = THREE.DataUtils.fromHalfFloat(half[i]);
    let bakedInside = 0, signDiffers = 0, signDiffersNearSurface = 0, maxDiff = 0, sumDiff = 0;
    let signDiffersFar = 0;
    const hist = new Array(8).fill(0); // |diff| in voxels: <0.01, <0.1, <0.5, <1, <2, <4, <8, more
    for (let i = 0; i < data.length; i++) {
      if (baked[i] < 0) bakedInside++;
      const diff = Math.abs(data[i] - baked[i]);
      sumDiff += diff;
      maxDiff = Math.max(maxDiff, diff);
      const v = diff / voxelSize;
      hist[v < 0.01 ? 0 : v < 0.1 ? 1 : v < 0.5 ? 2 : v < 1 ? 3 : v < 2 ? 4 : v < 4 ? 5 : v < 8 ? 6 : 7]++;
      if ((data[i] < 0) !== (baked[i] < 0)) {
        signDiffers++;
        if (Math.min(Math.abs(data[i]), Math.abs(baked[i])) < voxelSize) signDiffersNearSurface++;
        else signDiffersFar++;
      }
    }
    log(`baked: ${((100 * bakedInside) / data.length).toFixed(2)}% inside`);
    log(`|live - baked|: mean ${(sumDiff / data.length).toExponential(2)}, max ${maxDiff.toFixed(5)} (${(maxDiff / voxelSize).toFixed(2)} voxels)`);
    log(`|diff| in voxels, cells: <0.01 ${hist[0]}, <0.1 ${hist[1]}, <0.5 ${hist[2]}, <1 ${hist[3]}, <2 ${hist[4]}, <4 ${hist[5]}, <8 ${hist[6]}, more ${hist[7]}`);
    log(`sign differs: ${signDiffers} cells (${signDiffersNearSurface} within a voxel of the surface, ${signDiffersFar} further)`);
    // the worst cells: where, and the two values
    const worst = [];
    const [nx, ny] = dims;
    for (let i = 0; i < data.length; i++) {
      const diff = Math.abs(data[i] - baked[i]);
      if (diff > 8 * voxelSize) worst.push([diff, i]);
    }
    worst.sort((p, q) => q[0] - p[0]);
    const where = (i) => [i % nx, Math.floor(i / nx) % ny, Math.floor(i / (nx * ny))];
    let border = 0, liveBigger = 0;
    for (const [, i] of worst) {
      const g = where(i);
      if (g.some((c, a) => c < 2 || c >= dims[a] - 2)) border++;
      if (Math.abs(data[i]) > Math.abs(baked[i])) liveBigger++;
    }
    log(`worst (> 8 voxels): ${worst.length}, of which ${border} within 2 cells of the grid's border, ${liveBigger} with live further than baked`);
    for (const [diff, i] of worst.slice(0, 12)) log(`  cell ${where(i)} live ${data[i].toFixed(4)} baked ${baked[i].toFixed(4)}`);
    const sample = worst.filter((_, k) => k % Math.ceil(worst.length / 8) === 0);
    for (const [diff, i] of sample) log(`  cell ${where(i)} live ${data[i].toFixed(4)} baked ${baked[i].toFixed(4)}`);
  }
}
log('done');
