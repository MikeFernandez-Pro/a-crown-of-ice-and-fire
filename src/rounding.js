// Rounds a mesh's edges without shrinking it: Taubin smoothing, alternating a
// shrinking Laplacian step (lambda) with an inflating one (mu), over a mesh
// welded on position so the faces stay joined.
import * as THREE from 'three/webgpu';
import { mergeVertices, toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

const LAMBDA = 0.5;
const MU = -0.53;

export function weld(geometry) {
  const bare = new THREE.BufferGeometry();
  bare.setAttribute('position', geometry.getAttribute('position').clone());
  if (geometry.index) bare.setIndex(geometry.index.clone());
  return mergeVertices(bare, 1e-5);
}

// welded: an indexed, position-only geometry from weld(); returns a new
// non-indexed geometry, smoothed `iterations` times, with normals creased at
// `creaseAngle` radians (small: crisp facets; PI: smooth all over)
export function rounded(welded, iterations, creaseAngle) {
  const position = welded.getAttribute('position');
  const index = welded.index.array;
  const count = position.count;

  // vertex neighbours, packed
  const degree = new Uint32Array(count);
  for (let t = 0; t < index.length; t += 3) {
    for (let e = 0; e < 3; e++) degree[index[t + e]] += 2;
  }
  const start = new Uint32Array(count + 1);
  for (let v = 0; v < count; v++) start[v + 1] = start[v] + degree[v];
  const neighbours = new Uint32Array(start[count]);
  const fill = start.slice(0, count);
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t], b = index[t + 1], c = index[t + 2];
    neighbours[fill[a]++] = b; neighbours[fill[a]++] = c;
    neighbours[fill[b]++] = c; neighbours[fill[b]++] = a;
    neighbours[fill[c]++] = a; neighbours[fill[c]++] = b;
  }

  let p = Float32Array.from(position.array);
  let next = new Float32Array(p.length);
  const step = (factor) => {
    for (let v = 0; v < count; v++) {
      let x = 0, y = 0, z = 0;
      const n = start[v + 1] - start[v];
      for (let k = start[v]; k < start[v + 1]; k++) {
        const u = neighbours[k] * 3;
        x += p[u]; y += p[u + 1]; z += p[u + 2];
      }
      const o = v * 3;
      next[o] = p[o] + factor * (x / n - p[o]);
      next[o + 1] = p[o + 1] + factor * (y / n - p[o + 1]);
      next[o + 2] = p[o + 2] + factor * (z / n - p[o + 2]);
    }
    [p, next] = [next, p];
  };
  for (let i = 0; i < iterations; i++) {
    step(LAMBDA);
    step(MU);
  }

  const smoothed = new THREE.BufferGeometry();
  smoothed.setAttribute('position', new THREE.BufferAttribute(p, 3));
  smoothed.setIndex(welded.index.clone());
  const result = toCreasedNormals(smoothed, creaseAngle);
  smoothed.dispose();
  return result;
}
