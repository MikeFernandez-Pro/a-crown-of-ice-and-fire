// Marching cubes over a sampled signed distance field: the surface where the
// field equals `iso`, with normals from the field's gradient.
import { edgeTable, triTable } from 'three/addons/objects/MarchingCubes.js';

// corners and edges in the order of three's (Paul Bourke's) tables
const CORNERS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
  [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];

// field: { data: Float32Array, dims: [nx, ny, nz], min: [x, y, z], voxelSize },
// samples at the cell centres min + (i + 0.5) * voxelSize
export function isosurface(field, iso) {
  const { data, dims, min, voxelSize: v } = field;
  const [nx, ny, nz] = dims;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const value = (i, j, k) => data[at(i, j, k)] - iso;

  const positions = [];
  const normals = [];
  const index = [];
  const vertexOf = new Map(); // grid edge -> vertex, shared by the four cubes round it

  const gradient = (x, y, z, out) => {
    // central differences of the nearest samples round a point in grid coordinates
    const i = Math.min(Math.max(Math.round(x), 1), nx - 2);
    const j = Math.min(Math.max(Math.round(y), 1), ny - 2);
    const k = Math.min(Math.max(Math.round(z), 1), nz - 2);
    const gx = data[at(i + 1, j, k)] - data[at(i - 1, j, k)];
    const gy = data[at(i, j + 1, k)] - data[at(i, j - 1, k)];
    const gz = data[at(i, j, k + 1)] - data[at(i, j, k - 1)];
    const l = Math.hypot(gx, gy, gz) || 1;
    out[0] = gx / l;
    out[1] = gy / l;
    out[2] = gz / l;
  };

  const g = [0, 0, 0];
  const edgeVertex = (i, j, k, a, b) => {
    const A = CORNERS[a], B = CORNERS[b];
    const axis = A[0] !== B[0] ? 0 : A[1] !== B[1] ? 1 : 2;
    const lo = A[axis] < B[axis] ? A : B;
    const hi = lo === A ? B : A;
    const li = i + lo[0], lj = j + lo[1], lk = k + lo[2];
    const key = at(li, lj, lk) * 3 + axis;
    let vertex = vertexOf.get(key);
    if (vertex !== undefined) return vertex;
    const fa = value(li, lj, lk);
    const fb = value(i + hi[0], j + hi[1], k + hi[2]);
    const t = fa / (fa - fb);
    const p = [li, lj, lk];
    p[axis] += t;
    positions.push(min[0] + (p[0] + 0.5) * v, min[1] + (p[1] + 0.5) * v, min[2] + (p[2] + 0.5) * v);
    gradient(p[0], p[1], p[2], g);
    normals.push(g[0], g[1], g[2]);
    vertex = positions.length / 3 - 1;
    vertexOf.set(key, vertex);
    return vertex;
  };

  for (let k = 0; k < nz - 1; k++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        let cube = 0;
        for (let n = 0; n < 8; n++) {
          const c = CORNERS[n];
          if (value(i + c[0], j + c[1], k + c[2]) < 0) cube |= 1 << n;
        }
        if (edgeTable[cube] === 0) continue;
        // the tables wind every triangle the same way round; which way is
        // settled once for the whole mesh below (a per-triangle test against the
        // sampled gradient flips some where a wall is only a few voxels thick)
        for (let t = cube * 16; triTable[t] !== -1; t += 3) {
          index.push(
            edgeVertex(i, j, k, ...EDGES[triTable[t]]),
            edgeVertex(i, j, k, ...EDGES[triTable[t + 1]]),
            edgeVertex(i, j, k, ...EDGES[triTable[t + 2]]),
          );
        }
      }
    }
  }
  // a closed surface wound outward encloses a positive volume
  let volume = 0;
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t] * 3, b = index[t + 1] * 3, c = index[t + 2] * 3;
    volume += positions[a] * (positions[b + 1] * positions[c + 2] - positions[b + 2] * positions[c + 1])
      + positions[a + 1] * (positions[b + 2] * positions[c] - positions[b] * positions[c + 2])
      + positions[a + 2] * (positions[b] * positions[c + 1] - positions[b + 1] * positions[c]);
  }
  if (volume < 0) {
    for (let t = 0; t < index.length; t += 3) {
      const b = index[t + 1];
      index[t + 1] = index[t + 2];
      index[t + 2] = b;
    }
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    index: new Uint32Array(index),
  };
}
