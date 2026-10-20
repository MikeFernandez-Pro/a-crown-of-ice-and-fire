// A model's signed distance field, computed on the GPU when the page loads,
// from its triangles, with three-mesh-bvh's compute traversals: no baked file.
//
// The grid is the one stream1018-tools/bake-sdf.mjs laid out: `samples` cells
// along the longest side of the bounding box (widened by `margin`), PAD cells
// of padding all round, values at the cell centres, negative inside. Two
// passes:
//
// 1. The sign, by winding count. The model is many closed crystals, some
//    overlapping, and a few with holes, so the closest face's side is not to
//    be trusted. Instead one thread per grid line casts a ray down it and
//    counts: entering a front face adds one, leaving a back face takes one;
//    a cell is inside where the count is positive. One family of lines per
//    axis, and the three axes vote.
// 2. The distance to the closest surface point, signed by the vote: all three
//    axes agree, or two do right by the surface (a split vote far from any
//    surface is a ray family slipping through a hole).
//
// The result is read back to the CPU for marching cubes and seeding.
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, Break, globalId, float, int, uint, vec3, instancedArray, wgslFn, select } from 'three/tsl';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';

const PAD = 3; // voxels of margin round the bounding box
const MAX_HITS = 64; // surfaces a line may cross between two cell centres

// geometry: in the space the field is wanted in (node transforms applied);
// returns { data: Float32Array, dims, min, voxelSize, modelSize }
export async function distanceField(geometry, renderer, { samples = 198, margin = 0 } = {}) {
  geometry.computeBoundingBox();
  const modelExtent = geometry.boundingBox.getSize(new THREE.Vector3());
  const box = geometry.boundingBox.clone().expandByScalar(margin);
  const extent = box.getSize(new THREE.Vector3());
  const voxelSize = Math.max(extent.x, extent.y, extent.z) / (samples - 2 * PAD);
  const dims = ['x', 'y', 'z'].map((a) => Math.ceil(extent[a] / voxelSize) + 2 * PAD);
  const centre = box.getCenter(new THREE.Vector3());
  const min = dims.map((n, i) => centre.getComponent(i) - (n * voxelSize) / 2);
  const [nx, ny, nz] = dims;
  const cells = nx * ny * nz;
  const strides = [1, nx, nx * ny];

  // the triangles and their BVH, packed for the compute shaders
  const triangles = withoutSlivers(geometry);
  const bvh = new BVHComputeData(triangles);
  bvh.update();

  // WGSL round the traversals: the next surface along a ray, as (distance,
  // side: +1 entering a front face, -1 leaving through a back face), or
  // (-1, 0) when there is none; and the distance to the closest surface point
  const nextHit = wgslFn(/* wgsl */ `
    fn nextHit( origin: vec3f, direction: vec3f ) -> vec2f {
      var result: IntersectionResult;
      let ray = Ray( origin, direction, 0.0 );
      bvh_RaycastFirstHit( ray, &result );
      return select( vec2f( -1.0, 0.0 ), vec2f( result.dist, result.side ), result.didHit );
    }
  `, [bvh.fns.raycastFirstHit]);
  const closestDistance = wgslFn(/* wgsl */ `
    fn closestDistance( point: vec3f ) -> f32 {
      var result: PointQueryResult;
      bvh_ClosestPointToPoint( point, &result );
      return sqrt( max( result.distanceSq, 0.0 ) );
    }
  `, [bvh.fns.closestPointToPoint]);

  const votes = instancedArray(cells, 'uint');
  const field = instancedArray(cells, 'float');

  // --- pass 1: one thread per line along axis w, over the other two axes u, v
  const LINE_GROUP = [8, 8, 1];
  const winding = (w) => {
    const u = (w + 1) % 3, v = (w + 2) % 3;
    const direction = vec3(...[0, 1, 2].map((a) => (a === w ? 1 : 0)));
    const step = 1e-3 * voxelSize; // past a hit before casting on
    return Fn(() => {
      const a = globalId.x, b = globalId.y;
      If(a.lessThan(dims[u]).and(b.lessThan(dims[v])), () => {
        const at = [0, 0, 0];
        at[u] = float(a).add(0.5).mul(voxelSize).add(min[u]);
        at[v] = float(b).add(0.5).mul(voxelSize).add(min[v]);
        at[w] = float(min[w] - voxelSize); // a voxel before the grid
        const start = vec3(at[0], at[1], at[2]);
        const base = a.mul(strides[u]).add(b.mul(strides[v]));

        const hit = nextHit({ origin: start, direction }).toVar();
        const travelled = float(0).toVar(); // from start to where the last cast began
        const count = int(0).toVar();
        Loop(dims[w], ({ i }) => {
          // start to this cell's centre. A variable, so it is written here, in
          // this loop; an expression would be inlined where it is read, inside
          // the next loop, whose own counter shadows `i`
          const t = float(i).add(1.5).mul(voxelSize).toVar();
          Loop(MAX_HITS, () => {
            If(hit.x.lessThan(0).or(travelled.add(hit.x).greaterThanEqual(t)), () => {
              Break();
            });
            count.addAssign(int(hit.y));
            travelled.addAssign(hit.x.add(step));
            hit.assign(nextHit({ origin: start.add(direction.mul(travelled)), direction }));
          });
          If(count.greaterThan(0), () => {
            votes.element(base.add(uint(i).mul(strides[w]))).addAssign(uint(1));
          });
        });
      });
    })().computeKernel(LINE_GROUP);
  };

  // --- pass 2: one thread per cell
  const CELL_GROUP = [4, 4, 4];
  const distance = Fn(() => {
    const id = globalId;
    If(id.x.lessThan(nx).and(id.y.lessThan(ny)).and(id.z.lessThan(nz)), () => {
      const cell = id.x.add(id.y.mul(nx)).add(id.z.mul(nx * ny));
      const point = vec3(float(id.x), float(id.y), float(id.z)).add(0.5).mul(voxelSize).add(vec3(...min));
      const d = closestDistance({ point });
      const vote = votes.element(cell);
      const inside = vote.equal(uint(3)).or(vote.equal(uint(2)).and(d.lessThan(2 * voxelSize)));
      field.element(cell).assign(select(inside, d.negate(), d));
    });
  })().computeKernel(CELL_GROUP);

  const kernels = [];
  for (let w = 0; w < 3; w++) {
    const u = (w + 1) % 3, v = (w + 2) % 3;
    const kernel = winding(w);
    kernels.push(kernel);
    await renderer.computeAsync(kernel, [Math.ceil(dims[u] / LINE_GROUP[0]), Math.ceil(dims[v] / LINE_GROUP[1]), 1]);
  }
  kernels.push(distance);
  await renderer.computeAsync(distance, dims.map((n, i) => Math.ceil(n / CELL_GROUP[i])));
  const data = new Float32Array(await renderer.getArrayBufferAsync(field.value));

  for (const kernel of kernels) kernel.dispose();
  votes.value.dispose();
  field.value.dispose();
  bvh.dispose();
  triangles.dispose();

  return {
    data,
    dims,
    min,
    voxelSize,
    modelSize: Math.max(modelExtent.x, modelExtent.y, modelExtent.z),
  };
}

// The positions and triangles of a geometry, without its zero-area triangles:
// the crown has a few dozen, and the GPU closest-point query comes out NaN
// when one of them is the first triangle it tests, which then stands, as
// nothing compares less than NaN.
function withoutSlivers(geometry) {
  const position = geometry.getAttribute('position');
  const source = geometry.index ? geometry.index.array : Uint32Array.from({ length: position.count }, (_, i) => i);
  const kept = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  const tiny = 1e-10 * Math.max(size.x, size.y, size.z) ** 2;
  for (let t = 0; t < source.length; t += 3) {
    a.fromBufferAttribute(position, source[t]);
    b.fromBufferAttribute(position, source[t + 1]);
    c.fromBufferAttribute(position, source[t + 2]);
    if (b.sub(a).cross(c.sub(a)).length() > tiny) kept.push(source[t], source[t + 1], source[t + 2]);
  }
  const clean = new THREE.BufferGeometry();
  clean.setAttribute('position', position.clone());
  clean.setIndex(kept);
  clean.computeBoundingBox();
  return clean;
}
