#import "shaders/bindings.wgsl"::{VertexInfo, DispatchIndirectArgs, input_tex, index_lookup, vertex_buffer,
                                  index_buffer, counts, adaptivity_counts, debug_tex};

@group(1) @binding(0) var normal_tex: texture_3d<f32>;
@group(1) @binding(1) var normal_samp: sampler;

const VERTICES: array<vec3<u32>, 8> =
    array<vec3<u32>, 8>(
        vec3<u32>(0, 0, 0),
        vec3<u32>(1, 0, 0),
        vec3<u32>(1, 1, 0),
        vec3<u32>(0, 1, 0),
        vec3<u32>(0, 0, 1),
        vec3<u32>(1, 0, 1),
        vec3<u32>(1, 1, 1),
        vec3<u32>(0, 1, 1),
    );

const AXES: array<vec3<u32>, 3> =
    array<vec3<u32>, 3>(
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(0, 0, 1),
    );

const AXES_SELECTION: array<vec3<u32>, 3> =
    array<vec3<u32>, 3>(
        vec3(0, 1, 2),
        vec3(1, 0, 2),
        vec3(1, 2, 0),
    );

fn sample_grad(a: vec3<u32>, b: vec3<u32>, x: f32) -> vec3<f32> {
    let t_a = textureLoad(normal_tex, a, 0).xyz;
    let t_b = textureLoad(normal_tex, b, 0).xyz;

    return mix(t_a, t_b, x);
}

fn sample_sdf(pos: vec3<u32>) -> f32 {
    return textureLoad(input_tex, pos.xyz).x;
}

fn inside(x: f32) -> bool {
    return x <= 0.0;
}

fn outside(x: f32) -> bool {
    return x > 0.0;
}

fn is_edge(a: f32, b: f32) -> bool {
    return (inside(a) && outside(b)) ||
        (outside(a) && inside(b));
}

fn adapt(v0: f32, v1: f32) -> f32 {
    return (0.0 - v0) / (v1 - v0);
}

struct Plane {
    pos: vec3<f32>,
    normal: vec3<f32>,
};

var<workgroup> planes: array<Plane, 12>;
var<workgroup> n_edges: atomic<u32>;
var<workgroup> forces: array<vec3<f32>, 8>;

var<workgroup> wg_vtx_pos: vec3<f32>;

@compute @workgroup_size(3, 2, 2)
fn compute_adaptivity(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) invocation: vec3<u32>,
) {
    let vtx_id = wg_id.x;
    let vtx = vertex_buffer[vtx_id];
    let vtx_pos = vec3<u32>(floor(vec3(vtx.x, vtx.y, vtx.z)));

    let axis_id = invocation.x;
    let selection = AXES_SELECTION[axis_id];

    // the axis our edge is along
    let axis = AXES[selection.x];
    // the two perpendicular axes
    let a = AXES[selection.y] * invocation.y;
    let b = AXES[selection.z] * invocation.z;

    // we keep start and end in the cell-local coordinate space.
    let start = a + b;
    let end = start + axis;

    // calculate global position and sample
    let sdf_a = sample_sdf(vtx_pos + start);
    let sdf_b = sample_sdf(vtx_pos + end);

    if is_edge(sdf_a, sdf_b) {
        let pos_index = atomicAdd(&n_edges, u32(1));
        let intersection_pos = adapt(sdf_a, sdf_b);
        let normal = sample_grad(vtx_pos + start, vtx_pos + end, intersection_pos);
        let new_vtx_pos = mix(vec3<f32>(start), vec3<f32>(end), intersection_pos);

        planes[pos_index] = Plane(new_vtx_pos, normal);
    }

    workgroupBarrier();

    // how many planes we ended up with
    let n = atomicLoad(&n_edges);

    let index = calc_index(wg_id);

    // I can't figure out a way to do this in a uniform way, there
    // doesn't seem to be a way to perform atomic add on a vector
    //
    // average all intersection positions and make that the vertex
    // starting point
    var pos_calc = vec3<f32>(0.0);
    for (var i = 0; i < i32(n); i++) {
        pos_calc += planes[i].pos;
    }
    pos_calc = pos_calc / f32(n);

    let final_pos = pos_calc + vec3<f32>(vtx_pos);

    // this code is the same no matter what sooo we should be good
    vertex_buffer[vtx_id].x = final_pos.x;
    vertex_buffer[vtx_id].y = final_pos.y;
    vertex_buffer[vtx_id].z = final_pos.z;

    // we only have 8 corners unfortunately, some of them are gonna
    // have to go
    if index <= 8 && false {
        let corner_pos = VERTICES[index];

        const N_ITER: i32 = 3;

        let force = schmitz(n, corner_pos);
        forces[index] = force;
        // this is theoretically a non-uniform workgroupBarrier, I
        // hope it's ok tho

        for (var iter = 0; iter < N_ITER; iter++) {
            workgroupBarrier();
            let combined_force = trilinear_add();
            if index == 0 {
                wg_vtx_pos += combined_force;
            }
        }

        let final_pos = wg_vtx_pos + vec3<f32>(vtx_pos);

        // this code is the same no matter what sooo we should be good
        vertex_buffer[vtx_id].x = final_pos.x;
        vertex_buffer[vtx_id].y = final_pos.y;
        vertex_buffer[vtx_id].z = final_pos.z;
    }
}

fn calc_index(pos: vec3<u32>) -> u32 {
    return pos.x * 4 + pos.y * 2 + pos.z;
}

fn get_force(x: i32, y: i32, z: i32) -> vec3<f32> {
    return forces[x * 4 + y * 2 + z];
}

fn trilinear_add() -> vec3<f32> {
    // code very inspired by https://github.com/Tuntenfisch/Voxels

    let pos = wg_vtx_pos;

    let alpha = pos.x;

    let c00 = mix(get_force(0, 0, 0), get_force(1, 0, 0), alpha);
    let c01 = mix(get_force(0, 0, 1), get_force(1, 0, 1), alpha);
    let c10 = mix(get_force(0, 1, 0), get_force(1, 1, 0), alpha);
    let c11 = mix(get_force(0, 1, 1), get_force(1, 1, 1), alpha);

    let beta = pos.y;

    let c0 = mix(c00, c10, beta);
    let c1 = mix(c01, c11, beta);

    let gamma = pos.z;

    return mix(c0, c1, gamma);
}

// compute the Schmitz Particle approximation to figure out the vertex
// location
fn schmitz(n: u32, corner_pos: vec3<u32>) -> vec3<f32> {
    let pos = vec3<f32>(corner_pos);

    var force = vec3<f32>(0.0);

    for (var i = 0; i < i32(n); i++) {
        let p = project_onto_plane(planes[i], pos);
        force += p;
    }

    force *= 0.05;

    return force;
}

fn project_onto_plane(plane: Plane, point: vec3<f32>) -> vec3<f32> {
    let ab = point - plane.pos;
    let ab_proj = dot(ab, plane.normal) * plane.normal;
    let proj = point - ab_proj;

    return proj;
}
