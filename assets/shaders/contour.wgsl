
#import "shaders/bindings.wgsl"::{VertexInfo, DispatchIndirectArgs, input_tex, index_lookup, vertex_buffer,
                                  index_buffer, counts, adaptivity_counts, debug_tex};

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

fn calc_vtx_pos(cell_pos: vec3<u32>) -> vec3<f32> {
    return vec3(0.0);
}

fn sample(pos: vec3<u32>) -> f32 {
    return textureLoad(input_tex, pos).x;
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


// returns the vertex index
fn write_vertex(vtx_index: u32, vtx: vec3<f32>) {
    var v: VertexInfo;

    v.x = vtx.x;
    v.y = vtx.y;
    v.z = vtx.z;

    vertex_buffer[vtx_index] = v;
}

struct ComputeInput {
    @builtin(global_invocation_id) global_id: vec3<u32>,
    // @builtin(workgroup_id) local_index: u32,
};

@compute
@workgroup_size(1, 1, 1)
fn compute_vertices(input: ComputeInput) {
    let global_id = input.global_id;

    var samples: u32 = u32(0);
    for (var i: u32 = 0; i < 8; i++) {
        let corner_sample = u32(textureLoad(input_tex, global_id + VERTICES[i]).x < 0.0);

        samples = samples | (corner_sample << i);
    }

    let is_air = samples == 0;
    let is_solid = samples == 255;
    let is_surface = !is_air && !is_solid;

    if (is_surface) {
        let vtx_index = atomicAdd(&counts.vtx, u32(1));
        atomicAdd(&adaptivity_counts.x, u32(1));
        textureStore(index_lookup, global_id, vec4(vtx_index, 0, 0, 0));
        write_vertex(vtx_index, vec3<f32>(global_id) + calc_vtx_pos(global_id));
    }

    textureStore(debug_tex, global_id, vec4(0.0, f32(samples), 0.0, 0.0));
}

const AXES: array<vec3<u32>, 3> =
    array<vec3<u32>, 3>(
                        vec3(1, 0, 0),
                        vec3(0, 1, 0),
                        vec3(0, 0, 1),
    );

const AXIS_TABLE: array<bool, 6> = array<bool, 6>(
    // negative axes
    true,
    false,
    false,
    // positive axes
    false,
    true,
    true,
);

@compute @workgroup_size(1, 1, 1)
fn compute_edges(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_pos = global_id;

    var samples = vec4(sample(global_id), 0.0, 0.0, 0.0);

    for (var i: u32 = 0; i < 3; i++) {
        let start = global_id;
        let end = start + AXES[i];

        let a = sample(start);
        let b = sample(end);

        // if a && !b then these polygons should be facing a->b
        // if !a && b then they should be facing b->a
        //
        // +x: false
        // -x: true
        // +y: true
        // -y: false
        // +z: true
        // -z: false

        samples[i+1] = b;

        let is_positive = inside(a);

        if is_edge(a, b) {
            let winding_index = i + u32(is_positive) * 3;
            let winding = AXIS_TABLE[u32(is_positive) * 3 + i];
            gen_face(global_id, i, winding);
        }
    }
}

const OFFSETS_X: array<vec3<i32>, 4> =
    array<vec3<i32>, 4>(
        vec3(0, 0, 0),
        vec3(0, -1, 0),
        vec3(0, 0, -1),
        vec3(0, -1, -1),
    );

const OFFSETS_Y: array<vec3<i32>, 4> =
    array<vec3<i32>, 4>(
        vec3(0, 0, 0),
        vec3(-1, 0, 0),
        vec3(0, 0, -1),
        vec3(-1, 0, -1),
    );

const OFFSETS_Z: array<vec3<i32>, 4> =
    array<vec3<i32>, 4>(
        vec3(0, 0, 0),
        vec3(0, -1, 0),
        vec3(-1, 0, 0),
        vec3(-1, -1, 0),
    );

const EDGE_OFFSETS: array<array<vec3<i32>, 4>, 3> =
    array<array<vec3<i32>, 4>, 3>(
        OFFSETS_X,
        OFFSETS_Y,
        OFFSETS_Z,
    );

fn gen_face(pos: vec3<u32>, axis: u32, invert_winding: bool) {
    let offsets = EDGE_OFFSETS[axis];

    var vtx_indices: array<u32, 4> = array<u32, 4>(0, 0, 0, 0);

    for (var i: u32 = 0; i < 4; i++) {
        let cell_pos = vec3<u32>(vec3<i32>(pos) + offsets[i]);

        vtx_indices[i] = textureLoad(index_lookup, cell_pos).x;
    }

    write_quad(vtx_indices, invert_winding);
}

fn write_quad(indices: array<u32, 4>, invert_winding: bool) {
    // vtx orders: 0 1 2, 1 3 2
    let base = atomicAdd(&counts.idx, u32(6));

    if invert_winding {
        // tri1
        index_buffer[base+0] = indices[1];
        index_buffer[base+1] = indices[0];
        index_buffer[base+2] = indices[2];

        // tri2
        index_buffer[base+3] = indices[3];
        index_buffer[base+4] = indices[1];
        index_buffer[base+5] = indices[2];
    } else {
        // tri1
        index_buffer[base+0] = indices[0];
        index_buffer[base+1] = indices[1];
        index_buffer[base+2] = indices[2];

        // tri2
        index_buffer[base+3] = indices[1];
        index_buffer[base+4] = indices[3];
        index_buffer[base+5] = indices[2];
    }
}

@compute @workgroup_size(1, 1, 1)
fn cleanup() {
    atomicStore(&counts.vtx, u32(0));
    atomicStore(&counts.idx, u32(0));
    atomicStore(&adaptivity_counts.x, u32(0));
}

