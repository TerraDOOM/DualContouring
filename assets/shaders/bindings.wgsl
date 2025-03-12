@group(0) @binding(0)
var input_tex: texture_storage_3d<r32float, read_write>;

@group(0) @binding(1)
var index_lookup: texture_storage_3d<r32uint, read_write>;

@group(0) @binding(2)
    var<storage, read_write> vertex_buffer: array<VertexInfo>;

struct VertexInfo {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    v: f32,
    n_x: f32,
    n_y: f32,
    n_z: f32,
};


@group(0) @binding(3)
    var<storage, read_write> index_buffer: array<u32>;

@group(0) @binding(4)
    var<storage, read_write> counts: Counts;

struct Counts {
    vtx: atomic<u32>,
    idx: atomic<u32>,
};

@group(0) @binding(5) var<storage, read_write> adaptivity_counts: DispatchIndirectArgs;

struct DispatchIndirectArgs {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
};

@group(0) @binding(6) var debug_tex: texture_storage_3d<rgba32float, read_write>;
