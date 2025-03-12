
#import "shaders/bindings.wgsl"::{input_tex, index_lookup, vertex_buffer, index_buffer, counts, debug_tex};

@group(1) @binding(0) var grad_tex: texture_storage_3d<rgba8snorm, write>;

struct Sphere {
    pos: vec3<f32>,
    radius: f32,
};

fn sdf(pos: vec3<f32>) -> f32 {
    let sphere = Sphere(vec3(2.5, 2.5, 2.5), 1.0);

    let dist = length(sphere.pos - pos) - sphere.radius;

    return dist;
}

struct ComputeInput {
    @builtin(global_invocation_id) global_id: vec3<u32>,
}

@compute @workgroup_size(1, 1, 1)
fn compute_sdf(input: ComputeInput) {
    let pos = vec3<f32>(input.global_id);

    let dist = sdf(pos);
    let grad = sdf_grad(pos);

    textureStore(input_tex, input.global_id, vec4(dist).xxxx);
    textureStore(grad_tex, input.global_id, grad.xyzz);
}

fn estimate_normal(v: vec3<f32>, dv: vec3<f32>) -> f32 {
    return (sdf(v + dv) + sdf(v - dv)) / 2 / length(dv);
}

fn sdf_grad(pos: vec3<f32>) -> vec3<f32> {
    let e = 0.01;

    let dx = vec3(e, 0.0, 0.0);
    let dy = dx.yxy;
    let dz = dx.yyx;

    let x = estimate_normal(pos, dx);
    let y = estimate_normal(pos, dy);
    let z = estimate_normal(pos, dz);

    return vec3(x, y, z);
}
