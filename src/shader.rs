use std::{borrow::Cow, mem};

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::WHITE,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{GpuReadbackPlugin, Readback, ReadbackComplete},
        mesh::{allocator::MeshAllocator, Indices, MeshVertexAttribute},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    },
};

const SIZE_GRID: u32 = crate::N as u32 + 1;
const SIZE_CELLS: u32 = crate::N as u32;

const SIZE_CELLS_3: u32 = SIZE_CELLS * SIZE_CELLS * SIZE_CELLS;
const SIZE_GRID_3: u32 = SIZE_GRID * SIZE_GRID * SIZE_GRID;

#[derive(Resource)]
struct DualContouringPipeline {
    sdf_pipeline: CachedComputePipelineId,
    vertex_pipeline: CachedComputePipelineId,
    edge_pipeline: CachedComputePipelineId,
    adaptivity_pipeline: CachedComputePipelineId,
    cleanup_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
    adaptivity_bind_group_layout: BindGroupLayout,
    sdf_bind_group_layout: BindGroupLayout,
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
#[repr(C)]
struct MeshCounts {
    vtx: u32,
    idx: u32,
}

#[derive(ShaderType, Clone, Copy, Debug, Default)]
#[repr(C)]
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Resource, Clone, ExtractResource)]
struct DualContouringResources {
    input: Handle<Image>,
    index_lookup: Handle<Image>,
    debug_tex: Handle<Image>,
    vertex_buffer: Handle<ShaderStorageBuffer>,
    index_buffer: Handle<ShaderStorageBuffer>,
    count_buffer: Handle<ShaderStorageBuffer>,
    indirect_buffer: Handle<ShaderStorageBuffer>,
    mesh_handle: Handle<Mesh>,
    normal: Handle<Image>,
}

#[derive(Resource)]
struct DualContouringBindGroup {
    group: BindGroup,
    adaptivity_group: BindGroup,
    sdf_group: BindGroup,
}

impl FromWorld for DualContouringPipeline {
    fn from_world(world: &mut World) -> Self {
        use binding_types::storage_buffer;
        fn texture_storage_3d(
            binding: u32,
            format: TextureFormat,
            access: StorageTextureAccess,
        ) -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access,
                    format,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            }
        }

        let render_device = world.resource::<RenderDevice>();
        let bind_group_layout = render_device.create_bind_group_layout(
            "March group layout",
            &[
                // input_tex
                texture_storage_3d(0, TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                // index_lookup
                texture_storage_3d(1, TextureFormat::R32Uint, StorageTextureAccess::ReadWrite),
                // vertex_buffer1
                storage_buffer::<VertexInfo>(false).build(2, ShaderStages::COMPUTE),
                // index_buffer
                storage_buffer::<u32>(false).build(3, ShaderStages::COMPUTE),
                // counts
                storage_buffer::<MeshCounts>(false).build(4, ShaderStages::COMPUTE),
                // dispatch indirect args
                storage_buffer::<DispatchIndirectArgs>(false).build(5, ShaderStages::COMPUTE),
                // debug
                texture_storage_3d(
                    6,
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::ReadWrite,
                ),
            ],
        );

        let sdf_bind_group_layout = render_device.create_bind_group_layout(
            "SDF group layout",
            &[texture_storage_3d(
                0,
                TextureFormat::Rgba8Snorm,
                StorageTextureAccess::WriteOnly,
            )],
        );

        let adaptivity_bind_group_layout = render_device.create_bind_group_layout(
            "adaptivity group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    binding_types::texture_3d(TextureSampleType::Float { filterable: true }),
                    binding_types::sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        const CONTOUR_SHADER_PATH: &str = "shaders/contour.wgsl";
        const SDF_SHADER_PATH: &str = "shaders/sdf.wgsl";
        const ADAPTIVITY_SHADER_PATH: &str = "shaders/adaptivity.wgsl";

        let contour_shader = world.load_asset(CONTOUR_SHADER_PATH);
        let sdf_shader = world.load_asset(SDF_SHADER_PATH);
        let adaptivity_shader = world.load_asset(ADAPTIVITY_SHADER_PATH);

        let pipeline_cache = world.resource::<PipelineCache>();
        let make_pipeline =
            |shader: &Handle<_>, layouts: &[&BindGroupLayout], entrypoint: &'static str| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: None,
                    layout: layouts.iter().map(|&x| x).cloned().collect(),
                    push_constant_ranges: Vec::new(),
                    shader: shader.clone(),
                    shader_defs: vec![],
                    entry_point: Cow::from(entrypoint),
                    zero_initialize_workgroup_memory: true,
                })
            };

        let sdf_pipeline = make_pipeline(
            &sdf_shader,
            &[&bind_group_layout, &sdf_bind_group_layout],
            "compute_sdf",
        );

        let vertex_pipeline =
            make_pipeline(&contour_shader, &[&bind_group_layout], "compute_vertices");
        let edge_pipeline = make_pipeline(&contour_shader, &[&bind_group_layout], "compute_edges");
        let cleanup_pipeline = make_pipeline(&contour_shader, &[&bind_group_layout], "cleanup");

        let adaptivity_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: vec![
                    bind_group_layout.clone(),
                    adaptivity_bind_group_layout.clone(),
                ],
                push_constant_ranges: Vec::new(),
                shader: adaptivity_shader.clone(),
                shader_defs: vec![ShaderDefVal::UInt("GRID_SIZE".to_string(), SIZE_GRID)],
                entry_point: Cow::from("compute_adaptivity"),
                zero_initialize_workgroup_memory: true,
            });

        DualContouringPipeline {
            bind_group_layout,
            sdf_pipeline,
            vertex_pipeline,
            edge_pipeline,
            adaptivity_pipeline,
            cleanup_pipeline,
            adaptivity_bind_group_layout,
            sdf_bind_group_layout,
        }
    }
}

fn create_bind_group(
    mut commands: Commands,
    pipeline: Res<DualContouringPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut contouring_data: ResMut<DualContouringResources>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    render_device: Res<RenderDevice>,
) {
    let DualContouringResources {
        input,
        normal,
        index_lookup,
        debug_tex,
        vertex_buffer,
        index_buffer,
        count_buffer,
        indirect_buffer,
        mesh_handle: _,
    } = &mut *contouring_data;

    let render_device = &*render_device;

    let vertex_buffer = buffers.get(vertex_buffer).unwrap();
    let index_buffer = buffers.get(index_buffer).unwrap();
    let count_buffer = buffers.get(count_buffer).unwrap();
    let indirect_buffer = buffers.get(indirect_buffer).unwrap();

    let view_input = gpu_images.get(input).unwrap();
    let view_normal = gpu_images.get(normal).unwrap();
    let view_index = gpu_images.get(index_lookup).unwrap();
    let view_debug = gpu_images.get(debug_tex).unwrap();

    let group = render_device.create_bind_group(
        None,
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            &view_input.texture_view,
            &view_index.texture_view,
            vertex_buffer.buffer.as_entire_buffer_binding(),
            index_buffer.buffer.as_entire_buffer_binding(),
            count_buffer.buffer.as_entire_buffer_binding(),
            indirect_buffer.buffer.as_entire_buffer_binding(),
            &view_debug.texture_view,
        )),
    );

    let sdf_group = render_device.create_bind_group(
        None,
        &pipeline.sdf_bind_group_layout,
        &BindGroupEntries::sequential((&view_normal.texture_view,)),
    );
    let adaptivity_group = render_device.create_bind_group(
        None,
        &pipeline.adaptivity_bind_group_layout,
        &BindGroupEntries::sequential((&view_normal.texture_view, &view_normal.sampler)),
    );

    commands.insert_resource(DualContouringBindGroup {
        group,
        adaptivity_group,
        sdf_group,
    });
}

#[derive(Component, Default)]
struct ContouringMarker;

#[derive(Component, Default)]
struct BufferData {
    vtx: Option<Vec<VertexInfo>>,
    idx: Option<Vec<u32>>,
    counts: Option<MeshCounts>,
}

#[repr(C)]
#[derive(ShaderType, Copy, Clone, Default, Debug)]
struct VertexInfo {
    pos: [f32; 3],
    uv: [f32; 2],
    normal: [f32; 3],
}

#[derive(Bundle)]
struct ContouringMesh {
    mesh: Mesh3d,
    material: MeshMaterial3d<StandardMaterial>,
    buffer_data: BufferData,
    marker: ContouringMarker,
}

macro_rules! make_buffers {
    ($buffers:expr, $([$name:ident, $val:expr, $usage:ident]),*) => {
        $(
            let $name = {
                let mut buffer = ShaderStorageBuffer::from($val);
                buffer.buffer_description.usage |= BufferUsages::$usage;
                buffer.buffer_description.label = Some(concat!("contour ", stringify!($name)));
                $buffers.add(buffer)
            };
        )*
    };
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut input_tex = Image::new_fill(
        Extent3d {
            width: SIZE_GRID,
            height: SIZE_GRID,
            depth_or_array_layers: SIZE_GRID,
        },
        TextureDimension::D3,
        &[0, 0, 0, 0],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    input_tex.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    input_tex.texture_descriptor.label = Some("contour 3d sdf input");

    let mut normal_tex = Image::new_fill(
        Extent3d {
            width: SIZE_GRID,
            height: SIZE_GRID,
            depth_or_array_layers: SIZE_GRID,
        },
        TextureDimension::D3,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8Snorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    normal_tex.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    normal_tex.texture_descriptor.label = Some("contour 3d sdf normals");

    let mut index_tex = Image::new_fill(
        Extent3d {
            width: SIZE_CELLS,
            height: SIZE_CELLS,
            depth_or_array_layers: SIZE_CELLS,
        },
        TextureDimension::D3,
        &[0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );

    index_tex.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING;
    index_tex.texture_descriptor.label = Some("contour 3d index lookup");

    let mut debug_tex = Image::new_fill(
        Extent3d {
            width: SIZE_GRID,
            height: SIZE_GRID,
            depth_or_array_layers: SIZE_GRID,
        },
        TextureDimension::D3,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );

    debug_tex.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING;

    debug_tex.texture_descriptor.label = Some("contour 3d debug_tex");

    use crate::N;

    make_buffers!(
        buffers,
        [
            vertex_buffer,
            vec![VertexInfo::default(); N * N * N],
            COPY_SRC
        ],
        [index_buffer, vec![0u32; N * N * N * 3], COPY_SRC],
        [count_buffer, MeshCounts { vtx: 0, idx: 0 }, COPY_SRC],
        [
            indirect_buffer,
            DispatchIndirectArgs { x: 1, y: 1, z: 1 },
            INDIRECT
        ]
    );

    commands
        .spawn(Readback::buffer(vertex_buffer.clone()))
        .observe(update_vtx);
    commands
        .spawn(Readback::buffer(index_buffer.clone()))
        .observe(update_idx);
    commands
        .spawn(Readback::buffer(count_buffer.clone()))
        .observe(update_counts);

    let mesh = Cuboid::new(1.0, 1.0, 1.0).mesh();
    let mesh_handle = meshes.add(mesh);

    commands.spawn(ContouringMesh {
        mesh: Mesh3d(mesh_handle.clone()),
        material: MeshMaterial3d(materials.add(StandardMaterial::from_color(WHITE))),
        buffer_data: BufferData::default(),
        marker: ContouringMarker,
    });

    commands.insert_resource(DualContouringResources {
        input: images.add(input_tex),
        normal: images.add(normal_tex),
        index_lookup: images.add(index_tex),
        debug_tex: images.add(debug_tex),
        vertex_buffer,
        index_buffer,
        count_buffer,
        indirect_buffer,
        mesh_handle,
    });
}

fn update_vtx(
    trigger: Trigger<ReadbackComplete>,
    mut commands: Commands,
    mut query: Query<&mut BufferData>,
) {
    let data: Vec<VertexInfo> = trigger.event().to_shader_type();
    for mut buffer in &mut query {
        buffer.vtx.get_or_insert(data.clone());
        commands.trigger(AttemptMeshUpload);
    }
}

fn update_idx(
    trigger: Trigger<ReadbackComplete>,
    mut commands: Commands,
    mut query: Query<&mut BufferData>,
) {
    let data: Vec<u32> = trigger.event().to_shader_type();
    for mut buffer in &mut query {
        buffer.idx.get_or_insert(data.clone());
        commands.trigger(AttemptMeshUpload);
    }
}

fn update_counts(
    trigger: Trigger<ReadbackComplete>,
    mut commands: Commands,
    mut query: Query<&mut BufferData>,
) {
    let data: MeshCounts = trigger.event().to_shader_type();
    for mut buffer in &mut query {
        buffer.counts.get_or_insert(data);
        commands.trigger(AttemptMeshUpload);
    }
}

#[derive(Event)]
struct AttemptMeshUpload;

fn attempt_mesh_upload(
    _trigger: Trigger<AttemptMeshUpload>,
    mut query: Query<(&Mesh3d, &mut BufferData)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (mesh, mut data) in &mut query {
        let BufferData {
            vtx: vtx @ Some(_),
            idx: idx @ Some(_),
            counts: counts @ Some(_),
        } = &mut *data
        else {
            continue;
        };

        let mut vtx = vtx.take().unwrap();
        let mut idx = idx.take().unwrap();
        let counts = counts.take().unwrap();

        vtx.resize(counts.vtx as usize, VertexInfo::default());
        idx.resize(counts.idx as usize, 0);

        let mut vtx_new = Vec::new();
        let mut uv = Vec::new();

        for v in vtx {
            let [x, y, z] = v.pos;

            vtx_new.push(Vec3::new(x, y, z));
            uv.push(Vec2::new(0.0, 0.0));
        }

        let Some(mesh) = meshes.get_mut(&mesh.0) else {
            log::warn!("Mesh doesn't exist yet");
            continue;
        };
        mesh.remove_indices();
        mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
        mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION);
        mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vtx_new);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
        mesh.insert_indices(bevy::render::mesh::Indices::U32(idx));
        mesh.compute_normals();
    }
}

pub struct DualContouringPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ContouringLabel;

impl Plugin for DualContouringPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<DualContouringResources>::default())
            .add_systems(Startup, setup)
            .add_observer(attempt_mesh_upload);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<DualContouringPipeline>()
            .add_systems(
                Render,
                create_bind_group
                    .in_set(RenderSet::PrepareBindGroups)
                    .run_if(not(resource_exists::<DualContouringBindGroup>)),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();

        render_graph.add_node(ContouringLabel, ContouringNode(ContouringState::Loading));
        render_graph.add_node_edge(ContouringLabel, bevy::render::graph::CameraDriverLabel);
    }
}

enum ContouringState {
    Loading,
    Error,
    Done,
}

struct ContouringNode(ContouringState);

impl render_graph::Node for ContouringNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<DualContouringPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        use CachedPipelineState as CPS;

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.0 {
            ContouringState::Error => return,
            ContouringState::Loading => {
                let DualContouringPipeline {
                    sdf_pipeline,
                    vertex_pipeline,
                    edge_pipeline,
                    cleanup_pipeline,
                    adaptivity_pipeline,
                    adaptivity_bind_group_layout: _,
                    sdf_bind_group_layout: _,
                    bind_group_layout: _,
                } = &*pipeline;

                let mut done = true;

                for pipeline in [
                    sdf_pipeline,
                    vertex_pipeline,
                    edge_pipeline,
                    cleanup_pipeline,
                    adaptivity_pipeline,
                ] {
                    match pipeline_cache.get_compute_pipeline_state(*pipeline) {
                        CPS::Ok(_) => {}
                        CPS::Err(PipelineCacheError::ProcessShaderError(err)) => {
                            self.0 = ContouringState::Error;
                            done = false;
                            log::error!("Error while initializing shader: {err}")
                        }
                        _ => {
                            done = false;
                        }
                    }
                }

                if done {
                    self.0 = ContouringState::Done;
                }
            }
            ContouringState::Done => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        match self.0 {
            ContouringState::Loading | ContouringState::Error => return Ok(()),
            _ => {}
        }

        let bind_group = &world.resource::<DualContouringBindGroup>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<DualContouringPipeline>();
        let resources = world.resource::<DualContouringResources>();
        let buffers = world.resource::<RenderAssets<GpuShaderStorageBuffer>>();

        let encoder = render_context.command_encoder();

        let run_pass = |encoder: &mut CommandEncoder,
                        pipeline: CachedComputePipelineId,
                        (x, y, z): (u32, u32, u32)| {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            let pipeline = pipeline_cache.get_compute_pipeline(pipeline).unwrap();
            pass.set_bind_group(0, &bind_group.group, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(x, y, z);
        };

        let DualContouringPipeline {
            sdf_pipeline,
            vertex_pipeline,
            edge_pipeline,
            cleanup_pipeline,
            bind_group_layout: _,
            adaptivity_pipeline,
            adaptivity_bind_group_layout: _,
            sdf_bind_group_layout: _,
        } = pipeline;

        let per_grid_vertices = (SIZE_GRID, SIZE_GRID, SIZE_GRID);
        let per_grid_cells = (SIZE_CELLS, SIZE_CELLS, SIZE_CELLS);
        let once = (1, 1, 1);

        encoder.push_debug_group("render mesh");

        run_pass(encoder, *cleanup_pipeline, once);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

            let pipeline = pipeline_cache.get_compute_pipeline(*sdf_pipeline).unwrap();
            pass.set_bind_group(0, &bind_group.group, &[]);
            pass.set_bind_group(1, &bind_group.sdf_group, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(SIZE_GRID, SIZE_GRID, SIZE_GRID);
        }
        run_pass(encoder, *vertex_pipeline, per_grid_cells);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            let adaptivity_pipeline = pipeline_cache
                .get_compute_pipeline(*adaptivity_pipeline)
                .unwrap();
            pass.set_bind_group(0, &bind_group.group, &[]);
            pass.set_bind_group(1, &bind_group.adaptivity_group, &[]);
            pass.set_pipeline(adaptivity_pipeline);
            pass.dispatch_workgroups_indirect(
                &buffers.get(&resources.indirect_buffer).unwrap().buffer,
                0,
            );
        }
        run_pass(encoder, *edge_pipeline, per_grid_cells);

        encoder.pop_debug_group();

        Ok(())
    }
}
