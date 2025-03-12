use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages,
    color::palettes::css::{RED, WHITE},
    prelude::*,
    render::storage::ShaderStorageBuffer,
};
use bevy_egui::{egui, EguiContexts};

use crate::{sample_density_map, Case, DensityMap, CASES};

trait HasOptionsMenu {}

#[derive(Bundle)]
struct Node {
    mesh: Mesh3d,
    mtl: MeshMaterial3d<StandardMaterial>,
    click: Clicked,
    pos: Transform,
    density_idx: DensityIdx,
    density_value: DensityValue,
}

#[derive(Resource)]
struct Materials {
    unselected: MeshMaterial3d<StandardMaterial>,
    selected: MeshMaterial3d<StandardMaterial>,
}

pub fn editor_plugin(app: &mut App) {
    app.init_resource::<VisibilitySettings>()
        .add_systems(
            Startup,
            (
                make_materials,
                spawn_mesh,
                spawn_light,
                spawn_clickable_points,
            )
                .chain(),
        )
        .add_systems(Update, (make_edit_ui, set_materials))
        .add_systems(Update, update_density_map.after(make_edit_ui));
}

fn draw_gizmos(map: Res<DensityMap>, mut gizmos: Gizmos, vis: Res<VisibilitySettings>) {
    if !vis.gizmos {
        return;
    }

    let cells = crate::all_cells();

    for cell in cells {
        let case = sample_density_map(&*map, cell);
        let edges: &Case = &CASES[case.0 as usize];

        for tri in &edges.tris {
            for (a, b) in crate::edge_tri_to_lines(cell, *tri) {
                gizmos.line(a, b, RED);
            }
        }
    }
}

#[derive(Component, Deref, DerefMut, Copy, Clone, Default, Debug)]
struct Clicked(bool);

#[derive(Component, Deref, DerefMut, Copy, Clone, Default, Debug)]
struct DensityIdx(UVec3);

#[derive(Component, Deref, DerefMut, Copy, Clone, Default, Debug)]
struct DensityValue(f32);

fn make_materials(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.insert_resource(Materials {
        unselected: MeshMaterial3d(materials.add(Color::WHITE)),
        selected: MeshMaterial3d(materials.add(Color::linear_rgba(1.0, 0.0, 0.0, 1.0))),
    });
}

#[derive(Copy, Clone, Default, Resource)]
struct VisibilitySettings {
    nodes: bool,
    gizmos: bool,
}

fn make_edit_ui(
    mut context: EguiContexts,
    mut visibilities: ResMut<VisibilitySettings>,
    mut query: Query<(&Clicked, &DensityIdx, &mut DensityValue)>,
) {
    let ctx = context.ctx_mut();

    egui::Window::new("Edit").show(ctx, |ui| {
        ui.heading("Global");
        let mut show_nodes = visibilities.nodes;
        ui.checkbox(&mut show_nodes, "Show nodes");
        if show_nodes != visibilities.nodes {
            visibilities.nodes = show_nodes;
        }

        let mut show_gizmos = visibilities.gizmos;
        ui.checkbox(&mut show_gizmos, "Show gizmos");
        if show_gizmos != visibilities.gizmos {
            visibilities.gizmos = show_gizmos;
        }

        ui.heading("Densities");
        ui.separator();
        for (clicked, idx, mut value) in &mut query {
            let mut slider = **value;
            if **clicked {
                ui.label(format!("Index: {:?}", **idx));
                ui.add(egui::Slider::new(&mut slider, -10.0..=10.0));
                ui.separator();
            }

            // jumping through this hoop to make change detection work
            if slider != **value {
                **value = slider;
            }
        }
    });
}

fn set_materials(
    mut query: Query<(&Clicked, &mut MeshMaterial3d<StandardMaterial>)>,
    materials: Res<Materials>,
) {
    for (clicked, mut mtl) in &mut query {
        if **clicked {
            *mtl = materials.selected.clone();
        } else {
            *mtl = materials.unselected.clone();
        }
    }
}

fn update_density_map(
    mut map: ResMut<DensityMap>,
    query: Query<(&DensityIdx, &DensityValue), Changed<DensityValue>>,
) {
    for (idx, val) in &query {
        map[**idx] = **val;
    }
}

#[derive(Component)]
struct MarchedMesh;

fn spawn_mesh(mut commands: Commands, mtls: Res<Materials>) {
    commands.spawn((
        // this is probably fine, I hope...
        Mesh3d(Handle::weak_from_u128(0xdeadbeef)),
        mtls.unselected.clone(),
        MarchedMesh,
    ));
}

fn rebuild_mesh(
    map: Res<DensityMap>,
    mesh_query: Query<&Mesh3d, With<MarchedMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !map.is_changed() {
        return;
    }

    for mesh in &mesh_query {
        let mesh_id = mesh.0.id();

        let mesh = meshes.get_or_insert_with(mesh_id, || {
            Mesh::new(
                bevy::render::mesh::PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
        });

        let cells = crate::all_cells();

        let mut vtx = Vec::new();

        for cell in cells {
            let case = sample_density_map(&*map, cell);
            let edges: &Case = &CASES[case.0 as usize];

            for &edge_tri in &edges.tris {
                let tri = crate::edge_tri_to_triangle(cell, edge_tri);

                vtx.extend_from_slice(&bytemuck::cast::<[Vec3; 3], [[f32; 3]; 3]>(tri));
            }
        }

        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vtx);
        mesh.compute_flat_normals();
    }
}

fn update_mesh_visibility(
    mut meshes: Query<&mut Visibility, (With<Mesh3d>, With<DensityValue>)>,
    vis: Res<VisibilitySettings>,
) {
    if !vis.is_changed() {
        return;
    }

    for mut mesh in &mut meshes {
        *mesh = if vis.nodes {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn spawn_clickable_points(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    materials: Res<Materials>,
) {
    const N: u32 = super::N as u32;

    let sphere = meshes.add(Sphere::new(0.1).mesh());

    for x in 0..N + 1 {
        for y in 0..N + 1 {
            for z in 0..N + 1 {
                commands
                    .spawn(Node {
                        mesh: Mesh3d(sphere.clone()),
                        mtl: materials.unselected.clone(),
                        click: Clicked(false),
                        pos: Transform::from_xyz(x as f32, y as f32, z as f32),
                        density_idx: DensityIdx(UVec3 { x, y, z }),
                        density_value: DensityValue(0.0),
                    })
                    .observe(update_clicked);
            }
        }
    }
}

fn spawn_light(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
    ));

    // ambient light
    commands.insert_resource(AmbientLight {
        color: WHITE.into(),
        brightness: 0.15,
    });
}

fn update_clicked(
    trigger: Trigger<Pointer<Down>>,
    mut query: Query<(&mut Clicked, &DensityIdx, &mut DensityValue)>,
) {
    if let Ok((mut clicked, idx, mut val)) = query.get_mut(trigger.entity()) {
        **clicked = !**clicked;

        **val = if **clicked { 1.0 } else { 0.0 };
        log::info!("Clicked on node with idx {:?}", idx);
    }
}
