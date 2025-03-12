use std::ops::{Index, IndexMut};

use bevy::{
    // log::{Level, LogPlugin},
    prelude::*,
};
use bevy_egui::EguiPlugin;

use arrayvec::ArrayVec;

mod camera;
mod cases;
mod editor;
mod shader;

use cases::CASES;

fn main() {
    // LogPlugin {
    //    filter: "wgpu_hal=off".to_string(),
    //    level: Level::INFO,
    //    ..Default::default()
    // }

    App::new()
        .add_plugins((DefaultPlugins, EguiPlugin, MeshPickingPlugin))
        .add_plugins(shader::DualContouringPlugin)
        .add_plugins((editor::editor_plugin, camera::camera_plugin))
        .init_resource::<DensityMap>()
        .run();
}

const N: usize = 5;

struct CaseIndex(u8);
struct Case {
    tris: ArrayVec<[u8; 3], 5>,
}

// holy kludge
fn all_cells() -> Vec<UVec3> {
    let n = N as u32;

    let mut v = Vec::with_capacity((N * N * N) as usize);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                v.push(UVec3 { x, y, z });
            }
        }
    }
    v
}

#[derive(Resource)]
struct DensityMap {
    densities: [[[f32; N + 1]; N + 1]; N + 1],
}

impl Default for DensityMap {
    fn default() -> Self {
        Self {
            densities: [[[0.0; N + 1]; N + 1]; N + 1],
        }
    }
}

impl Index<UVec3> for DensityMap {
    type Output = f32;

    fn index(&self, idx: UVec3) -> &Self::Output {
        let UVec3 { x, y, z } = idx;

        &self.densities[x as usize][y as usize][z as usize]
    }
}

impl IndexMut<UVec3> for DensityMap {
    fn index_mut(&mut self, idx: UVec3) -> &mut f32 {
        let UVec3 { x, y, z } = idx;

        &mut self.densities[x as usize][y as usize][z as usize]
    }
}

fn corners_from_cell(pos: UVec3) -> impl Iterator<Item = UVec3> {
    VERTICES
        .into_iter()
        .map(move |(x, y, z)| UVec3 { x, y, z } + pos)
}

fn sample_density_map(map: &DensityMap, pos: UVec3) -> CaseIndex {
    let mut case = 0u8;

    for (i, corner_pos) in corners_from_cell(pos).enumerate() {
        let sample = map[corner_pos];
        if sample > 0.0 {
            case |= 1 << i as usize;
        }
    }

    CaseIndex(case)
}

fn edge_to_vtx(edge: usize) -> Vec3 {
    let (v0_idx, v1_idx) = EDGES[edge];

    let v0 = UVec3::from(VERTICES[v0_idx]).as_vec3();
    let v1 = UVec3::from(VERTICES[v1_idx]).as_vec3();

    v0.midpoint(v1)
}

const VERTICES: [(u32, u32, u32); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
];

const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn edge_tri_to_triangle(pos: UVec3, tri: [u8; 3]) -> [Vec3; 3] {
    let pos_f = pos.as_vec3();

    let tri = tri
        .into_iter()
        .map(|edge| edge_to_vtx(edge as usize) + pos_f)
        .collect::<ArrayVec<Vec3, 3>>()
        .into_inner()
        .unwrap();

    tri
}

fn edge_tri_to_lines(pos: UVec3, tri: [u8; 3]) -> [(Vec3, Vec3); 3] {
    let ab = (tri[0], tri[1]);
    let bc = (tri[1], tri[2]);
    let ac = (tri[0], tri[2]);

    let pos_f = pos.as_vec3();

    let edge_pair_to_vtx_pair = |(a, b)| {
        let (a, b) = (edge_to_vtx(a as usize), edge_to_vtx(b as usize));
        (pos_f + a, pos_f + b)
    };

    [
        edge_pair_to_vtx_pair(ab),
        edge_pair_to_vtx_pair(bc),
        edge_pair_to_vtx_pair(ac),
    ]
}

// def edge_to_boundary_vertex(edge):
// """Returns the vertex in the middle of the specified edge"""
//     # Find the two vertices specified by this edge, and interpolate between
// # them according to adapt, as in the 2d case
// v0, v1 = EDGES[edge]
//     f0 = f_eval[v0]
//     f1 = f_eval[v1]
//     t0 = 1 - adapt(f0, f1)
//     t1 = 1 - t0
//     vert_pos0 = VERTICES[v0]
//     vert_pos1 = VERTICES[v1]
//     return V3(x + vert_pos0[0] * t0 + vert_pos1[0] * t1,
//               y + vert_pos0[1] * t0 + vert_pos1[1] * t1,
//               z + vert_pos0[2] * t0 + vert_pos1[2] * t1)
//
