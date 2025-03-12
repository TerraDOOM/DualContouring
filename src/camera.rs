// from the bevy cookbook

use bevy::{
    color::palettes::css::*,
    input::{
        common_conditions::input_pressed,
        mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    },
    prelude::*,
};

use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub fn camera_plugin(app: &mut App) {
    app.add_systems(Startup, spawn_camera)
        .add_systems(
            Update,
            pan_orbit_camera.run_if(any_with_component::<PanOrbitState>),
        )
        .add_systems(
            Update,
            show_coordinate_system.run_if(input_pressed(KeyCode::Space)),
        );
}

fn show_coordinate_system(mut gizmos: Gizmos) {
    let orig = Vec3::ZERO;
    let x = orig + Vec3::new(1.0, 0.0, 0.0);
    let y = orig + Vec3::new(0.0, 1.0, 0.0);
    let z = orig + Vec3::new(0.0, 0.0, 1.0);

    gizmos.arrow(orig, x, RED);
    gizmos.arrow(orig, y, BLUE);
    gizmos.arrow(orig, z, GREEN);
}

// Bundle to spawn our custom camera easily
#[derive(Bundle, Default)]
pub struct PanOrbitCameraBundle {
    pub camera: Camera3d,
    pub state: PanOrbitState,
    pub settings: PanOrbitSettings,
}

// The internal state of the pan-orbit controller
#[derive(Component)]
pub struct PanOrbitState {
    pub center: Vec3,
    pub radius: f32,
    pub upside_down: bool,
    pub pitch: f32,
    pub yaw: f32,
}

/// The configuration of the pan-orbit controller
#[derive(Component)]
pub struct PanOrbitSettings {
    /// World units per pixel of mouse motion
    pub pan_sensitivity: f32,
    /// Radians per pixel of mouse motion
    pub orbit_sensitivity: f32,
    /// Exponent per pixel of mouse motion
    pub zoom_sensitivity: f32,
    /// Key to hold for panning
    pub pan_key: InputBinding,
    /// Key to hold for orbiting
    pub orbit_key: InputBinding,
    /// Key to hold for zooming
    pub zoom_key: InputBinding,
    /// What action is bound to the scroll wheel?
    pub scroll_action: Option<PanOrbitAction>,
    /// For devices with a notched scroll wheel, like desktop mice
    pub scroll_line_sensitivity: f32,
    /// For devices with smooth scrolling, like touchpads
    pub scroll_pixel_sensitivity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanOrbitAction {
    Pan,
    Orbit,
    Zoom,
}

impl Default for PanOrbitState {
    fn default() -> Self {
        PanOrbitState {
            center: Vec3::ZERO,
            radius: 1.0,
            upside_down: false,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

impl Default for PanOrbitSettings {
    fn default() -> Self {
        PanOrbitSettings {
            pan_sensitivity: 0.001,                 // 1000 pixels per world unit
            orbit_sensitivity: 0.1f32.to_radians(), // 0.1 degree per pixel
            zoom_sensitivity: 0.01,
            pan_key: InputBinding::Mouse(MouseButton::Right),
            orbit_key: InputBinding::Mouse(MouseButton::Left),
            zoom_key: InputBinding::Key(KeyCode::ShiftLeft),
            scroll_action: Some(PanOrbitAction::Zoom),
            scroll_line_sensitivity: 16.0, // 1 "line" == 16 "pixels of motion"
            scroll_pixel_sensitivity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InputBinding {
    Key(KeyCode),
    Mouse(MouseButton),
    #[default]
    Unbound,
}

impl InputBinding {
    pub fn pressed(&self, kbd: &ButtonInput<KeyCode>, mouse: &ButtonInput<MouseButton>) -> bool {
        match self {
            &Self::Key(x) => kbd.pressed(x),
            &Self::Mouse(x) => mouse.pressed(x),
            Self::Unbound => false,
        }
    }

    pub fn just_pressed(
        &self,
        kbd: &ButtonInput<KeyCode>,
        mouse: &ButtonInput<MouseButton>,
    ) -> bool {
        match self {
            &Self::Key(x) => kbd.just_pressed(x),
            &Self::Mouse(x) => mouse.just_pressed(x),
            Self::Unbound => false,
        }
    }
}

fn spawn_camera(mut commands: Commands) {
    let mut camera = PanOrbitCameraBundle::default();
    // Position our camera using our component,
    // not Transform (it would get overwritten)
    camera.state.center = Vec3::new(0.5, 0.5, 0.5);
    camera.state.radius = 10.0;
    camera.state.pitch = 15.0f32.to_radians();
    camera.state.yaw = 30.0f32.to_radians();
    commands.spawn(camera);
}

fn pan_orbit_camera(
    kbd: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut evr_motion: EventReader<MouseMotion>,
    mut evr_scroll: EventReader<MouseWheel>,
    mut q_camera: Query<(&PanOrbitSettings, &mut PanOrbitState, &mut Transform)>,
) {
    // First, accumulate the total amount of
    // mouse motion and scroll, from all pending events:
    let mut total_motion: Vec2 = evr_motion.read().map(|ev| ev.delta).sum();

    // Reverse Y (Bevy's Worldspace coordinate system is Y-Up,
    // but events are in window/ui coordinates, which are Y-Down)
    total_motion.y = -total_motion.y;

    let mut total_scroll_lines = Vec2::ZERO;
    let mut total_scroll_pixels = Vec2::ZERO;
    for ev in evr_scroll.read() {
        match ev.unit {
            MouseScrollUnit::Line => {
                total_scroll_lines.x += ev.x;
                total_scroll_lines.y -= ev.y;
            }
            MouseScrollUnit::Pixel => {
                total_scroll_pixels.x += ev.x;
                total_scroll_pixels.y -= ev.y;
            }
        }
    }

    for (settings, mut state, mut transform) in &mut q_camera {
        // Check how much of each thing we need to apply.
        // Accumulate values from motion and scroll,
        // based on our configuration settings.

        let kbd = &*kbd;
        let mouse = &*mouse;

        let mut total_pan = Vec2::ZERO;
        if settings.pan_key.pressed(kbd, mouse) {
            total_pan -= total_motion * settings.pan_sensitivity;
        }
        if settings.scroll_action == Some(PanOrbitAction::Pan) {
            total_pan -=
                total_scroll_lines * settings.scroll_line_sensitivity * settings.pan_sensitivity;
            total_pan -=
                total_scroll_pixels * settings.scroll_pixel_sensitivity * settings.pan_sensitivity;
        }

        let mut total_orbit = Vec2::ZERO;
        if settings.orbit_key.pressed(kbd, mouse) {
            total_orbit -= total_motion * settings.orbit_sensitivity;
        }
        if settings.scroll_action == Some(PanOrbitAction::Orbit) {
            total_orbit -=
                total_scroll_lines * settings.scroll_line_sensitivity * settings.orbit_sensitivity;
            total_orbit -= total_scroll_pixels
                * settings.scroll_pixel_sensitivity
                * settings.orbit_sensitivity;
        }

        let mut total_zoom = Vec2::ZERO;
        if settings.zoom_key.pressed(kbd, mouse) {
            total_zoom -= total_motion * settings.zoom_sensitivity;
        }
        if settings.scroll_action == Some(PanOrbitAction::Zoom) {
            total_zoom -=
                total_scroll_lines * settings.scroll_line_sensitivity * settings.zoom_sensitivity;
            total_zoom -=
                total_scroll_pixels * settings.scroll_pixel_sensitivity * settings.zoom_sensitivity;
        }

        // Upon starting a new orbit maneuver (key is just pressed),
        // check if we are starting it upside-down
        if settings.orbit_key.just_pressed(kbd, mouse) {
            state.upside_down = state.pitch < -FRAC_PI_2 || state.pitch > FRAC_PI_2;
        }

        // If we are upside down, reverse the X orbiting
        if state.upside_down {
            total_orbit.x = -total_orbit.x;
        }

        // Now we can actually do the things!

        let mut any = false;

        // To ZOOM, we need to multiply our radius.
        if total_zoom != Vec2::ZERO {
            any = true;
            // in order for zoom to feel intuitive,
            // everything needs to be exponential
            // (done via multiplication)
            // not linear
            // (done via addition)

            // so we compute the exponential of our
            // accumulated value and multiply by that
            state.radius *= (-total_zoom.y).exp();
        }

        // To ORBIT, we change our pitch and yaw values
        if total_orbit != Vec2::ZERO {
            any = true;
            state.yaw += total_orbit.x;
            state.pitch += total_orbit.y;
            // wrap around, to stay between +- 180 degrees
            if state.yaw > PI {
                state.yaw -= TAU; // 2 * PI
            }
            if state.yaw < -PI {
                state.yaw += TAU; // 2 * PI
            }
            if state.pitch > PI {
                state.pitch -= TAU; // 2 * PI
            }
            if state.pitch < -PI {
                state.pitch += TAU; // 2 * PI
            }
        }

        // To PAN, we can get the UP and RIGHT direction
        // vectors from the camera's transform, and use
        // them to move the center point. Multiply by the
        // radius to make the pan adapt to the current zoom.
        if total_pan != Vec2::ZERO {
            any = true;
            let radius = state.radius;
            state.center += transform.right() * total_pan.x * radius;
            state.center += transform.up() * total_pan.y * radius;
        }

        // Finally, compute the new camera transform.
        // (if we changed anything, or if the pan-orbit
        // controller was just added and thus we are running
        // for the first time and need to initialize)
        if any || state.is_added() {
            // YXZ Euler Rotation performs yaw/pitch/roll.
            transform.rotation = Quat::from_euler(EulerRot::YXZ, -state.yaw, state.pitch, 0.0);
            // To position the camera, get the backward direction vector
            // and place the camera at the desired radius from the center.
            transform.translation = state.center + transform.back() * state.radius;
        }
    }
}
