use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions, NearestMode, TileSize};

fn default_options() -> InterpolateOptions {
    InterpolateOptions::new(InterpolateMode::Bilinear)
}

fn assert_tile_layout(
    area: usize,
    tile_target_aspect_ratio: f32,
    expected_width: usize,
    expected_height: usize,
) {
    let tile_size = TileSize::new(area, tile_target_aspect_ratio, default_options());
    assert_eq!(
        (tile_size.width(), tile_size.height()),
        (expected_width, expected_height),
        "Failed for area: {}, tile_target_aspect_ratio: {}",
        area,
        tile_target_aspect_ratio
    );
}

#[test]
fn test_perfect_squares() {
    assert_tile_layout(16, 1.0, 4, 4);
    assert_tile_layout(100, 1.0, 10, 10);
    assert_tile_layout(1, 1.0, 1, 1);
}

#[test]
fn test_exact_landscape_matches() {
    assert_tile_layout(12, 3.0, 6, 2);
    assert_tile_layout(16, 4.0, 8, 2);
}

#[test]
fn test_exact_portrait_matches() {
    assert_tile_layout(12, 1.0 / 3.0, 2, 6);
    assert_tile_layout(16, 0.25, 2, 8);
}

#[test]
fn test_imperfect_factors() {
    assert_tile_layout(10, 1.0, 5, 2);
    assert_tile_layout(12, 1.0, 4, 3);
}

#[test]
fn test_prime_areas() {
    assert_tile_layout(7, 1.0, 7, 1);
    assert_tile_layout(7, 0.1, 1, 7);
}

#[test]
fn test_extreme_targets() {
    assert_tile_layout(16, 1000.0, 16, 1);
    assert_tile_layout(16, 0.001, 1, 16);
}

#[test]
fn test_edge_cases_early_exit() {
    assert_tile_layout(16, 0.0, 16, 1);
    assert_tile_layout(16, -1.5, 16, 1);
    assert_tile_layout(0, 1.0, 0, 1);
}

#[test]
fn test_flattened_options() {
    let options = InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact));
    let tile_size = TileSize::new(16, 1.0, options);

    assert_eq!(tile_size.width(), 16);
    assert_eq!(tile_size.height(), 1);
}
