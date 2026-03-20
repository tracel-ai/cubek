use cubecl::{CubeCount, Runtime, client::ComputeClient};

pub fn cube_count_spread_with_total<R: Runtime>(
    client: &ComputeClient<R>,
    num_cubes: usize,
) -> (CubeCount, usize) {
    let cube_count = cube_count_spread(&client.properties().hardware.max_cube_count, num_cubes);

    (
        CubeCount::Static(
            cube_count[0] as u32,
            cube_count[1] as u32,
            cube_count[2] as u32,
        ),
        cube_count[0] * cube_count[1] * cube_count[2],
    )
}

fn cube_count_spread(max_cube_count: &(u32, u32, u32), num_cubes: usize) -> [usize; 3] {
    let max_cube_count = [max_cube_count.0, max_cube_count.1, max_cube_count.2];
    let mut num_cubes = [num_cubes, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_cubes[i] <= max_cube_count[i] as usize {
            return true;
        }

        loop {
            num_cubes[i] = num_cubes[i].div_ceil(base);
            num_cubes[i + 1] *= base;

            if num_cubes[i] <= max_cube_count[i] as usize {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_cubes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_num_cubes_even() {
        let max = (32, 32, 32);
        let required = 2048;

        let actual = cube_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test]
    fn safe_num_cubes_odd() {
        let max = (48, 32, 16);
        let required = 3177;

        let actual = cube_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }
}
