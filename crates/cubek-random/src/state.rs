use cubecl::prelude::*;

/// The four generators a unit advances, one independent stream per lane.
///
/// Three Tausworthe generators and an LCG are combined by xor: the hybrid of GPU Gems 3,
/// chapter 37, with a combined period near 2^121. Every step is element-wise, so a
/// wide line is genuine SIMD rather than a lane loop.
/// <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application>
#[derive(CubeType)]
pub(crate) struct PrngState<N: Size> {
    taus_0: Vector<u32, N>,
    taus_1: Vector<u32, N>,
    taus_2: Vector<u32, N>,
    lcg: Vector<u32, N>,
}

/// The four seeds a launch draws from the host generator, as one argument.
#[derive(CubeLaunch, CubeType)]
pub(crate) struct Seeds {
    taus_0: u32,
    taus_1: u32,
    taus_2: u32,
    lcg: u32,
}

#[cube]
impl<N: Size> PrngState<N> {
    /// Give lane `lane` of unit `unit` the stream of the `unit * N + lane`th generator, so
    /// a wider line splits the same sequence of streams further instead of repeating one.
    pub fn seeded(unit: usize, seeds: Seeds) -> PrngState<N> {
        // A large prime spreads consecutive stream indices across the u32 range;
        // truncation is fine here, a repeated seed is no issue.
        #[allow(arithmetic_overflow)]
        let stream = Vector::new(1000000007u32)
            * (Vector::new(unit as u32 * N::value() as u32) + lane_indices::<N>());

        PrngState::<N> {
            taus_0: stream + Vector::new(seeds.taus_0),
            taus_1: stream + Vector::new(seeds.taus_1),
            taus_2: stream + Vector::new(seeds.taus_2),
            lcg: stream + Vector::new(seeds.lcg),
        }
    }

    /// Advance every generator and return their combined output.
    pub fn next(&mut self) -> Vector<u32, N> {
        // L'Ecuyer's taus88 components, periods 2^31-1, 2^29-1, and 2^28-1; each mask
        // zeroes the low state bits that sit outside its component's period.
        self.taus_0 = taus_step(self.taus_0, 13u32, 19u32, 12u32, 4294967294u32);
        self.taus_1 = taus_step(self.taus_1, 2u32, 25u32, 4u32, 4294967288u32);
        self.taus_2 = taus_step(self.taus_2, 3u32, 11u32, 17u32, 4294967280u32);
        self.lcg = lcg_step(self.lcg);

        self.taus_0 ^ self.taus_1 ^ self.taus_2 ^ self.lcg
    }
}

#[cube]
fn lane_indices<N: Size>() -> Vector<u32, N> {
    let mut indices = Vector::empty();

    #[unroll]
    for lane in 0..N::value() {
        indices.insert(lane, comptime!(lane as u32));
    }

    indices
}

#[cube]
fn taus_step<N: Size>(
    z: Vector<u32, N>,
    #[comptime] s1: u32,
    #[comptime] s2: u32,
    #[comptime] s3: u32,
    #[comptime] m: u32,
) -> Vector<u32, N> {
    let b = ((z << Vector::new(s1)) ^ z) >> Vector::new(s2);

    ((z & Vector::new(m)) << Vector::new(s3)) ^ b
}

/// One linear congruential step, with the multiplier and increment from
/// Numerical Recipes.
#[cube]
fn lcg_step<N: Size>(z: Vector<u32, N>) -> Vector<u32, N> {
    z * Vector::new(1664525u32) + Vector::new(1013904223u32)
}
