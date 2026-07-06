use rand::{RngExt, rngs::StdRng};

use crate::Distribution;

pub fn random_data(rng: &mut StdRng, dist: Distribution, n: usize) -> Vec<f32> {
    match dist {
        Distribution::Uniform(lo, hi) => (0..n)
            .map(|_| lo + (hi - lo) * rng.random::<f32>())
            .collect(),
        Distribution::Bernoulli(p) => (0..n)
            .map(|_| if rng.random::<f32>() < p { 1.0 } else { 0.0 })
            .collect(),
        Distribution::Normal { mean, std } => (0..n)
            .map(|_| {
                {
                    // Box–Muller, avoids pulling in rand_distr
                    let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
                    let u2 = rng.random::<f32>();
                    mean + std * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const SEED: u64 = 42;
    const OTHER_SEED: u64 = 43;

    const N: usize = 100_000;

    fn sample(seed: u64, dist: Distribution) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        random_data(&mut rng, dist, N)
    }

    fn mean(data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }

    fn std_dev(data: &[f32], mean: f32) -> f32 {
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        var.sqrt()
    }

    #[test]
    fn same_seed_is_deterministic() {
        let dist = Distribution::Normal {
            mean: 1.0,
            std: 2.0,
        };
        assert_eq!(
            sample(SEED, dist),
            sample(SEED, dist),
            "same seed must reproduce data"
        );
    }

    #[test]
    fn different_seed_differs() {
        let dist = Distribution::Uniform(0.0, 1.0);
        assert_ne!(
            sample(SEED, dist),
            sample(OTHER_SEED, dist),
            "different seeds should differ"
        );
    }

    #[test]
    fn uniform_stays_in_range_with_expected_mean() {
        let (lo, hi) = (-3.0_f32, 5.0_f32);
        let data = sample(SEED, Distribution::Uniform(lo, hi));

        assert!(
            data.iter().all(|&x| x >= lo && x <= hi),
            "uniform values must land in [{lo}, {hi}]",
        );
        // Expected mean of U[lo, hi] is the midpoint.
        let expected = (lo + hi) / 2.0;
        assert!(
            (mean(&data) - expected).abs() < 0.05,
            "uniform mean {} too far from {expected}",
            mean(&data),
        );
    }

    #[test]
    fn bernoulli_is_binary_with_expected_rate() {
        let p = 0.3_f32;
        let data = sample(SEED, Distribution::Bernoulli(p));

        assert!(
            data.iter().all(|&x| x == 0.0 || x == 1.0),
            "bernoulli values must be 0 or 1",
        );
        assert!(
            (mean(&data) - p).abs() < 0.01,
            "bernoulli rate {} too far from {p}",
            mean(&data),
        );
    }

    #[test]
    fn normal_matches_mean_and_std() {
        let (m, s) = (1.5_f32, 0.5_f32);
        let data = sample(SEED, Distribution::Normal { mean: m, std: s });

        let got_mean = mean(&data);
        let got_std = std_dev(&data, got_mean);
        assert!(
            (got_mean - m).abs() < 0.02,
            "normal mean {got_mean} too far from {m}",
        );
        assert!(
            (got_std - s).abs() < 0.02,
            "normal std {got_std} too far from {s}",
        );
    }
}
