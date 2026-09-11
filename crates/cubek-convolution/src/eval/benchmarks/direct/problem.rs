use cubek_test_utils::CatalogEntry;

#[derive(Clone, Copy)]
pub struct DirectProblem {
    pub batch: usize,
    pub channels_in: usize,
    pub channels_out: usize,
    pub size: usize,
    pub kernel: usize,
    pub stride: usize,
}

impl DirectProblem {
    pub fn padding(&self) -> usize {
        self.kernel / 2
    }

    pub fn out_size(&self) -> usize {
        (self.size + 2 * self.padding() - self.kernel) / self.stride + 1
    }

    pub fn in_shape(&self) -> [usize; 4] {
        [self.batch, self.size, self.size, self.channels_in]
    }

    pub fn weight_shape(&self) -> [usize; 4] {
        [
            self.channels_out,
            self.kernel,
            self.kernel,
            self.channels_in,
        ]
    }

    pub fn out_shape(&self) -> [usize; 4] {
        let out = self.out_size();
        [self.batch, out, out, self.channels_out]
    }
}

const LAYERS: [(&str, usize, usize, usize, usize, usize); 14] = [
    ("stem_3_64_k7s2", 3, 64, 224, 7, 2),
    ("r1_64_64_k1", 64, 64, 56, 1, 1),
    ("r1_64_64_k3", 64, 64, 56, 3, 1),
    ("r1_64_256_k1", 64, 256, 56, 1, 1),
    ("r1_256_64_k1", 256, 64, 56, 1, 1),
    ("r2_128_128_k3", 128, 128, 28, 3, 1),
    ("r2_128_128_k3s2", 128, 128, 56, 3, 2),
    ("r3_256_256_k3", 256, 256, 14, 3, 1),
    ("r4_512_512_k3", 512, 512, 7, 3, 1),
    ("r4_512_2048_k1", 512, 2048, 7, 1, 1),
    ("c4_32_k3", 4, 32, 56, 3, 1),
    ("c8_32_k3", 8, 32, 56, 3, 1),
    ("c16_32_k3", 16, 32, 56, 3, 1),
    ("c32_32_k3", 32, 32, 56, 3, 1),
];

fn batch() -> usize {
    std::env::var("CUBEK_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

pub fn problems() -> Vec<CatalogEntry<DirectProblem>> {
    let batch = batch();

    LAYERS
        .iter()
        .map(|&(id, channels_in, channels_out, size, kernel, stride)| {
            let problem = DirectProblem {
                batch,
                channels_in,
                channels_out,
                size,
                kernel,
                stride,
            };
            let label =
                format!("b{batch} {channels_in}->{channels_out}c {size}px k{kernel} s{stride}");
            CatalogEntry::new(id, label, problem)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DirectStrategy {
    Routine,
}

pub fn strategies() -> Vec<CatalogEntry<DirectStrategy>> {
    vec![CatalogEntry::new(
        "routine",
        "the routine's own choice",
        DirectStrategy::Routine,
    )]
}
