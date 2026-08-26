//! The depthwise shapes an encoder actually runs, and the tilings to run them under.

use cubek_test_utils::CatalogEntry;

use crate::{DepthwiseStrategy, DepthwiseTiling};

/// One depthwise convolution: square map, square window, one filter per channel.
///
/// Padding is not stated. Every one of these keeps its resolution, so it is `d·(k-1)/2` and
/// stating it separately would only let the catalogue drift from that.
#[derive(Clone, Copy)]
pub struct DepthwiseProblem {
    pub batch: usize,
    pub channels: usize,
    /// The side of the input map, in pixels.
    pub size: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl DepthwiseProblem {
    pub fn padding(&self) -> usize {
        self.dilation * (self.kernel - 1) / 2
    }

    /// The side of the output map.
    pub fn out_size(&self) -> usize {
        let reach = self.dilation * (self.kernel - 1) + 1;
        (self.size + 2 * self.padding() - reach) / self.stride + 1
    }

    pub fn in_shape(&self) -> [usize; 4] {
        [self.batch, self.size, self.size, self.channels]
    }

    /// Burn's depthwise weight layout: `[out_channels, kh, kw, in_channels / groups]`, and
    /// `in_channels / groups` is 1.
    pub fn weight_shape(&self) -> [usize; 4] {
        [self.channels, self.kernel, self.kernel, 1]
    }

    pub fn out_shape(&self) -> [usize; 4] {
        let out = self.out_size();
        [self.batch, out, out, self.channels]
    }

    /// Multiply-accumulates, counted as two flops each.
    pub fn flops(&self) -> f64 {
        let out = self.out_size();
        2.0 * (self.batch * out * out * self.channels * self.kernel * self.kernel) as f64
    }

    /// What the convolution must move at least once: both maps and the filter. The ratio of
    /// this to the measured time is the number to read — a depthwise pass has too little
    /// arithmetic per byte to be anything but bandwidth-bound, so its ceiling is the device's
    /// copy rate and not its flop rate.
    pub fn bytes(&self, elem_size: usize) -> f64 {
        let out = self.out_size();
        let elems = self.batch * self.size * self.size * self.channels
            + self.batch * out * out * self.channels
            + self.channels * self.kernel * self.kernel;
        (elems * elem_size) as f64
    }
}

/// Every distinct depthwise convolution EfficientNet-B4 runs at a 768px input and an output
/// stride of 16, in the order the encoder first reaches them, with how many of its blocks run
/// each. The counts are what turn a per-call median into a share of the pass: block 23's shape
/// runs seven times and block 0's once, so they are not worth the same.
///
/// Derived by hand from `blocks(Coefficients::B4, 16)` in the model this targets, and stated
/// here rather than imported because cubek does not depend on it.
const B4: [(&str, usize, usize, usize, usize, usize, usize); 14] = [
    // id, channels, size, kernel, stride, dilation, blocks running it
    ("b23_1632c_48px_k5_d2", 1632, 48, 5, 1, 2, 7),
    ("b11_672c_48px_k3", 672, 48, 3, 1, 1, 5),
    ("b17_960c_48px_k5", 960, 48, 5, 1, 1, 5),
    ("b3_192c_192px_k3", 192, 192, 3, 1, 1, 3),
    ("b7_336c_96px_k5", 336, 96, 5, 1, 1, 3),
    ("b31_2688c_48px_k3_d2", 2688, 48, 3, 1, 2, 1),
    ("b30_1632c_48px_k3_d2", 1632, 48, 3, 1, 2, 1),
    ("b22_960c_48px_k5_d2", 960, 48, 5, 1, 2, 1),
    ("b16_672c_48px_k5", 672, 48, 5, 1, 1, 1),
    ("b10_336c_96px_k3_s2", 336, 96, 3, 2, 1, 1),
    ("b6_192c_192px_k5_s2", 192, 192, 5, 2, 1, 1),
    ("b2_144c_384px_k3_s2", 144, 384, 3, 2, 1, 1),
    ("b0_48c_384px_k3", 48, 384, 3, 1, 1, 1),
    ("b1_24c_384px_k3", 24, 384, 3, 1, 1, 1),
];

/// How many images a problem runs at. A training batch by default; `CUBEK_BENCH_BATCH=1`
/// times the inference the segmenter serves.
fn batch() -> usize {
    std::env::var("CUBEK_BENCH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// How many of the encoder's blocks run `id`'s shape — the weight to give its median when
/// asking what the pass costs. Zero for an id the catalogue does not carry.
pub fn blocks_running(id: &str) -> usize {
    B4.iter()
        .find(|entry| entry.0 == id)
        .map(|entry| entry.6)
        .unwrap_or(0)
}

pub fn problems() -> Vec<CatalogEntry<DepthwiseProblem>> {
    let batch = batch();

    B4.iter()
        .map(|&(id, channels, size, kernel, stride, dilation, repeats)| {
            let problem = DepthwiseProblem {
                batch,
                channels,
                size,
                kernel,
                stride,
                dilation,
            };
            let label = format!(
                "b{batch} {channels}c {size}px k{kernel} s{stride} d{dilation} (x{repeats})"
            );
            CatalogEntry::new(id, label, problem)
        })
        .collect()
}

/// The tilings to time each problem under.
///
/// Every one of them solves every problem — the space ceil-divides, and a tail cube is simply
/// short — so the sweep is a plain cross product and no entry is skipped.
pub fn strategies() -> Vec<CatalogEntry<DepthwiseStrategy>> {
    // What ships, first: the tiling the routine picks for itself. Every entry below it is one the
    // routine could have picked and did not, which is what makes the table readable as "was the
    // rule right".
    let mut out = vec![CatalogEntry::new(
        "routine",
        "the routine's own choice",
        DepthwiseStrategy::Routine,
    )];

    // The cross product of the two spatial edges with the line width. Both matter and they trade
    // against each other — a wide line is fewer instructions per channel but more registers per
    // lane and a wider channel tile, so a narrow block runs out of lanes to give it — and neither
    // is predictable from the shape alone, which is what a sweep is for.
    for lines in [1, 2, 4] {
        for (rows, cols) in [
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 2),
            (8, 4),
            (8, 8),
            (16, 4),
        ] {
            out.push(CatalogEntry::new(
                format!("r{rows}_c{cols}_l{lines}"),
                format!("{rows} rows, {cols} cols, {lines}-wide lines"),
                DepthwiseStrategy::Fixed(DepthwiseTiling {
                    rows,
                    cols,
                    chans: 1,
                    lines,
                }),
            ));
        }
    }

    out
}
