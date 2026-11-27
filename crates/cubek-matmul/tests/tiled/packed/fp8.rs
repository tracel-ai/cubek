//! 8-bit floats under one scale a row against MXFP4 under one scale a column, as an MXFP8 activation
//! meets an MXFP4 weight: both outer scales factored out of the sum.

use cubecl::client::Client;
use cubecl::quant::scheme::{QuantScheme, QuantStore};
use cubecl::zspace::Tiling;
use cubecl::zspace::metadata::Metadata;
use cubecl_common::e4m3;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{CubeStagedMatmulRoutine, PlaneShuffleMatmulRoutine};
use cubek_std::matmul_dimensions::{K, M};
use cubek_test_utils::{ScopeChecker, TestOutcome, ValidationResult};

use crate::matmul::Declines;
use crate::tiled::packed::{PackedFormat, PackedProduct, Stored};
use crate::tiled::{ByRoutine, upload};

/// How [`Fp8Rows`] lie.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Fp8Layout {
    /// Row after row, a byte a value.
    Rows,
    /// Tiles of 16 rows by 8 of `K`, each row by row: `[m / 16][k / 8][m % 16][k % 8]`, as RDNA4's
    /// matrix instruction takes its first operand.
    Tiles,
}

/// `rows` rows of `k` values of `e4m3`, each row under a scale of its own.
pub(crate) struct Fp8Rows {
    rows: usize,
    k: usize,
}

impl Fp8Rows {
    pub(crate) fn of([rows, k]: [usize; 2]) -> Self {
        Self { rows, k }
    }

    /// The matrix as a kernel reads it, its values laid as `layout` says.
    pub(crate) fn stored(&self, client: &Client, layout: Fp8Layout) -> StoredTensor {
        let values = match layout {
            Fp8Layout::Rows => upload(client, &[self.rows, self.k], &self.values()),
            Fp8Layout::Tiles => self.tiles(client),
        };
        let scales: Vec<f32> = (0..self.rows).map(Self::scale).collect();
        StoredTensor::of_quantized(
            client,
            values.binding,
            QuantScheme::fp8_per_row().with_store(QuantStore::Native),
            upload(client, &[self.rows, 1], &scales),
            None,
            [M, K],
        )
        .expect("one scale a row over 8-bit floats")
    }

    /// Its values, dequantized, row after row.
    pub(crate) fn dequantized(&self) -> Vec<f32> {
        self.values()
            .chunks(self.k)
            .enumerate()
            .flat_map(|(row, values)| values.iter().map(move |v| v.to_f32() * Self::scale(row)))
            .collect()
    }

    fn values(&self) -> Vec<e4m3> {
        (0..self.rows * self.k)
            .map(|at| {
                let bits = (at * 7919 % 2003) as u8;
                // One to four binades under one, either sign.
                e4m3::from_bits((bits & 0x87) | ((3 + (bits >> 3 & 0b11)) << 3))
            })
            .collect()
    }

    fn scale(row: usize) -> f32 {
        [0.5, 1.25, 1., 0.75][row % 4]
    }

    fn tiles(&self, client: &Client) -> PlainTensor {
        let (rows, k) = (self.rows, self.k);
        let values = self.values();
        let mut tiles = vec![e4m3::from_bits(0); rows * k];
        for (at, value) in values.into_iter().enumerate() {
            let (m, kk) = (at / k, at % k);
            let tile = m / 16 * (k / 8) + kk / 8;
            tiles[tile * 128 + m % 16 * 8 + kk % 8] = value;
        }
        let shape = [rows / 16, k / 8, 16, 8];
        let strides = [k * 16, 128, 8, 1];
        let meta = Metadata::new(shape, strides)
            .with_tiling(Tiling::new(&[2, 2]).expect("two pieces a dim"))
            .expect("the tiling lists the dims");
        upload(client, &shape, &tiles).viewed(meta)
    }
}

/// `m` rows of 8-bit floats laid as `layout` says, against MXFP4 columns under a scale each.
fn projection(layout: Fp8Layout) -> PackedProduct {
    PackedProduct::of(
        Stored::Fp8Rows { layout },
        Stored::Packed {
            format: PackedFormat::Mxfp4RowFactored,
            seed: 5,
        },
    )
}

/// Every packed setting of the staged cube answers, those that read the activation where it lies
/// among them, at a prompt's rows and at rows and columns no box divides.
#[test]
fn rows_of_fp8_answer_at_every_staged_setting() {
    let rows = projection(Fp8Layout::Rows);
    check_staged(&[rows.clone().m(128).n(128).k(1024), rows.m(33).n(96).k(512)]);
}

#[test]
fn tiles_of_fp8_answer_at_every_staged_setting() {
    let tiles = projection(Fp8Layout::Tiles);
    check_staged(&[
        tiles.clone().m(128).n(128).k(1024),
        tiles.m(48).n(96).k(512),
    ]);
}

#[test]
fn rows_of_fp8_answer_a_decode_step() {
    let products = [projection(Fp8Layout::Rows).m(4).n(96).k(512)];
    let plane = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    plane.check(plane.at(PlaneShuffleMatmulRoutine::packed_settings(), &products));
}

/// `products` at every packed setting of the staged cube, one of which at least answers on a device
/// that answers any.
fn check_staged(products: &[PackedProduct]) {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let client = cubecl::test_device().client();
    let cases = staged.at(
        CubeStagedMatmulRoutine.packed_settings(staged.hw()),
        products,
    );
    match cases.first().map(|first| staged.outcome(&client, first)) {
        Some(TestOutcome::Validated(ValidationResult::Skipped(_))) | None => staged.check(cases),
        Some(_) => staged.check_some(cases),
    }
}
