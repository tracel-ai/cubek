//! A dense product with a packed operand, as a model quantized to block-scaled formats runs its
//! projections: a packed weight, a packed activation against a float weight, or both packed. Each
//! routine that reads a packed operand answers the dequantized product.

use std::sync::{Arc, OnceLock};

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::Scalar;
use cubecl::prelude::{CubePrimitive, TensorBinding};
use cubecl::quant::scheme::{QuantScheme, QuantStore};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Field;
use cubecl::tile::launch::Decline;
use cubecl::zspace::Shape;
use cubecl::zspace::Strides;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::MatmulRoutine;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_matmul::tiled::storage::{StoredValues, stored_tile, tile, tile_order_on};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperand};
use cubek_std::Hardware;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order::{self, TileWords};
use cubek_std::matmul_dimensions::{K, N};
use cubek_test_utils::{HostData, HostDataVec};
use half::bf16;

use crate::tiled::packed::{HostSlabs, PackedFormat, PackedSlabs, PackedSlabsSpec};
use crate::tiled::{RoutineProduct, upload};

/// Loose enough for a `bf16` output.
const BF16_TOLERANCE: f32 = 1e-2;

/// How an operand of a [`PackedProduct`] is stored.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Stored {
    /// Dense `bf16` values in `[-1, 1)`, row-major.
    Bf16,
    /// Packed along `k` in `format`, seeded by `seed`.
    Packed { format: PackedFormat, seed: usize },
    /// A rhs `[n, k]` packed along `k` in `format`, seen through the transposed view `[k, n]` a
    /// framework hands a tied embedding to the vocabulary head: its words run along the view's
    /// first dimension.
    PackedViewedTransposed { format: PackedFormat, seed: usize },
    /// A rhs `[k, n]` packed down `k` in `format`, stored in the tile order a load stores a
    /// model's weight in on the device ([`tile_order_on`]): its words relaid through the tile DSL,
    /// its scales in the rows the order keeps them in. As it lies in rows where the device stores
    /// no tile for it.
    PackedInTileOrder { format: PackedFormat, seed: usize },
}

/// `m × n`, `k` deep, of two operands stored as stated, written in `bf16`.
#[derive(Clone)]
pub(crate) struct PackedProduct {
    m: usize,
    n: usize,
    k: usize,
    lhs: Stored,
    rhs: Stored,
    /// The product on the host, computed once for every setting it is launched at: shared by the
    /// clones of one product, and reset by a builder that changes it.
    reference: Arc<OnceLock<HostData>>,
}

impl core::fmt::Debug for PackedProduct {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}×{}×{}, lhs {:?}, rhs {:?}",
            self.m, self.n, self.k, self.lhs, self.rhs
        )
    }
}

/// An operand's values on the host, read at `(row of its own, k)`.
enum HostOperand {
    Dense {
        values: Vec<f32>,
        row_stride: usize,
        k_stride: usize,
    },
    Packed(HostSlabs),
}

impl HostOperand {
    fn at(&self, row: usize, k: usize) -> f32 {
        match self {
            HostOperand::Dense {
                values,
                row_stride,
                k_stride,
            } => values[row * row_stride + k * k_stride],
            HostOperand::Packed(slabs) => slabs.cell(0, k, row),
        }
    }
}

impl PackedProduct {
    /// A `1 × 1 × 1` product of `lhs` and `rhs`: its extents are stated by [`m`](Self::m),
    /// [`n`](Self::n) and [`k`](Self::k).
    pub(crate) fn of(lhs: Stored, rhs: Stored) -> Self {
        Self {
            m: 1,
            n: 1,
            k: 1,
            lhs,
            rhs,
            reference: Arc::default(),
        }
    }

    /// The same product with `m` rows.
    pub(crate) fn m(self, m: usize) -> Self {
        Self {
            m,
            reference: Arc::default(),
            ..self
        }
    }

    /// The same product with `n` columns.
    pub(crate) fn n(self, n: usize) -> Self {
        Self {
            n,
            reference: Arc::default(),
            ..self
        }
    }

    /// The same product, `k` deep.
    pub(crate) fn k(self, k: usize) -> Self {
        Self {
            k,
            reference: Arc::default(),
            ..self
        }
    }

    /// `count` `bf16` values in `[-1, 1)`.
    fn halves(count: usize) -> Vec<bf16> {
        (0..count)
            .map(|i| bf16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
            .collect()
    }

    /// The packed slabs of `operand` in `format`, `rows` rows of `k`, seeded by `seed`.
    fn spec(&self, operand: MatmulOperand, format: PackedFormat, seed: usize) -> PackedSlabsSpec {
        PackedSlabsSpec {
            format,
            operand,
            slabs: 1,
            rows: match operand {
                MatmulOperand::Lhs => self.m,
                _ => self.n,
            },
            k: self.k,
            seed,
        }
    }

    /// The operand `operand`, stored as `stored`: `[m, k]` for the lhs, `[k, n]` for the rhs.
    fn stored(&self, client: &Client, operand: MatmulOperand, stored: Stored) -> StoredTensor {
        let dims = match operand {
            MatmulOperand::Lhs => [self.m, self.k],
            _ => [self.k, self.n],
        };
        match stored {
            Stored::Bf16 => {
                StoredTensor::Plain(upload(client, &dims, &Self::halves(dims[0] * dims[1])))
            }
            Stored::Packed { format, seed } => StoredTensor::Packed(
                PackedSlabs::new(client, self.spec(operand, format, seed)).matrix(),
            ),
            Stored::PackedViewedTransposed { format, seed } => {
                let matrix = PackedSlabs::new(client, self.spec(operand, format, seed)).matrix();
                StoredTensor::Packed(Self::viewed_transposed(client, matrix, self.k))
            }
            Stored::PackedInTileOrder { format, seed } => {
                let matrix = PackedSlabs::new(client, self.spec(operand, format, seed)).matrix();
                StoredTensor::Packed(self.in_tile_order(client, matrix))
            }
        }
    }

    /// `matrix`, `[n, k]` packed along `k`, as a framework's transpose restates it: bound as
    /// `[k, n]` under swapped strides, its blocks and words along its first dimension.
    fn viewed_transposed(client: &Client, matrix: PackedTensor, k: usize) -> PackedTensor {
        let mut scheme = matrix.scheme();
        scheme.swap_block_dims(2, 0, 1);
        scheme.store = QuantStore::PackedU32(1);
        let mut values = matrix.words().transposed().binding;
        values.shape[0] = k;
        let mut scales = matrix.scales().cloned();
        let block_scales = scales.next().expect("block scales").transposed();
        PackedTensor::of_quantized(client, values, scheme, block_scales, scales.next(), [K, N])
            .expect("a transposed view of a packed matrix reads as stored")
    }

    /// `matrix`, `[n, k]` packed down `k`, in the tile order a load stores a weight in on the
    /// device against a `bf16` activation, as a load lays it out: the device's tile
    /// ([`stored_tile`]) and its order of it ([`tile_order_on`]), read off the scheme a `[k, n]`
    /// weight states; the words relaid as one of the device's loads reads them; the block scales
    /// relaid in tiles of their own where the order keeps them so, and otherwise left in their
    /// rows; the whole read back as a framework hands a quantized weight over
    /// ([`PackedTensor::of_quantized`]). As it lies where the device has no such tile, or lays its
    /// tiles block by block.
    fn in_tile_order(&self, client: &Client, matrix: PackedTensor) -> PackedTensor {
        let hw = Hardware::from_client(client);
        let weight = Self::stated_over_k_n(&matrix.scheme());
        let bf16 = ElemType::Float(FloatKind::BF16);
        let order = stored_tile(&hw, MatmulOperand::Rhs, StoredValues::Packed(weight), bf16)
            .and_then(|tile| {
                tile_order_on(
                    &hw,
                    &weight,
                    (tile.extent(K), tile.extent(N)),
                    (self.k, self.n),
                )
            })
            .filter(|order| order.words() == TileWords::RowByRow);
        let Some(order) = order else {
            return matrix;
        };
        let word = u32::elem_type_native();
        let words_per_load = hw.load_values(word);
        let (block, blocks) = (order.block(), self.k / order.block());
        let relaid = |binding: TensorBinding, dimensions: &[_], field, storage| {
            tile(client, binding, dimensions, word, Some(field), storage)
                .expect("the device's tile order closes the matrix in whole tiles")
        };
        // The words as the kernel names the values, `[n, k/block, block]`, counted in values.
        let values = relaid(
            TensorBinding {
                shape: Shape::from(vec![self.n, blocks, block]),
                strides: Strides::from(vec![self.k, block, 1]),
                ..matrix.words().binding.clone()
            },
            &tile_order::values_dimensions(N),
            Field::from(weight.value),
            order.values_storage(N, words_per_load),
        );
        let values = PlainTensor::of_handle(values.clone())
            .bound_as(tile_order::matrix_values(&values.metadata));
        // The block scales, one byte each in `[n, k/block]` rows, as a `[k/block, n]` matrix.
        let mut scales = matrix.scales().cloned();
        let rows = scales.next().expect("block scales");
        let rows_a_load = order.rows_a_load(N, words_per_load);
        let block_scales = match order.scales_storage(N, rows_a_load) {
            Some(storage) => {
                let tiled = relaid(
                    TensorBinding {
                        shape: Shape::from(vec![self.n, blocks]),
                        strides: Strides::from(vec![blocks, 1]),
                        ..rows.binding.clone()
                    },
                    &tile_order::scales_dimensions(N),
                    Field::of_scale(weight.scale_dtype()),
                    storage,
                );
                PlainTensor::of_handle(tiled.clone())
                    .viewed(tile_order::matrix_scales(&tiled.metadata))
            }
            None => rows.viewed(Metadata::new(
                Shape::from(vec![blocks, self.n]),
                Strides::from(vec![1, blocks]),
            )),
        };
        PackedTensor::of_quantized(client, values, weight, block_scales, scales.next(), [K, N])
            .expect("a weight laid as a load lays it reads as a framework hands it over")
    }

    /// `scheme`, which the slabs state over their own `[n, k]` rows packed along `k`, as a `[k, n]`
    /// weight states it: packed along its first dimension, each block `[block, 1]`.
    fn stated_over_k_n(scheme: &QuantScheme) -> QuantScheme {
        let block = Operand::of_scheme(scheme, K, [N, K])
            .and_then(|operand| operand.scale(K))
            .expect("the slabs scale blocks along k");
        let block = u8::try_from(block).expect("a block of under 256 values");
        scheme
            .with_store(QuantStore::PackedU32(1))
            .per_block([block, 1], scheme.scale_dtype())
    }

    /// The operand's values on the host.
    fn host(&self, operand: MatmulOperand, stored: Stored) -> HostOperand {
        match (stored, operand) {
            (Stored::Bf16, MatmulOperand::Lhs) => HostOperand::Dense {
                values: Self::floats(self.m * self.k),
                row_stride: self.k,
                k_stride: 1,
            },
            (Stored::Bf16, _) => HostOperand::Dense {
                values: Self::floats(self.k * self.n),
                row_stride: 1,
                k_stride: self.n,
            },
            (
                Stored::Packed { format, seed }
                | Stored::PackedViewedTransposed { format, seed }
                | Stored::PackedInTileOrder { format, seed },
                _,
            ) => HostOperand::Packed(self.spec(operand, format, seed).dequantized()),
        }
    }

    /// [`halves`](Self::halves) read back.
    fn floats(count: usize) -> Vec<f32> {
        Self::halves(count)
            .iter()
            .map(|half| half.to_f32())
            .collect()
    }
}

impl RoutineProduct for PackedProduct {
    fn launch<R, M, SM>(
        &self,
        routine: &R,
        hw: &Hardware,
        client: &Client,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Decline>
    where
        R: MatmulRoutine<M, SM>,
        M: MatmulMicrokernel,
    {
        let (out, bound) = Self::poisoned(
            client,
            Shape::from(vec![self.m, self.n]),
            bf16::elem_type_native(),
        );
        routine.launch_into(
            hw,
            client,
            self.stored(client, MatmulOperand::Lhs, self.lhs),
            self.stored(client, MatmulOperand::Rhs, self.rhs),
            bound,
            settings,
        )?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        self.reference
            .get_or_init(|| {
                let (m, n, k) = (self.m, self.n, self.k);
                let lhs = self.host(MatmulOperand::Lhs, self.lhs);
                let rhs = self.host(MatmulOperand::Rhs, self.rhs);
                let out = (0..m * n)
                    .map(|at| {
                        let (row, column) = (at / n, at % n);
                        (0..k).map(|kk| lhs.at(row, kk) * rhs.at(column, kk)).sum()
                    })
                    .collect();
                HostData::row_major(HostDataVec::F32(out), Shape::from(vec![m, n]))
            })
            .clone()
    }

    fn tolerance(&self) -> f32 {
        BF16_TOLERANCE
    }
}
