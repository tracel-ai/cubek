//! A dense product with a packed operand, as a model quantized to block-scaled formats runs its
//! projections: a packed weight, a packed activation against a float weight, or both packed. Each
//! routine that reads a packed operand answers the dequantized product.

use std::sync::{Arc, OnceLock};

use cubecl::client::Client;
use cubecl::prelude::Scalar;
use cubecl::quant::scheme::QuantStore;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::MatmulRoutine;
use cubek_matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperand};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, N};
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
    /// NVFP4, packed along `k`, seeded by `seed`.
    Nvfp4 { seed: usize },
    /// An NVFP4 rhs `[n, k]` packed along `k`, seen through the transposed view `[k, n]` a
    /// framework hands a tied embedding to the vocabulary head: its words run along the view's
    /// first axis.
    Nvfp4ViewedTransposed { seed: usize },
}

/// `m × n`, `k` deep, of two operands stored as stated, written in `bf16`.
#[derive(Clone)]
pub(crate) struct PackedProduct {
    m: usize,
    n: usize,
    k: usize,
    lhs: Stored,
    rhs: Stored,
    /// The product on the host, computed once for every setting it is launched at.
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
        Self { m, ..self }
    }

    /// The same product with `n` columns.
    pub(crate) fn n(self, n: usize) -> Self {
        Self { n, ..self }
    }

    /// The same product, `k` deep.
    pub(crate) fn k(self, k: usize) -> Self {
        Self { k, ..self }
    }

    /// `count` `bf16` values in `[-1, 1)`.
    fn halves(count: usize) -> Vec<bf16> {
        (0..count)
            .map(|i| bf16::from_f32(((i * 7919 % 2003) as f32 / 1001.5) - 1.))
            .collect()
    }

    /// The packed slabs of `operand`, `rows` rows of `k`, seeded by `seed`.
    fn spec(&self, operand: MatmulOperand, seed: usize) -> PackedSlabsSpec {
        PackedSlabsSpec {
            format: PackedFormat::Nvfp4,
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
            Stored::Nvfp4 { seed } => {
                StoredTensor::Packed(PackedSlabs::new(client, self.spec(operand, seed)).matrix())
            }
            Stored::Nvfp4ViewedTransposed { seed } => {
                let matrix = PackedSlabs::new(client, self.spec(operand, seed)).matrix();
                StoredTensor::Packed(Self::viewed_transposed(client, matrix, self.k))
            }
        }
    }

    /// `matrix`, `[n, k]` packed along `k`, as a framework's transpose restates it: bound as
    /// `[k, n]` under swapped strides, its blocks and words along its first axis.
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
            (Stored::Nvfp4 { seed } | Stored::Nvfp4ViewedTransposed { seed }, _) => {
                HostOperand::Packed(self.spec(operand, seed).dequantized())
            }
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
    ) -> Result<TensorHandle, Refusal>
    where
        R: MatmulRoutine<M, SM>,
        M: MatmulMicrokernel,
    {
        let (out, bound) = PlainTensor::empty_contiguous(
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
