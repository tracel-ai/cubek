//! What every matmul routine computes, `out = lhs · rhs` batch by batch, on the host from the same
//! seeded inputs the routine reads; and the two ways a test reaches a routine: by name at one of
//! its settings, or through the public [`launch_ref`] at a [`Strategy`], as burn reaches it.

use core::fmt::{Debug, Formatter};

use cubecl::client::Client;
use cubecl::ir::{AddressType, ElemType, FloatKind, IntKind};
use cubecl::prelude::{Scalar, tf32};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem},
    eval::cpu_reference::{matmul_cpu_reference, matmul_epsilon},
    launch::launch_ref,
    strategy::Strategy,
};
use cubek_std::config::MatmulConfig;
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{HostData, HostDataType, ScopeUnderTest, StridedLayout, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);
pub(crate) const F64: ElemType = ElemType::Float(FloatKind::F64);
pub(crate) const I32: ElemType = ElemType::Int(IntKind::I32);

/// The extents of a product: `[m, k] · [k, n]` a batch.
#[derive(Debug, Clone, Copy)]
struct Extents {
    m: usize,
    n: usize,
    k: usize,
}

/// The axis of one operand a stride of zero broadcasts: every index along it reads one element.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Broadcast {
    /// The lhs's batch: one lhs matrix for every batch.
    LhsBatch,
    /// The rhs's batch: one rhs matrix for every batch, as a weight is.
    RhsBatch,
    /// The lhs's rows: every output row reads the same lhs row.
    LhsRows,
}

/// The size of the tiles the rhs is stored in, a tile `n` columns by `k` rows, laid along `n`
/// first: a weight a checkpoint stored for the matrix instruction that reads it.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RhsTiles {
    pub n: usize,
    pub k: usize,
}

#[cfg(feature = "tiled")]
impl RhsTiles {
    /// `rhs`, `[k, n]` of `element`, copied into these tiles.
    fn store(self, client: &Client, rhs: TensorHandle, element: ElemType) -> TensorHandle {
        use cubecl::tile::Axis;
        use cubek_matmul::tiled::storage::{StorageLevels, tile};
        const K: Axis = Axis(0);
        const N: Axis = Axis(1);
        let storage = StorageLevels::new(&[(N, self.n), (K, self.k)]).grid(&[N, K]);
        tile(client, rhs.binding(), &[K, N], element, None, storage)
            .unwrap_or_else(|why| panic!("{self:?} store no such rhs: {why}"))
    }
}

/// The values a product's inputs hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Values {
    /// Seeded in `[-1, 1)` for floats.
    Seeded,
    /// Eighths in `[-1, 1]`, which every float element, tf32 included, holds exactly, and whose
    /// sums an `f32` holds exactly at the depths a test takes.
    Eighths,
}

/// One product a routine runs: its extents and batches, the elements it reads and writes, and how
/// its operands lie in memory.
#[derive(Clone)]
pub(crate) struct Product {
    extents: Extents,
    lhs_batches: Shape,
    rhs_batches: Shape,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    elems: MatmulGlobalElems,
    broadcast: Option<Broadcast>,
    values: Values,
    #[cfg(feature = "tiled")]
    rhs_tiles: Option<RhsTiles>,
}

/// Which operand an input is.
#[derive(Debug, Clone, Copy)]
enum Side {
    Lhs,
    Rhs,
}

impl Product {
    /// A `1 × 1 × 1` product of `element`, unbatched, both operands row-major: its extents are
    /// stated by [`m`](Self::m), [`n`](Self::n) and [`k`](Self::k).
    pub(crate) fn of(element: ElemType) -> Self {
        Self {
            extents: Extents { m: 1, n: 1, k: 1 },
            lhs_batches: shape![1],
            rhs_batches: shape![1],
            lhs_layout: MatrixLayout::RowMajor,
            rhs_layout: MatrixLayout::RowMajor,
            elems: MatmulElems::from_single_dtype(element).as_global_elems(),
            broadcast: None,
            values: Values::Seeded,
            #[cfg(feature = "tiled")]
            rhs_tiles: None,
        }
    }

    /// The same product with `m` rows.
    pub(crate) fn m(mut self, m: usize) -> Self {
        self.extents.m = m;
        self
    }

    /// The same product with `n` columns.
    pub(crate) fn n(mut self, n: usize) -> Self {
        self.extents.n = n;
        self
    }

    /// The same product, `k` deep.
    pub(crate) fn k(mut self, k: usize) -> Self {
        self.extents.k = k;
        self
    }

    /// The same product over batches: the lhs's and the rhs's, a `1` broadcast against the other.
    pub(crate) fn batched(self, lhs: Shape, rhs: Shape) -> Self {
        Self {
            lhs_batches: lhs,
            rhs_batches: rhs,
            ..self
        }
    }

    /// The same product, its lhs laid along `m`, as a weight gradient's transposed activation is.
    pub(crate) fn lhs_along_m(self) -> Self {
        Self {
            lhs_layout: MatrixLayout::ColMajor,
            ..self
        }
    }

    /// The same product, its rhs laid along `k`, as a linear layer's weight is.
    pub(crate) fn rhs_along_k(self) -> Self {
        Self {
            rhs_layout: MatrixLayout::ColMajor,
            ..self
        }
    }

    /// The same product, written in `out` rather than its inputs' element.
    pub(crate) fn writing(self, out: ElemType) -> Self {
        Self {
            elems: MatmulGlobalElems { out, ..self.elems },
            ..self
        }
    }

    /// The same product, the stride of `axis` zero.
    pub(crate) fn broadcast(self, axis: Broadcast) -> Self {
        Self {
            broadcast: Some(axis),
            ..self
        }
    }

    /// The same product, its inputs eighths: what any element reads exactly, so an `f32` product
    /// read at tf32 is held to `f32`'s tolerance.
    pub(crate) fn in_eighths(self) -> Self {
        Self {
            values: Values::Eighths,
            ..self
        }
    }

    /// The same product, its rhs stored in `tiles`.
    #[cfg(feature = "tiled")]
    pub(crate) fn rhs_in(self, tiles: RhsTiles) -> Self {
        Self {
            rhs_tiles: Some(tiles),
            ..self
        }
    }

    /// The element both inputs hold.
    pub(crate) fn input_element(&self) -> ElemType {
        self.elems.lhs
    }

    /// The problem as the multi-level routines state it.
    pub(crate) fn problem(&self) -> MatmulProblem {
        let Extents { m, n, k } = self.extents;
        let mut problem = MatmulProblem::from_parameters(
            m,
            n,
            k,
            self.lhs_batches.clone(),
            self.rhs_batches.clone(),
            self.lhs_layout,
            self.rhs_layout,
            MatrixLayout::RowMajor,
            None,
            None,
            self.elems.clone(),
            AddressType::U32,
        );
        // Strides `[batch, row, column]`.
        match self.broadcast {
            Some(Broadcast::LhsBatch) => problem.lhs_strides[0] = 0,
            Some(Broadcast::RhsBatch) => problem.rhs_strides[0] = 0,
            Some(Broadcast::LhsRows) => problem.lhs_strides[1] = 0,
            None => {}
        }
        problem
    }

    /// The operand `side` as the routine reads it, and its values on the host. Floats are seeded
    /// in `[-1, 1)`; integers are small and distinct enough that a wrong pairing of rows and
    /// columns changes the sum.
    fn operand(&self, client: &Client, side: Side) -> (TensorHandle, HostData) {
        let problem = self.problem();
        let (shape, element, layout, strides, seed, (step, offset)) = match side {
            Side::Lhs => (
                problem.lhs_shape,
                self.elems.lhs,
                self.lhs_layout,
                problem.lhs_strides,
                1234,
                (1, 0),
            ),
            Side::Rhs => (
                problem.rhs_shape,
                self.elems.rhs,
                self.rhs_layout,
                problem.rhs_strides,
                5678,
                (3, 1),
            ),
        };
        let layout = match (self.broadcast, layout) {
            (Some(_), _) => StridedLayout::Explicit(strides.to_vec()),
            (None, MatrixLayout::RowMajor) => StridedLayout::RowMajor,
            (None, MatrixLayout::ColMajor) => StridedLayout::ColMajor,
        };
        let input = TestInput::builder(client.clone(), shape.clone())
            .dtype(element)
            .layout(layout);
        let count = shape.num_elements();
        let input = match (element, self.values) {
            (ElemType::Int(_), _) => input.custom(
                (0..count as i64)
                    .map(|i| ((i * 7 + seed) % 11 - 5) as f32)
                    .collect(),
            ),
            (_, Values::Eighths) => input.custom(
                (0..count)
                    .map(|i| ((i * step + offset) % 17) as f32 / 8. - 1.)
                    .collect(),
            ),
            (_, Values::Seeded) => input.uniform(seed as u64, -1., 1.),
        };
        input.generate_with_host_data(HostDataType::from(element))
    }

    /// An output filled with values no product here reaches, so a kernel that wrote nothing shows.
    fn poisoned_out(&self, client: &Client) -> TensorHandle {
        let problem = self.problem();
        let out = TestInput::builder(client.clone(), problem.out_shape.clone())
            .dtype(self.elems.out)
            .layout(StridedLayout::RowMajor);
        match self.elems.out {
            ElemType::Int(_) => out.custom(vec![77_777.; problem.out_shape.num_elements()]),
            _ => out.uniform(4242, 10., 100.),
        }
        .generate_without_host_data()
    }

    /// The operands and the poisoned output a launch reads and writes: the rhs in its tiles where
    /// the product stores it so.
    pub(crate) fn tensors(&self, client: &Client) -> (TensorHandle, TensorHandle, TensorHandle) {
        let (lhs, _) = self.operand(client, Side::Lhs);
        let (rhs, _) = self.operand(client, Side::Rhs);
        #[cfg(feature = "tiled")]
        let rhs = match self.rhs_tiles {
            Some(tiles) => tiles.store(client, rhs, self.elems.rhs),
            None => rhs,
        };
        (lhs, rhs, self.poisoned_out(client))
    }

    /// The product on the host, in `f64`, of the inputs as the routine's elements hold them.
    pub(crate) fn reference(&self) -> HostData {
        let client = cubecl::test_device().client();
        let (_, lhs) = self.operand(&client, Side::Lhs);
        let (_, rhs) = self.operand(&client, Side::Rhs);
        matmul_cpu_reference(&lhs, &rhs, &self.problem(), None)
    }

    /// Exact for integers. For floats, the epsilon of every element the product is read and summed
    /// at, five hundred times over: seeded `f32` inputs read at tf32 where the process allows it and
    /// the device takes it.
    pub(crate) fn tolerance(&self, client: &Client) -> f32 {
        if let ElemType::Int(_) = self.elems.out {
            return 0.;
        }
        let mut read = MatmulElems::from_globals(&self.elems);
        let tf32 = tf32::elem_type_native();
        if self.values == Values::Seeded
            && self.elems.lhs == F32
            && self.elems.rhs == F32
            && MatmulConfig::get().allow_tf32
            && client.properties().features.supports_type(tf32)
        {
            read.lhs_stage = tf32;
            read.rhs_stage = tf32;
            read.lhs_register = tf32;
            read.rhs_register = tf32;
        }
        matmul_epsilon(&read, 500.)
    }

    pub(crate) fn written(out: TensorHandle) -> Vec<Written> {
        let host = HostDataType::from(out.dtype);
        vec![Written {
            name: "out",
            tensor: out,
            host,
        }]
    }
}

impl Debug for Product {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let Extents { m, n, k } = self.extents;
        write!(
            f,
            "{:?}·{:?} batches, {m}×{n}×{k}, lhs {:?} rhs {:?}, {:?}·{:?} into {:?}",
            self.lhs_batches.as_slice(),
            self.rhs_batches.as_slice(),
            self.lhs_layout,
            self.rhs_layout,
            self.elems.lhs,
            self.elems.rhs,
            self.elems.out,
        )?;
        if let Some(axis) = self.broadcast {
            write!(f, ", {axis:?} broadcast")?;
        }
        if self.values == Values::Eighths {
            write!(f, ", in eighths")?;
        }
        #[cfg(feature = "tiled")]
        if let Some(tiles) = self.rhs_tiles {
            write!(f, ", rhs in {tiles:?}")?;
        }
        Ok(())
    }
}

/// Which refusals a scope may answer a product with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declines {
    /// None: a floor, which serves any product on any device.
    Never,
    /// Its own, where it serves no such product on the test device.
    AsItStates,
}

/// A [`Strategy`] reached through [`launch_ref`], against the host's product.
pub(crate) struct ByStrategy {
    strategy: Strategy,
    declines: Declines,
}

impl ByStrategy {
    pub(crate) fn new(strategy: impl Into<Strategy>, declines: Declines) -> Self {
        Self {
            strategy: strategy.into(),
            declines,
        }
    }
}

impl ScopeUnderTest for ByStrategy {
    type Case = Product;

    /// # Panics
    ///
    /// Where a strategy that declines nothing refuses a product.
    fn launch(&self, client: &Client, product: &Product) -> Result<Vec<Written>, Refusal> {
        let (lhs, rhs, out) = product.tensors(client);
        let elems = &product.elems;
        let launched = launch_ref(
            &self.strategy,
            client,
            InputBinding::Normal(lhs.binding(), elems.lhs),
            InputBinding::Normal(rhs.binding(), elems.rhs),
            out.clone().binding(),
            &mut MatmulElems::from_globals(elems),
        );
        match (launched, self.declines) {
            (Ok(()), _) => Ok(Product::written(out)),
            (Err(_), Declines::AsItStates) => Err(Refusal::Declined {
                why: "the strategy serves no such product on this device",
            }),
            (Err(why), Declines::Never) => {
                panic!("{} refused {product:?}: {why}", self.strategy)
            }
        }
    }

    fn reference(&self, product: &Product) -> Vec<HostData> {
        vec![product.reference()]
    }

    fn tolerance(&self, product: &Product) -> f32 {
        product.tolerance(&cubecl::test_device().client())
    }
}
