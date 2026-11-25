use crate::kernels::indexed::routed::RoutedMatmulKey;
#[cfg(doc)]
use crate::kernels::indexed::stack::MATMUL_STACK_NOT_WHOLE_SLABS;
use crate::kernels::indexed::stack::{SlabDimension, SlabView};
use crate::kernels::select::problem::OperandKind;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::{MatmulTuneKey, OperandLayout};
use crate::kernels::tune::{Product, matmul_ceiling};
use crate::kernels::{
    MATMUL_SLABS_TAKE_NO_STORED_TILES, MatmulIndex, MatmulIndexTable, MatmulInputs, MatmulRoutine,
    MatmulTensors,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::tune::Work;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::Roofline;
use cubek_std::layout::contiguous;
use cubek_std::matmul_dimensions::{K, N};

/// Operands of a routed product whose shapes are no `[rows, k]` against slabs of `k` rows, each
/// row routed as many times as the next.
pub const ROUTED_SHAPES_DISAGREE: Decline = Decline::Unserved {
    why: "a routed product is [rows, k] against slabs of [k, n], its routes [rows, slots]",
};

/// Routes of a routed product that are not `u32` indices into the stack.
pub const ROUTES_NOT_U32: Decline = Decline::Unserved {
    why: "routes are u32 indices into the stack",
};

/// The operands of one routed product: the rows, the stack of `experts` slabs laid `along`, and
/// each row's routes, `out[r, s] = lhs[r] · rhs[routes[r, s]]`.
///
/// It is a dense product: each row's slot is a batch of one row, `[rows, slots]` batches, and the
/// stack is read through the routes, one entry a batch ([`MatmulIndex::Slab`]), so a dense
/// routine serves it ([`launch`](Self::launch)).
#[derive(Clone)]
pub struct RoutedMatmulTensors {
    client: Client,
    lhs: PlainTensor,
    rhs: StoredTensor,
    routes: PlainTensor,
    experts: usize,
    slots: usize,
    along: SlabDimension,
}

impl core::fmt::Debug for RoutedMatmulTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the product.
        f.debug_struct("RoutedMatmulTensors")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .field("routes", &self.routes)
            .field("experts", &self.experts)
            .field("slots", &self.slots)
            .field("along", &self.along)
            .finish()
    }
}

impl RoutedMatmulTensors {
    /// The most rows a key tells apart: past it a grid is full whatever the count.
    const ROWS_KEYED: usize = 65536;

    /// `lhs[rows, k]` against the stack `rhs` of `experts` slabs laid `along`, float or packed,
    /// each row routed to the slabs its row of `routes[rows, slots]` names, on `client`'s device.
    /// The rows and the routes are made dense, and a float stack is copied only where it is
    /// contiguous along neither dimension.
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`] for a float stack, [`MATMUL_SLABS_TAKE_NO_STORED_TILES`] for a
    /// stack stored in tiles, [`MATMUL_STACK_NOT_WHOLE_SLABS`], [`ROUTED_SHAPES_DISAGREE`] and
    /// [`ROUTES_NOT_U32`].
    pub fn new(
        client: &Client,
        lhs: PlainTensor,
        rhs: StoredTensor,
        routes: PlainTensor,
        experts: usize,
        along: SlabDimension,
    ) -> Result<Self, Decline> {
        if rhs.values().is_tiled() {
            return Err(MATMUL_SLABS_TAKE_NO_STORED_TILES);
        }
        along.admit(&rhs, experts)?;
        let lhs_shape = lhs.shape();
        let [k, _] = along.slab_shape(&rhs, experts);
        if lhs_shape.rank() != 2 || lhs_shape[1] != k {
            return Err(ROUTED_SHAPES_DISAGREE);
        }
        if routes.elem != ElemType::UInt(UIntKind::U32) {
            return Err(ROUTES_NOT_U32);
        }
        let routes_shape = routes.shape();
        let slots = *routes_shape
            .as_slice()
            .last()
            .ok_or(ROUTED_SHAPES_DISAGREE)?;
        if routes_shape.num_elements() != lhs_shape[0] * slots {
            return Err(ROUTED_SHAPES_DISAGREE);
        }
        let rhs = match rhs {
            StoredTensor::Packed(packed) => StoredTensor::Packed(packed),
            StoredTensor::Plain(rhs) => {
                let (_, rhs) = MatmulInputs::new(client, lhs.clone(), rhs)?.into_parts();
                StoredTensor::Plain(SlabView::readable(client, rhs))
            }
        };
        Ok(Self {
            client: client.clone(),
            lhs: lhs.into_contiguous(client),
            rhs,
            routes: routes.into_contiguous(client),
            experts,
            slots,
            along,
        })
    }

    /// The lhs's rows.
    pub fn rows(&self) -> usize {
        self.lhs.shape()[0]
    }

    /// Experts each row selects: the routes a row, and the output rows it writes.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The output's shape: a row per lhs row's slot.
    pub fn output_shape(&self) -> [usize; 3] {
        let [_, n] = self.slab_shape();
        [self.rows(), self.slots(), n]
    }

    /// The element the lhs and the output bind at.
    pub fn elem(&self) -> ElemType {
        self.lhs.elem
    }

    /// The key the routed autotune measures these operands under.
    pub fn key(&self) -> RoutedMatmulKey {
        let [k, n] = self.slab_shape();
        RoutedMatmulKey::new(
            self.elem(),
            k,
            n,
            self.experts,
            MatmulTuneKey::anchored(self.rows(), Self::ROWS_KEYED),
            self.slots,
            self.stack_kind(),
            OperandLayout::along(contiguous(&self.rhs.values().meta(), [K, N])),
            self.along,
        )
    }

    /// An uninitialized output of the product's shape: what [`launch_into`](Self::launch_into)
    /// writes.
    pub fn output(&self) -> TensorHandle {
        PlainTensor::empty_contiguous(&self.client, Shape::from(self.output_shape()), self.elem()).0
    }

    /// The product on `routine` at `settings`, prepared for `hw`: `[rows, slots, n]` at the
    /// lhs's element. No row is an empty answer, with no launch.
    ///
    /// # Errors
    ///
    /// Why `hw` cannot run the product at these settings, or why the routine declines a table or
    /// the stack as it is stored.
    pub fn launch<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let out = self.output();
        self.launch_into(hw, routine, settings, &out)?;
        Ok(out)
    }

    /// [`launch`](Self::launch) into `out`, the product's `[rows, slots, n]` the caller allocated
    /// row-major ([`output`](Self::output)).
    ///
    /// # Errors
    ///
    /// As [`launch`](Self::launch).
    pub fn launch_into<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &R::Settings,
        out: &TensorHandle,
    ) -> Result<(), Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        if self.rows() == 0 {
            return Ok(());
        }
        let (lhs, stack, table) = self.operands()?;
        let bound = self.batched(out);
        routine.launch_indexed_into(hw, &self.client, lhs, stack, table, bound, settings)
    }

    /// Whether `R` at `settings` prepares the product on `hw`, of a row at least.
    ///
    /// # Errors
    ///
    /// As [`launch`](Self::launch).
    pub fn served_by<Mk, SM, R>(&self, hw: &Hardware, settings: &R::Settings) -> Result<(), Decline>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let (lhs, stack, table) = self.operands()?;
        // Left unwritten: preparing a launch reads only how its output lies.
        let out = self.batched(&self.output());
        let tensors = MatmulTensors::indexed(&self.client, lhs, stack, table, out, R::ROWS_KIND)?;
        R::prepare(hw, &tensors.problem(), settings).map(|_| ())
    }

    /// The same product with `routes` in place of the call's.
    pub(crate) fn with_routes(&self, routes: PlainTensor) -> Self {
        Self {
            routes,
            ..self.clone()
        }
    }

    /// Each row's routes, `u32`.
    #[cfg(test)]
    pub(crate) fn routes(&self) -> &PlainTensor {
        &self.routes
    }

    /// The roofline a round stops at: the contraction's operations, and the bytes of the rows, the
    /// output and the slabs the routes can reach, one a route at most.
    pub(crate) fn roofline(&self, hw: &Hardware) -> Roofline {
        let [k, n] = self.slab_shape();
        let routed = self.rows() * self.slots;
        let reached = self.rhs.bytes() / self.experts * routed.min(self.experts);
        Roofline {
            client: self.client.clone(),
            ceiling: matmul_ceiling(hw, self.elem()),
            work: Work {
                compute_ops: 2 * routed * n * k,
                bytes: self.lhs.bytes() + reached + routed * n * self.elem().size(),
            },
        }
    }

    /// The client the operands live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Whether the stack is float or packed, and how.
    pub fn stack_kind(&self) -> OperandKind {
        self.rhs.kind()
    }

    /// The product's kind, for the settings listed for it.
    pub(crate) fn product(&self) -> Product {
        Product::of([self.stack_kind()])
    }

    /// Experts in the stack.
    pub(crate) fn experts(&self) -> usize {
        self.experts
    }

    /// The product as the dense routines read it: each row's slot a batch of one row, the rows
    /// the first batch dim, read again for each of a row's slots; the stack; and the routes, one
    /// entry a batch.
    fn operands(&self) -> Result<(PlainTensor, StoredTensor, MatmulIndexTable), Decline> {
        let [rows, slots, _] = self.output_shape();
        let [k, _] = self.slab_shape();
        let lhs = self
            .lhs
            .as_rows(rows, k)
            .viewed(Metadata::new([rows, 1, 1, k], [k, k, k, 1]));
        let table = MatmulIndexTable {
            index: MatmulIndex::Slab {
                slabs: self.experts,
            },
            entries: self.routes.as_rows(rows, slots),
        };
        Ok((lhs, self.along.read_slabs(&self.rhs, self.experts)?, table))
    }

    /// `out`, the product's `[rows, slots, n]`, as the dense routines write it: each row's slot a
    /// batch of one row, `[rows, slots, 1, n]`.
    fn batched(&self, out: &TensorHandle) -> PlainTensor {
        let [rows, slots, n] = self.output_shape();
        PlainTensor::of_handle(out.clone())
            .viewed(Metadata::new([rows, slots, 1, n], [slots * n, n, n, 1]))
    }

    /// One slab's `[k, n]`.
    fn slab_shape(&self) -> [usize; 2] {
        self.along.slab_shape(&self.rhs, self.experts)
    }
}
