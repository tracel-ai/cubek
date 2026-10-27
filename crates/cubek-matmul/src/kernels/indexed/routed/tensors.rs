use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::layout::contiguous;
use cubek_std::matmul_axes::{K, N};

use crate::kernels::indexed::routed::RoutedMatmulKey;
use crate::kernels::indexed::stack::{SlabAxis, SlabView};
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::OperandLayout;
use crate::kernels::{
    MATMUL_SLABS_TAKE_NO_STORED_TILES, MatmulIndex, MatmulIndexTable, MatmulInputs, MatmulRoutine,
    MatmulTensors,
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
    along: SlabAxis,
}

impl RoutedMatmulTensors {
    /// `lhs[rows, k]` against the stack `rhs` of `experts` slabs laid `along`, float or packed,
    /// each row routed to the slabs its row of `routes[rows, slots]` names, on `client`'s device.
    /// The rows and the routes are made dense, and a float stack is copied only where it is
    /// contiguous along neither axis.
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`] for a float stack, and [`MATMUL_SLABS_TAKE_NO_STORED_TILES`] for a
    /// stack stored in tiles.
    ///
    /// # Panics
    ///
    /// On an lhs other than `[rows, k]`, a stack other than rank two or not whole slabs, a
    /// contraction the slabs do not share, or routes other than `u32`.
    pub fn new(
        client: &Client,
        lhs: PlainTensor,
        rhs: StoredTensor,
        routes: PlainTensor,
        experts: usize,
        along: SlabAxis,
    ) -> Result<Self, Refusal> {
        if rhs.values().is_tiled() {
            return Err(MATMUL_SLABS_TAKE_NO_STORED_TILES);
        }
        assert_eq!(
            rhs.values().shape().rank(),
            2,
            "a routed stack is a rank-two view"
        );
        let rhs = match rhs {
            StoredTensor::Packed(packed) => StoredTensor::Packed(packed),
            StoredTensor::Plain(rhs) => {
                let (_, rhs) = MatmulInputs::new(client, lhs.clone(), rhs)?.into_parts();
                StoredTensor::Plain(SlabView::readable(client, rhs))
            }
        };
        assert_eq!(
            routes.elem,
            ElemType::UInt(UIntKind::U32),
            "routes are `u32` indices into the stack"
        );
        let lhs_shape = lhs.shape();
        assert_eq!(lhs_shape.rank(), 2, "a routed lhs is `[rows, k]`");
        let [k, _] = along.slab_shape(&rhs, experts);
        assert_eq!(lhs_shape[1], k, "the rows contract against one slab");
        let routes_shape = routes.shape();
        let slots = *routes_shape
            .as_slice()
            .last()
            .expect("routes hold each row's experts along their last dim");
        assert_eq!(
            routes_shape.num_elements(),
            lhs_shape[0] * slots,
            "every row selects {slots} experts: routes {routes_shape:?} over {} rows",
            lhs_shape[0]
        );
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
            self.slots(),
            self.rows(),
            self.rhs.kind(),
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
    ) -> Result<TensorHandle, Refusal>
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
    ) -> Result<(), Refusal>
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

    /// Whether `routine` at `settings` prepares the product on `hw`, of a row at least.
    ///
    /// # Errors
    ///
    /// As [`launch`](Self::launch).
    pub fn served_by<Mk, SM, R>(
        &self,
        hw: &Hardware,
        routine: &R,
        settings: &R::Settings,
    ) -> Result<(), Refusal>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let (lhs, stack, table) = self.operands()?;
        // Left unwritten: preparing a launch reads only how its output lies.
        let out = self.batched(&self.output());
        let tensors = MatmulTensors::indexed::<Mk>(&self.client, lhs, stack, table, out)?;
        routine
            .prepare(hw, &tensors.problem(), settings)
            .map(|_| ())
    }

    /// The client the operands live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Whether the stack is packed, read as it is stored.
    pub fn packed(&self) -> bool {
        matches!(self.rhs, StoredTensor::Packed(_))
    }

    /// The product as the dense routines read it: each row's slot a batch of one row, the rows
    /// the first batch dim, read again for each of a row's slots; the stack; and the routes, one
    /// entry a batch.
    fn operands(&self) -> Result<(PlainTensor, StoredTensor, MatmulIndexTable), Refusal> {
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
