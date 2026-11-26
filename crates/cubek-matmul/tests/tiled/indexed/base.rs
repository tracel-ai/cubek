//! Products whose inputs read a table, one entry per batch: each batch against the slab of a
//! stacked rhs its entry names, or over its run of `K`. An entry past the stack, or an empty run,
//! answers zeros. Each product holds its own host reference.

use core::num::NonZeroUsize;
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{MatmulIndexTable, MatmulRoutine, MatmulRowRuns, TileRows};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperand};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataVec};
use half::f16;

use crate::tiled::packed::{HostSlabs, PackedFormat, PackedSlabs, PackedSlabsSpec};
use crate::tiled::{RoutineProduct, upload};

/// Close enough for an `f32` output.
const FLOAT_TOLERANCE: f32 = 1e-3;

/// Loose enough for an `f16` output.
const HALF_TOLERANCE: f32 = 1e-2;

/// Deterministic values in `[-1, 1)`, different for each `seed`.
pub(crate) fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7919 + seed * 104_729) % 2003) as f32 / 1001.5) - 1.)
        .collect()
}

/// How the stack a table indexes is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stack {
    /// `f32` slabs against an `f32` activation.
    Float,
    /// Slabs packed in `format`, words and scales both, against an `f16` activation.
    Packed(PackedFormat),
}

impl Stack {
    fn tolerance(self) -> f32 {
        match self {
            Stack::Float => FLOAT_TOLERANCE,
            Stack::Packed(_) => HALF_TOLERANCE,
        }
    }

    /// The activation, `values(count, seed)` as the stack's activation element holds them: its
    /// tensor of `shape`, and its values read back.
    fn activation(self, client: &Client, shape: &[usize], seed: usize) -> (PlainTensor, Vec<f32>) {
        let floats = values(shape.iter().product(), seed);
        match self {
            Stack::Float => (upload(client, shape, &floats), floats),
            Stack::Packed(_) => {
                let halves: Vec<f16> = floats.into_iter().map(f16::from_f32).collect();
                let floats = halves.iter().map(|half| half.to_f32()).collect();
                (upload(client, shape, &halves), floats)
            }
        }
    }

    /// The activation's values as its element holds them, on the host.
    fn activation_values(self, count: usize, seed: usize) -> Vec<f32> {
        let floats = values(count, seed);
        match self {
            Stack::Float => floats,
            Stack::Packed(_) => floats
                .into_iter()
                .map(|value| f16::from_f32(value).to_f32())
                .collect(),
        }
    }
}

/// `tokens` rows, each routed to `picks` of `slabs` slabs of `[k, n]`: batches `[tokens, picks]` of
/// one row, each against the slab its route names. The last route names one past the stack.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Routed {
    pub tokens: usize,
    pub picks: usize,
    pub slabs: usize,
    pub k: usize,
    pub n: usize,
    pub stack: Stack,
}

impl Routed {
    /// A mixture of experts' routed product.
    pub(crate) const MOE: Self = Self {
        tokens: 5,
        picks: 3,
        slabs: 4,
        k: 128,
        n: 96,
        stack: Stack::Float,
    };

    /// The same, its `n` no multiple of the column tile.
    pub(crate) const NARROW: Self = Self { n: 80, ..Self::MOE };

    /// The same product, its stack packed in `format`.
    pub(crate) const fn packed(self, format: PackedFormat) -> Self {
        Self {
            stack: Stack::Packed(format),
            ..self
        }
    }

    pub(crate) fn routes(&self) -> Vec<u32> {
        let mut routes: Vec<u32> = (0..self.tokens * self.picks)
            .map(|i| ((i * 3 + 1) % self.slabs) as u32)
            .collect();
        *routes.last_mut().unwrap() = self.slabs as u32;
        routes
    }

    /// The activation, `[tokens, k]`, and its values on the host.
    pub(crate) fn activation(&self, client: &Client) -> (PlainTensor, Vec<f32>) {
        self.stack.activation(client, &[self.tokens, self.k], 1)
    }

    /// The stack's packed slabs, where it packs them.
    pub(crate) fn packed_slabs(&self) -> Option<PackedSlabsSpec> {
        match self.stack {
            Stack::Float => None,
            Stack::Packed(format) => Some(PackedSlabsSpec {
                format,
                operand: MatmulOperand::Rhs,
                slabs: self.slabs,
                rows: self.n,
                k: self.k,
                seed: 2,
            }),
        }
    }

    /// The stack's values, `stack(slab, k, j)`, on the host.
    fn stack_values(&self) -> HostStack {
        match self.packed_slabs() {
            Some(spec) => HostStack::Packed(spec.dequantized()),
            None => HostStack::Float {
                k: self.k,
                n: self.n,
                values: values(self.slabs * self.k * self.n, 2),
            },
        }
    }

    /// The product, `[tokens, picks, n]` row-major, on the host.
    pub(crate) fn expected(&self) -> Vec<f32> {
        let Self {
            tokens,
            picks,
            slabs,
            k,
            n,
            ..
        } = *self;
        let lhs = self.stack.activation_values(tokens * k, 1);
        let stack = self.stack_values();
        let routes = self.routes();
        let mut out = vec![0.; tokens * picks * n];
        for t in 0..tokens {
            for p in 0..picks {
                let slab = routes[t * picks + p] as usize;
                if slab >= slabs {
                    continue;
                }
                for j in 0..n {
                    out[(t * picks + p) * n + j] = (0..k)
                        .map(|kk| lhs[t * k + kk] * stack.cell(slab, kk, j))
                        .sum();
                }
            }
        }
        out
    }

    pub(crate) fn tolerance(&self) -> f32 {
        self.stack.tolerance()
    }
}

impl RoutineProduct for Routed {
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
        let Self {
            tokens, picks, k, ..
        } = *self;
        // A route is a batch of one row: the activation's rows are its first batch dim, broadcast
        // along the picks.
        let (lhs, _) = self.stack.activation(client, &[tokens, 1, 1, k], 1);
        let stack = match self.packed_slabs() {
            Some(spec) => PackedSlabs::new(client, spec).stack(),
            None => StoredTensor::Plain(upload(
                client,
                &[self.slabs, k, self.n],
                &values(self.slabs * k * self.n, 2),
            )),
        };
        let table = MatmulIndexTable::Slab {
            slabs: self.slabs,
            entries: upload(client, &[tokens, picks], &self.routes()),
        };
        let shape = Shape::from(vec![tokens, picks, 1, self.n]);
        let (out, bound) = Self::poisoned(client, shape, lhs.elem);
        routine.launch_indexed_into(hw, client, lhs, stack, table, bound, settings)?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let shape = vec![self.tokens, self.picks, 1, self.n];
        HostData::row_major(HostDataVec::F32(self.expected()), Shape::from(shape))
    }

    fn tolerance(&self) -> f32 {
        Routed::tolerance(self)
    }
}

/// A group of rows against one slab of a packed stack, one past it naming no slab.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowGroup {
    pub rows: usize,
    pub slab: usize,
}

/// A flat, unpadded activation `[rows, k]` cut into tiles of at most `tile_rows` rows, each group
/// in whole tiles but its last and an empty group in one empty tile, then `sentinels` empty tiles
/// past the groups and `trailing` rows no tile covers. The output is the flat `[rows, n]`, poisoned:
/// only the rows a tile's run covers are written.
#[derive(Debug, Clone)]
pub(crate) struct RowRuns {
    pub tile_rows: usize,
    pub groups: Vec<RowGroup>,
    pub sentinels: usize,
    pub trailing: usize,
    pub k: usize,
    pub format: PackedFormat,
}

impl RowRuns {
    const SLABS: usize = 4;
    const N: usize = 96;

    /// An empty group first, a group ending mid-tile, an empty one, a last tile whose slab is one
    /// past the stack, two sentinel tiles and seven trailing rows, against a stack in `format`.
    pub(crate) fn ragged(tile_rows: usize, k: usize, format: PackedFormat) -> Self {
        let groups = [
            (0, 0),
            (2 * tile_rows + tile_rows / 2, 1),
            (0, 2),
            (3, Self::SLABS),
        ];
        Self {
            tile_rows,
            groups: groups
                .iter()
                .map(|&(rows, slab)| RowGroup { rows, slab })
                .collect(),
            sentinels: 2,
            trailing: 7,
            k,
            format,
        }
    }

    /// Each tile's run `(start, end)` and slab.
    fn tiles(&self) -> Vec<(usize, usize, u32)> {
        let mut tiles = Vec::new();
        let mut start = 0;
        for group in &self.groups {
            let slab = group.slab as u32;
            if group.rows == 0 {
                tiles.push((start, start, slab));
            }
            for first in (0..group.rows).step_by(self.tile_rows) {
                let rows = self.tile_rows.min(group.rows - first);
                tiles.push((start + first, start + first + rows, slab));
            }
            start += group.rows;
        }
        tiles.extend((0..self.sentinels).map(|_| (start, start, Self::SLABS as u32)));
        tiles
    }

    fn rows(&self) -> usize {
        self.groups.iter().map(|group| group.rows).sum::<usize>() + self.trailing
    }

    fn packed_slabs(&self) -> PackedSlabsSpec {
        PackedSlabsSpec {
            format: self.format,
            operand: MatmulOperand::Rhs,
            slabs: Self::SLABS,
            rows: Self::N,
            k: self.k,
            seed: 2,
        }
    }
}

impl RoutineProduct for RowRuns {
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
        let stack = Stack::Packed(self.format);
        let (lhs, _) = stack.activation(client, &[self.rows(), self.k], 1);
        let tiles = self.tiles();
        let entries: Vec<u32> = tiles.iter().map(|&(_, _, slab)| slab).collect();
        let ends: Vec<u32> = tiles.iter().map(|&(_, end, _)| end as u32).collect();
        let table = MatmulIndexTable::RowRuns(Box::new(MatmulRowRuns {
            slabs: Self::SLABS,
            tile_rows: TileRows(NonZeroUsize::new(self.tile_rows).expect("tiles hold rows")),
            entries: upload(client, &[tiles.len()], &entries),
            runs: upload(client, &[tiles.len()], &ends),
        }));
        let shape = Shape::from(vec![self.rows(), Self::N]);
        let (out, bound) = Self::poisoned(client, shape, lhs.elem);
        let rhs = PackedSlabs::new(client, self.packed_slabs()).stack();
        routine.launch_indexed_into(hw, client, lhs, rhs, table, bound, settings)?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let (k, n) = (self.k, Self::N);
        let lhs = Stack::Packed(self.format).activation_values(self.rows() * k, 1);
        let stack = self.packed_slabs().dequantized();
        let mut out = vec![f32::NAN; self.rows() * n];
        for (start, end, slab) in self.tiles() {
            for row in start..end {
                for j in 0..n {
                    out[row * n + j] = match slab as usize >= Self::SLABS {
                        true => 0.,
                        false => (0..k)
                            .map(|kk| lhs[row * k + kk] * stack.cell(slab as usize, kk, j))
                            .sum(),
                    };
                }
            }
        }
        HostData::row_major(HostDataVec::F32(out), Shape::from(vec![self.rows(), n]))
    }

    fn tolerance(&self) -> f32 {
        HALF_TOLERANCE
    }
}

/// A stack's values on the host, read `stack(slab, k, j)`.
enum HostStack {
    Float {
        k: usize,
        n: usize,
        values: Vec<f32>,
    },
    Packed(HostSlabs),
}

impl HostStack {
    fn cell(&self, slab: usize, k: usize, j: usize) -> f32 {
        match self {
            HostStack::Float {
                k: depth,
                n,
                values,
            } => values[(slab * depth + k) * n + j],
            HostStack::Packed(slabs) => slabs.cell(slab, k, j),
        }
    }
}

/// `rows` rows grouped by `ends`, each group's cumulative end: `out[g] = lhs[.., rows of g] ·
/// rhs[rows of g, ..]`, batches `[groups]` over the runs of `K`.
#[derive(Debug, Clone)]
pub(crate) struct RunsOfK {
    pub m: usize,
    pub n: usize,
    pub ends: Vec<u32>,
}

impl RunsOfK {
    fn rows(&self) -> usize {
        *self.ends.last().unwrap() as usize
    }
}

impl RoutineProduct for RunsOfK {
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
        let (m, n, rows) = (self.m, self.n, self.rows());
        let lhs = upload(client, &[m, rows], &values(m * rows, 3));
        let rhs = upload(client, &[rows, n], &values(rows * n, 4));
        let table = MatmulIndexTable::RunOfK {
            ends: upload(client, &[self.ends.len()], &self.ends),
        };
        let shape = Shape::from(vec![self.ends.len(), m, n]);
        let (out, bound) = Self::poisoned(client, shape, lhs.elem);
        routine.launch_indexed_into(
            hw,
            client,
            lhs,
            StoredTensor::Plain(rhs),
            table,
            bound,
            settings,
        )?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let (m, n, rows) = (self.m, self.n, self.rows());
        let (lhs, rhs) = (values(m * rows, 3), values(rows * n, 4));
        let mut out = vec![0.; self.ends.len() * m * n];
        let mut start = 0;
        for (g, &end) in self.ends.iter().enumerate() {
            for i in 0..m {
                for j in 0..n {
                    out[(g * m + i) * n + j] = (start..end as usize)
                        .map(|r| lhs[i * rows + r] * rhs[r * n + j])
                        .sum();
                }
            }
            start = end as usize;
        }
        let shape = Shape::from(vec![self.ends.len(), m, n]);
        HostData::row_major(HostDataVec::F32(out), shape)
    }

    fn tolerance(&self) -> f32 {
        FLOAT_TOLERANCE
    }
}

/// `chunks` chunks of `rows` rows each, sorted by slab and padded to whole chunks: batches
/// `[chunks]`, each against the slab its entry names. The last chunk is padding, its entry one past
/// the stack.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Chunked {
    pub chunks: usize,
    pub rows: usize,
    pub slabs: usize,
    pub k: usize,
    pub n: usize,
}

impl Chunked {
    fn slab_of_each(&self) -> Vec<u32> {
        let mut slabs: Vec<u32> = (0..self.chunks)
            .map(|c| ((c * 5 + 2) % self.slabs) as u32)
            .collect();
        *slabs.last_mut().unwrap() = self.slabs as u32;
        slabs
    }
}

impl RoutineProduct for Chunked {
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
        let Self {
            chunks,
            rows,
            slabs,
            k,
            n,
        } = *self;
        let lhs = upload(client, &[chunks, rows, k], &values(chunks * rows * k, 5));
        let stack = upload(client, &[slabs, k, n], &values(slabs * k * n, 6));
        let table = MatmulIndexTable::Slab {
            slabs,
            entries: upload(client, &[chunks], &self.slab_of_each()),
        };
        let shape = Shape::from(vec![chunks, rows, n]);
        let (out, bound) = Self::poisoned(client, shape, lhs.elem);
        routine.launch_indexed_into(
            hw,
            client,
            lhs,
            StoredTensor::Plain(stack),
            table,
            bound,
            settings,
        )?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let Self {
            chunks,
            rows,
            slabs,
            k,
            n,
        } = *self;
        let (lhs, stack) = (values(chunks * rows * k, 5), values(slabs * k * n, 6));
        let mut out = vec![0.; chunks * rows * n];
        for (c, &slab) in self.slab_of_each().iter().enumerate() {
            let slab = slab as usize;
            if slab >= slabs {
                continue;
            }
            for r in c * rows..(c + 1) * rows {
                for j in 0..n {
                    out[r * n + j] = (0..k)
                        .map(|kk| lhs[r * k + kk] * stack[(slab * k + kk) * n + j])
                        .sum();
                }
            }
        }
        HostData::row_major(HostDataVec::F32(out), Shape::from(vec![chunks, rows, n]))
    }

    fn tolerance(&self) -> f32 {
        FLOAT_TOLERANCE
    }
}

/// Rows grouped by `sizes`, each group a slab of a packed stack: what a mixture of experts' tokens
/// sorted by expert make. The second group is empty.
#[derive(Debug, Clone)]
pub(crate) struct GroupedRows {
    pub k: usize,
    pub n: usize,
    pub sizes: [usize; 4],
    pub format: PackedFormat,
}

impl GroupedRows {
    /// Groups of 5, 0, 11 and 2 rows, `128` deep, `96` wide, against slabs packed in `format`.
    pub(crate) fn of(format: PackedFormat) -> Self {
        Self {
            k: 128,
            n: 96,
            sizes: [5, 0, 11, 2],
            format,
        }
    }

    /// The same rows, grouped by `sizes`.
    pub(crate) fn sized(self, sizes: [usize; 4]) -> Self {
        Self { sizes, ..self }
    }

    pub(crate) fn rows(&self) -> usize {
        self.sizes.iter().sum()
    }

    pub(crate) fn ends(&self) -> Vec<u32> {
        self.sizes
            .iter()
            .scan(0, |end, size| {
                *end += size;
                Some(*end as u32)
            })
            .collect()
    }

    /// The activation, `[rows, k]` in `f16`.
    pub(crate) fn activation(&self, client: &Client) -> PlainTensor {
        Stack::Packed(self.format)
            .activation(client, &[self.rows(), self.k], 3)
            .0
    }

    /// The stack's slabs, a group's each.
    pub(crate) fn packed_slabs(&self) -> PackedSlabsSpec {
        PackedSlabsSpec {
            format: self.format,
            operand: MatmulOperand::Rhs,
            slabs: self.sizes.len(),
            rows: self.n,
            k: self.k,
            seed: 4,
        }
    }

    /// The product, `[rows, n]`, on the host.
    pub(crate) fn reference(&self) -> HostData {
        let (k, n) = (self.k, self.n);
        let lhs = Stack::Packed(self.format).activation_values(self.rows() * k, 3);
        let stack = self.packed_slabs().dequantized();
        let ends = self.ends();
        let mut out = vec![0.; self.rows() * n];
        for row in 0..self.rows() {
            let group = ends
                .iter()
                .position(|&end| row < end as usize)
                .expect("every row is in a group");
            for j in 0..n {
                out[row * n + j] = (0..k)
                    .map(|kk| lhs[row * k + kk] * stack.cell(group, kk, j))
                    .sum();
            }
        }
        HostData::row_major(HostDataVec::F32(out), Shape::from(vec![self.rows(), n]))
    }

    pub(crate) fn tolerance(&self) -> f32 {
        HALF_TOLERANCE
    }
}
