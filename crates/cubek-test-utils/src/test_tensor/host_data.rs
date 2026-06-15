use cubecl::{
    CubeElement, TestRuntime,
    client::ComputeClient,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::test_tensor::{cast::copy_casted, strides::physical_extent};

#[derive(Debug, Clone)]
pub struct HostData {
    pub data: HostDataVec,
    pub shape: Shape,
    pub strides: Strides,
}

#[derive(Eq, PartialEq, PartialOrd, Clone, Copy, Debug)]
pub enum HostDataType {
    F32,
    I32,
    Bool,
}

#[derive(Clone, Debug)]
pub enum HostDataVec {
    F32(Vec<f32>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
}

impl HostDataVec {
    pub fn get_f32(&self, i: usize) -> f32 {
        match self {
            HostDataVec::F32(items) => items[i],
            _ => panic!("Can't get as f32"),
        }
    }

    pub fn get_bool(&self, i: usize) -> bool {
        match self {
            HostDataVec::Bool(items) => items[i],
            _ => panic!("Can't get as bool"),
        }
    }

    pub fn get_i32(&self, i: usize) -> i32 {
        match self {
            HostDataVec::I32(items) => items[i],
            _ => panic!("Can't get as i32"),
        }
    }

    pub fn try_get_f32(&self, i: usize) -> Option<f32> {
        match self {
            HostDataVec::F32(items) => items.get(i).copied(),
            _ => None,
        }
    }

    pub fn try_get_i32(&self, i: usize) -> Option<i32> {
        match self {
            HostDataVec::I32(items) => items.get(i).copied(),
            _ => None,
        }
    }

    pub fn try_get_bool(&self, i: usize) -> Option<bool> {
        match self {
            HostDataVec::Bool(items) => items.get(i).copied(),
            _ => None,
        }
    }
}

impl HostData {
    pub fn from_tensor_handle(
        client: &ComputeClient<TestRuntime>,
        mut tensor_handle: TensorHandle<TestRuntime>,
        host_data_type: HostDataType,
    ) -> Self {
        let shape = tensor_handle.shape().clone();
        let strides = tensor_handle.strides().clone();

        // Reshape to a flat 1D view of the full physical buffer so the read
        // covers every offset the jumpy strides might reach. Without this, a
        // shape like [256,256] with strides [512,1] would only read the
        // shape.product() (65536) elements that `copy_casted`'s contiguous
        // rewrite walks, and HostData.get_f32 would then index out-of-bounds
        // when the logical walk crosses the padding.
        let physical_len = physical_extent(&shape, &strides);
        tensor_handle.metadata.shape = Shape::from(vec![physical_len]);
        tensor_handle.metadata.strides = Strides::new(&[1]);

        let data = match host_data_type {
            HostDataType::F32 => {
                let handle = copy_casted(
                    client,
                    tensor_handle,
                    f32::as_type_native_unchecked().storage_type(),
                );
                let data = f32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::F32(data)
            }
            HostDataType::I32 => {
                let handle = copy_casted(
                    client,
                    tensor_handle,
                    i32::as_type_native_unchecked().storage_type(),
                );
                let data = i32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::I32(data)
            }
            HostDataType::Bool => {
                let handle = copy_casted(
                    client,
                    tensor_handle,
                    u32::as_type_native_unchecked().storage_type(),
                );
                let data = u32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::Bool(data.iter().map(|&x| x > 0).collect())
            }
        };

        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn get_f32(&self, index: &[usize]) -> f32 {
        self.data.get_f32(self.strided_index(index))
    }

    pub fn get_bool(&self, index: &[usize]) -> bool {
        self.data.get_bool(self.strided_index(index))
    }

    pub fn get_i32(&self, index: &[usize]) -> i32 {
        self.data.get_i32(self.strided_index(index))
    }

    /// Like [`get_f32`], but returns `None` if the underlying data isn't `F32`
    /// (or the index is out of bounds), instead of panicking.
    pub fn try_get_f32(&self, index: &[usize]) -> Option<f32> {
        self.data.try_get_f32(self.strided_index(index))
    }

    pub fn try_get_i32(&self, index: &[usize]) -> Option<i32> {
        self.data.try_get_i32(self.strided_index(index))
    }

    pub fn try_get_bool(&self, index: &[usize]) -> Option<bool> {
        self.data.try_get_bool(self.strided_index(index))
    }

    /// Iterate every logical index in row-major order, yielding the index vector.
    ///
    /// Useful when callers want to walk a non-contiguous tensor without
    /// re-implementing the rank recursion themselves.
    pub fn iter_indices(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        IndexIter::new(self.shape.as_slice().to_vec())
    }

    /// Iterate `(index, f32 value)` pairs in row-major order.
    /// Panics if the underlying data isn't `F32`.
    pub fn iter_indexed_f32(&self) -> impl Iterator<Item = (Vec<usize>, f32)> + '_ {
        self.iter_indices().map(move |idx| {
            let v = self.get_f32(&idx);
            (idx, v)
        })
    }

    /// Iterate `(index, i32 value)` pairs in row-major order.
    /// Panics if the underlying data isn't `I32`.
    pub fn iter_indexed_i32(&self) -> impl Iterator<Item = (Vec<usize>, i32)> + '_ {
        self.iter_indices().map(move |idx| {
            let v = self.get_i32(&idx);
            (idx, v)
        })
    }

    /// Iterate `(index, bool value)` pairs in row-major order.
    /// Panics if the underlying data isn't `Bool`.
    pub fn iter_indexed_bool(&self) -> impl Iterator<Item = (Vec<usize>, bool)> + '_ {
        self.iter_indices().map(move |idx| {
            let v = self.get_bool(&idx);
            (idx, v)
        })
    }

    fn strided_index(&self, index: &[usize]) -> usize {
        let mut i = 0usize;
        for (d, idx) in index.iter().enumerate() {
            i += idx * self.strides[d];
        }
        i
    }

    /// Render the tensor as one or more 2-D tables.
    ///
    /// - rank 1: a single row.
    /// - rank 2: a table.
    /// - rank ≥ 3: one labeled table per combination of leading-dim indices
    ///   (the last two dims are always the row/col axes).
    pub fn pretty_print(&self) -> String {
        self.pretty_print_filtered(None)
    }

    /// Like [`pretty_print`], but only prints slices whose leading-dim indices
    /// match the filter. Wildcards (`DimFilter::Any`) iterate every value.
    ///
    /// `filter` accepts both `Vec<std::ops::Range<usize>>` and the canonical
    /// `TensorFilter` (the `CUBE_TEST_MODE` `M-K` syntax).
    pub fn pretty_print_slice<I>(&self, filter: I) -> String
    where
        I: IntoIterator,
        I::Item: Into<crate::DimFilter>,
    {
        let f: crate::TensorFilter = filter.into_iter().map(Into::into).collect();
        assert_eq!(
            f.len(),
            self.shape.rank(),
            "pretty_print_slice: filter rank ({}) must match tensor rank ({})",
            f.len(),
            self.shape.rank(),
        );
        self.pretty_print_filtered(Some(f))
    }

    fn pretty_print_filtered(&self, filter: Option<crate::TensorFilter>) -> String {
        let rank = self.shape.rank();
        match rank {
            0 => String::new(),
            1 => {
                // Single-row table; the only filter entry filters the col axis.
                let col_filter = filter.as_ref().and_then(|f| f.first());
                let cols = axis_indices(col_filter, self.shape[0]);
                let rows = vec![0usize];
                pretty_print_table(&rows, &cols, |_row_label, col_label| {
                    self.cell_string(self.strided_index(&[col_label]))
                })
            }
            2 => {
                // Last two filter entries (here filter[0], filter[1]) drive
                // row and col selection respectively.
                let row_filter = filter.as_ref().and_then(|f| f.first());
                let col_filter = filter.as_ref().and_then(|f| f.get(1));
                let rows = axis_indices(row_filter, self.shape[0]);
                let cols = axis_indices(col_filter, self.shape[1]);
                pretty_print_table(&rows, &cols, |row_label, col_label| {
                    self.cell_string(self.strided_index(&[row_label, col_label]))
                })
            }
            _ => self.print_higher_rank(filter.as_ref()),
        }
    }

    fn cell_string(&self, idx: usize) -> String {
        match &self.data {
            HostDataVec::I32(_) => self.data.get_i32(idx).to_string(),
            HostDataVec::F32(_) => format!("{:.3}", self.data.get_f32(idx)),
            HostDataVec::Bool(_) => self.data.get_bool(idx).to_string(),
        }
    }

    fn print_higher_rank(&self, filter: Option<&crate::TensorFilter>) -> String {
        let rank = self.shape.rank();
        let leading_dims = rank - 2;
        let row_dim = self.shape[rank - 2];
        let col_dim = self.shape[rank - 1];

        // Filter entries for the row and col axes (the last two), if any.
        let row_filter = filter.and_then(|f| f.get(rank - 2));
        let col_filter = filter.and_then(|f| f.get(rank - 1));
        let row_indices = axis_indices(row_filter, row_dim);
        let col_indices = axis_indices(col_filter, col_dim);

        let mut out = String::new();
        let mut leading = vec![0usize; leading_dims];

        // Iterate every leading-index combination, lexicographically.
        loop {
            let print_this = match filter {
                None => true,
                Some(f) => leading_indices_match(&leading, f),
            };

            if print_this {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("{}:\n", format_leading_label(&leading, rank)));

                let table = pretty_print_table(&row_indices, &col_indices, |row, col| {
                    let mut full = leading.clone();
                    full.push(row);
                    full.push(col);
                    self.cell_string(self.strided_index(&full))
                });
                out.push_str(&table);
            }

            // Increment the leading-index counter.
            if !increment_lex(&mut leading, &self.shape.as_slice()[..leading_dims]) {
                break;
            }
        }

        out
    }
}

pub fn pretty_print_zip(tensors: &[&HostData]) -> String {
    assert!(!tensors.is_empty(), "Need at least one tensor");

    let dims = tensors[0].shape.as_slice();

    for t in tensors {
        assert_eq!(t.shape.as_slice(), dims, "All tensors must have same shape");
    }

    let rank = tensors[0].shape.rank();

    let cell = |full: &[usize]| -> String {
        let mut parts = Vec::with_capacity(tensors.len());
        for t in tensors {
            let idx = t.strided_index(full);
            parts.push(t.cell_string(idx));
        }
        parts.join("/")
    };

    match rank {
        0 => String::new(),
        1 => {
            let cols: Vec<usize> = (0..dims[0]).collect();
            pretty_print_table(&[0], &cols, |_, col| cell(&[col]))
        }
        2 => {
            let rows: Vec<usize> = (0..dims[0]).collect();
            let cols: Vec<usize> = (0..dims[1]).collect();
            pretty_print_table(&rows, &cols, |row, col| cell(&[row, col]))
        }
        _ => {
            let leading_dims = rank - 2;
            let rows: Vec<usize> = (0..dims[rank - 2]).collect();
            let cols: Vec<usize> = (0..dims[rank - 1]).collect();
            let mut out = String::new();
            let mut leading = vec![0usize; leading_dims];
            loop {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("{}:\n", format_leading_label(&leading, rank)));
                let table = pretty_print_table(&rows, &cols, |row, col| {
                    let mut full = leading.clone();
                    full.push(row);
                    full.push(col);
                    cell(&full)
                });
                out.push_str(&table);

                if !increment_lex(&mut leading, &dims[..leading_dims]) {
                    break;
                }
            }
            out
        }
    }
}

/// Match leading indices against the leading slice of a tensor filter. The
/// trailing two filter entries (covering the row/col axes) are ignored — we
/// always print the full row × col table for the slices we keep.
fn leading_indices_match(leading: &[usize], filter: &crate::TensorFilter) -> bool {
    use crate::DimFilter::*;
    for (dim, &idx) in leading.iter().enumerate() {
        let f = filter.get(dim).unwrap_or(&Any);
        match f {
            Any => {}
            Exact(v) => {
                if idx != *v {
                    return false;
                }
            }
            Range { start, end } => {
                if idx < *start || idx > *end {
                    return false;
                }
            }
        }
    }
    true
}

/// Lexicographic increment over `idx[i] in 0..bounds[i]`. Returns `false` when
/// the counter has wrapped past the last position (i.e. iteration is done).
fn increment_lex(idx: &mut [usize], bounds: &[usize]) -> bool {
    if idx.is_empty() {
        return false;
    }
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] < bounds[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

/// Row-major index iterator. Yields every position in a tensor of the given
/// shape, lexicographically (last dim varies fastest). For a rank-0 tensor
/// (empty shape) the iterator yields a single empty index vector and stops.
struct IndexIter {
    shape: Vec<usize>,
    next: Option<Vec<usize>>,
}

impl IndexIter {
    fn new(shape: Vec<usize>) -> Self {
        // Empty dim → no indices.
        let next = if shape.contains(&0) {
            None
        } else {
            Some(vec![0; shape.len()])
        };
        Self { shape, next }
    }
}

impl Iterator for IndexIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next.clone()?;

        // Advance the counter for the next call. `increment_lex` returns
        // `false` when we've passed the last position; rank-0 also lands
        // here on the first call.
        let mut tentative = current.clone();
        if !increment_lex(&mut tentative, &self.shape) {
            self.next = None;
        } else {
            self.next = Some(tentative);
        }

        Some(current)
    }
}

fn format_leading_label(leading: &[usize], rank: usize) -> String {
    let mut parts: Vec<String> = leading.iter().map(|i| i.to_string()).collect();
    // Rows/cols are the last two dims — render as `*` so the label reads
    // `[i, j, *, *]`.
    for _ in 0..(rank - leading.len()) {
        parts.push("*".to_string());
    }
    format!("[{}]", parts.join(", "))
}

/// Resolve which indices along a single dim should be rendered, given a
/// per-dim filter entry. `None` means "render everything", which is the
/// default for unfiltered prints.
fn axis_indices(f: Option<&crate::DimFilter>, dim_size: usize) -> Vec<usize> {
    use crate::DimFilter::*;
    match f {
        None | Some(Any) => (0..dim_size).collect(),
        Some(Exact(v)) => {
            if *v < dim_size {
                vec![*v]
            } else {
                Vec::new()
            }
        }
        Some(Range { start, end }) => {
            if *start >= dim_size {
                Vec::new()
            } else {
                (*start..=(*end).min(dim_size.saturating_sub(1))).collect()
            }
        }
    }
}

fn pretty_print_table<F>(rows: &[usize], cols: &[usize], mut cell: F) -> String
where
    F: FnMut(usize, usize) -> String,
{
    let mut max_width = 0;

    for &r in rows {
        for &c in cols {
            max_width = max_width.max(cell(r, c).len());
        }
    }

    // Also account for the column-label width (so a tensor sliced down to
    // `[10-12]` renders header `10 11 12` without crowding).
    let label_width = cols.iter().map(|c| c.to_string().len()).max().unwrap_or(0);
    max_width = max_width.max(label_width).max(2);

    let row_label_width = rows
        .iter()
        .map(|r| r.to_string().len())
        .max()
        .unwrap_or(0)
        .max(3);

    let mut s = String::new();

    // header
    s.push_str(&format!("{:>width$} |", "", width = row_label_width));
    for &col in cols {
        s.push_str(&format!(" {:>width$}", col, width = max_width));
    }
    s.push('\n');

    // separator
    s.push_str(&"-".repeat(row_label_width + 1));
    s.push('+');
    for _ in cols {
        s.push_str(&"-".repeat(max_width + 1));
    }
    s.push('\n');

    // rows
    for &row in rows {
        s.push_str(&format!("{:>width$} |", row, width = row_label_width));

        for &col in cols {
            let value = cell(row, col);
            s.push_str(&format!(" {:>width$}", value, width = max_width));
        }

        s.push('\n');
    }

    s
}
