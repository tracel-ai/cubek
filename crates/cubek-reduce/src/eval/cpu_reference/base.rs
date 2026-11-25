use crate::ReduceOperation;
use cubecl::tile::Geometry;
use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::{HostData, HostDataVec, Progress};

/// What a reduction writes, computed on the host.
#[derive(Debug)]
pub struct ReduceReference {
    pub output: HostData,
}

impl ReduceReference {
    /// What `operation` writes for every row of `input` along `axis`, laid as its output is: each
    /// row's [`ReduceOperation::reference_cells`] along `axis`, `progress` advanced by one a cell.
    ///
    /// # Panics
    ///
    /// Where `axis` is not an axis of `input`, or as [`ReduceOperation::reference_cells`] does.
    pub fn new(
        input: &HostData,
        axis: usize,
        operation: ReduceOperation,
        progress: Option<&Progress>,
    ) -> Self {
        let out_shape = output_shape_for(input.shape.as_slice(), axis, operation);
        let out = Geometry::row_major(&out_shape);
        let mut rows_shape = out_shape.clone();
        rows_shape[axis] = 1;
        // Every cell is written: a row writes each of its cells.
        let mut data = vec![0.0; out_shape.iter().product()];
        for_each_coord(&rows_shape, |row_coord| {
            let row: Vec<f32> = row_along(input, axis, row_coord).collect();
            let first: usize = row_coord
                .iter()
                .zip(out.strides())
                .map(|(c, s)| c * s)
                .sum();
            for (slot, cell) in operation.reference_cells(&row).into_iter().enumerate() {
                data[first + slot * out.strides()[axis]] = cell;
                bump(progress);
            }
        });
        Self {
            output: HostData {
                data: HostDataVec::F32(data),
                strides: Strides::new(out.strides()),
                shape: Shape::from(out_shape),
            },
        }
    }
}

/// The shape of what `operation` writes of an input of `shape` reduced along `axis`.
pub(crate) fn output_shape_for(
    shape: &[usize],
    axis: usize,
    operation: ReduceOperation,
) -> Vec<usize> {
    let mut out = shape.to_vec();
    out[axis] = operation.written_per_row();
    out
}

/// `visit` called on every coordinate of `shape`, the last axis fastest.
fn for_each_coord(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    let mut coord = vec![0usize; shape.len()];
    for linear in 0..shape.iter().product() {
        let mut rest = linear;
        for d in (0..shape.len()).rev() {
            coord[d] = rest % shape[d];
            rest /= shape[d];
        }
        visit(&coord);
    }
}

/// `progress` advanced by one output cell, where the caller tracks any.
fn bump(progress: Option<&Progress>) {
    if let Some(progress) = progress {
        progress.bump();
    }
}

/// The values of `input` along `axis` that reduce into the row at `row_coord`, in order.
fn row_along<'a>(
    input: &'a HostData,
    axis: usize,
    row_coord: &[usize],
) -> impl Iterator<Item = f32> + 'a {
    let mut coord = row_coord.to_vec();
    (0..input.shape[axis]).map(move |i| {
        coord[axis] = i;
        input.get_f32(&coord)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_row_is_reduced_along_its_axis() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let input = HostData::row_major(HostDataVec::F32(values), Shape::new([2, 2, 2]));
        let reference = ReduceReference::new(&input, 1, ReduceOperation::Sum, None);
        let HostDataVec::F32(written) = reference.output.data else {
            panic!("a reference is written in f32");
        };
        assert_eq!(written, vec![2.0, 4.0, 10.0, 12.0]);
    }

    /// Its rows lie a cell apart, each row's cells a row's count of them apart.
    #[test]
    fn each_cell_of_a_row_is_laid_along_its_axis() {
        let values = vec![0.0, 3.0, 1.0, 2.0, 2.0, 1.0];
        let input = HostData::row_major(HostDataVec::F32(values), Shape::new([3, 2]));
        let reference = ReduceReference::new(&input, 0, ReduceOperation::TopK { k: 2 }, None);
        let HostDataVec::F32(written) = reference.output.data else {
            panic!("a reference is written in f32");
        };
        assert_eq!(written, vec![2.0, 3.0, 1.0, 2.0]);
    }
}
