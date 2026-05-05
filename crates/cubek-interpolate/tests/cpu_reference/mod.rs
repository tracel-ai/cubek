pub mod nearest;
use cubecl::zspace::Strides;

pub(crate) fn for_each_output_coord(output_shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let rank = output_shape.len();
    if rank == 0 {
        f(0, &[]);
        return;
    }
    let num: usize = output_shape.iter().product();
    let mut coord = vec![0usize; rank];
    for linear in 0..num {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % output_shape[d];
            rem /= output_shape[d];
        }
        f(linear, &coord);
    }
}

pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let n = shape.len();
    if n == 0 {
        return Strides::new(&[] as &[usize]);
    }
    let mut s = vec![0usize; n];
    s[n - 1] = 1;
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    Strides::new(&s)
}
