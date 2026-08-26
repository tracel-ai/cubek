use cubecl::{
    ir::ElemType,
    throughput::{ThroughputKey, ThroughputMode},
    tune::Work,
};

use crate::definition::{
    InterpolateBackwardProblem, InterpolateForwardProblem, InterpolateMode, InterpolateProblem,
    mode_properties,
};

/// Minimal representation of interpolate cost dependencies: the shapes the resampling maps
/// between and the element type its traffic is counted in.
///
/// The problem already carries nothing but extents and a mode, so it is held rather than
/// unpacked into a parallel set of fields.
#[derive(Debug, Clone)]
pub struct InterpolateCost {
    /// The resampling being costed.
    pub problem: InterpolateProblem,
    /// Element type of the tensors read and written.
    pub dtype: ElemType,
}

impl InterpolateCost {
    pub fn new(problem: InterpolateProblem, dtype: ElemType) -> Self {
        Self { problem, dtype }
    }

    /// Calculates arithmetic operations and compulsory memory traffic for the resampling.
    ///
    /// `compute_ops` counts the arithmetic the reference filter emits per output element,
    /// selects and comparisons included, so a mode that spends its time choosing between
    /// branches is not scored as free.
    pub fn work(&self) -> Work {
        match &self.problem {
            InterpolateProblem::Forward(prob) => self.forward_work(prob),
            InterpolateProblem::Backward(prob) => self.backward_work(prob),
        }
    }

    /// Throughput key for the filter's arithmetic, which runs at the element type the
    /// tensors are read and written in.
    pub fn compute_key(&self) -> ThroughputKey {
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect { dtype: self.dtype },
        }
    }

    fn forward_work(&self, prob: &InterpolateForwardProblem) -> Work {
        let planes = prob.batch * prob.channels;
        let outputs = planes * prob.output_height * prob.output_width;
        let taps = mode_properties(prob.options.mode).taps;

        // Each output position pulls a `halo`-wide window, so the windows together span
        // `halo` times the output extent. An upsample re-reads rows the previous window
        // already covered, which the input extent caps; a downsample coarse enough to
        // outrun the window skips the rows no window reaches, which the product caps.
        let rows_read = prob.input_height.min(prob.output_height * taps);
        let cols_read = prob.input_width.min(prob.output_width * taps);

        Work {
            compute_ops: outputs * ops_per_output(prob.options.mode),
            bytes: (planes * rows_read * cols_read + outputs) * self.dtype.size(),
        }
    }

    fn backward_work(&self, prob: &InterpolateBackwardProblem) -> Work {
        let [batch, grad_height, grad_width, channels] = prob.out_grad_shape;
        let planes = batch * channels;
        let grads = planes * grad_height * grad_width;
        let outputs = planes * prob.input_size[0] * prob.input_size[1];

        Work {
            // The gather windows tile the gradient plane: consecutive output positions
            // start where the previous one ended, so the window sizes telescope to the
            // gradient extent and every gradient element is summed exactly once,
            // whichever direction the forward pass resampled in.
            compute_ops: grads,
            bytes: (grads + outputs) * self.dtype.size(),
        }
    }
}

/// Arithmetic operations the filter emits for one output element of one channel.
///
/// Counted from `compute_value`, which is the reference the kernels implement.
fn ops_per_output(mode: InterpolateMode) -> usize {
    // Nearest resolves no weights and contracts nothing: it clamps the row and the column
    // into range, a max and a min each, and reads the single element it landed on.
    if let InterpolateMode::Nearest(_) = mode {
        return 4;
    }

    let taps = mode_properties(mode).taps;
    // One weight per tap on each of the two axes.
    let weights = 2 * taps * weight_ops(mode);
    // Every tap multiplies its column weight in and adds the product into the row.
    let contraction = taps * taps * 2;
    // Every row multiplies its row weight in and adds the product into the total.
    let rows = taps * 2;

    weights + contraction + rows + bound_check_ops(mode)
}

/// What renormalizing against the in-bounds weight adds on top of the plain contraction.
///
/// Zero for the modes whose weights vanish at the window edge, which need no such guard.
fn bound_check_ops(mode: InterpolateMode) -> usize {
    if !mode_properties(mode).renormalizes {
        return 0;
    }

    let taps = mode_properties(mode).taps;
    // Per tap: four comparisons and the three and operations that fold them into one flag, the two
    // selects that zero an out-of-bounds value and its weight, and the extra add that
    // accumulates the weight alongside the value.
    let per_tap = 7 + 2 + 1;
    // Per row: the row weight multiplied into the row's weight sum, and that added in.
    let per_row = 2;
    // Once: clamping the total weight away from zero, and the divide it guards.
    taps * taps * per_tap + taps * per_row + 2
}

/// Arithmetic operations one `compute_weight` call emits.
///
/// A transcendental counts as one operation, as everything else here does: this is an
/// instruction count, not a cycle estimate, so Lanczos3's two sines are understated by
/// whatever their hardware cost is.
fn weight_ops(mode: InterpolateMode) -> usize {
    match mode {
        // The weight is a constant one, folded away rather than computed.
        InterpolateMode::Nearest(_) => 0,
        // An abs, the comparison against one, the subtraction, and the select.
        InterpolateMode::Bilinear => 4,
        // An abs and the two multiplies that raise it to the second and third powers, four
        // operations for the inner cubic, six for the outer one, then the two comparisons
        // and two selects that pick between them and zero.
        InterpolateMode::Bicubic => 3 + 4 + 6 + 4,
        // An abs, the scale by pi, the squared denominator and its third; the comparison
        // and select guarding a zero denominator; the two sines, the argument's third, the
        // product and the divide; then the two comparisons and two selects around them.
        InterpolateMode::Lanczos3 => 4 + 2 + 5 + 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;

    use crate::definition::{InterpolateOptions, NearestMode};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    fn forward(
        input: [usize; 4],
        output: [usize; 2],
        mode: InterpolateMode,
    ) -> InterpolateForwardProblem {
        InterpolateForwardProblem::from_input_output_shapes(
            &input.into(),
            &output,
            InterpolateOptions::new(mode),
        )
    }

    fn work(problem: InterpolateProblem) -> Work {
        InterpolateCost::new(problem, F32).work()
    }

    fn nearest() -> InterpolateMode {
        InterpolateMode::Nearest(NearestMode::Floor)
    }

    #[test]
    fn nearest_downsample_reads_only_the_rows_it_lands_on() {
        let problem =
            InterpolateProblem::Forward(forward([1, 2048, 2048, 3], [1024, 1024], nearest()));

        // One tap per output, so half the rows and half the columns are never touched:
        // the read is the output's own extent, not the input's.
        let reads = 3 * 1024 * 1024;
        let writes = 3 * 1024 * 1024;
        assert_eq!(work(problem).bytes, (reads + writes) * F32.size());
    }

    #[test]
    fn a_two_tap_window_covers_the_whole_input_at_the_same_scale() {
        let problem = InterpolateProblem::Forward(forward(
            [1, 2048, 2048, 3],
            [1024, 1024],
            InterpolateMode::Bilinear,
        ));

        // Halving with a two-wide window leaves no gap between consecutive windows, so
        // every input element is read despite the output being a quarter of the size.
        let reads = 3 * 2048 * 2048;
        let writes = 3 * 1024 * 1024;
        assert_eq!(work(problem).bytes, (reads + writes) * F32.size());
    }

    #[test]
    fn an_upsample_never_counts_the_input_more_than_once() {
        let problem = InterpolateProblem::Forward(forward(
            [1, 2048, 2048, 3],
            [4096, 4096],
            InterpolateMode::Lanczos3,
        ));

        // Six taps over a doubled output would span the input twelve times over; what a
        // cache actually has to move is the input, once.
        let reads = 3 * 2048 * 2048;
        let writes = 3 * 4096 * 4096;
        assert_eq!(work(problem).bytes, (reads + writes) * F32.size());
    }

    #[test]
    fn wider_filters_cost_more_per_output() {
        let ops = |mode| ops_per_output(mode);

        assert!(ops(nearest()) < ops(InterpolateMode::Bilinear));
        assert!(ops(InterpolateMode::Bilinear) < ops(InterpolateMode::Bicubic));
        assert!(ops(InterpolateMode::Bicubic) < ops(InterpolateMode::Lanczos3));
    }

    #[test]
    fn only_the_renormalizing_filter_pays_for_bounds() {
        assert_eq!(bound_check_ops(nearest()), 0);
        assert_eq!(bound_check_ops(InterpolateMode::Bilinear), 0);
        assert_eq!(bound_check_ops(InterpolateMode::Bicubic), 0);
        assert!(bound_check_ops(InterpolateMode::Lanczos3) > 0);
    }

    #[test]
    fn the_backward_gather_visits_every_gradient_once() {
        let grads = 4096 * 4096 * 3;
        let outputs = 2048 * 2048 * 3;

        let downsampling_gradient =
            work(InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 2048],
                out_grad_shape: [1, 4096, 4096, 3],
                options: InterpolateOptions::new(nearest()),
            }));

        assert_eq!(downsampling_gradient.compute_ops, grads);
        assert_eq!(downsampling_gradient.bytes, (grads + outputs) * F32.size());

        // The windows telescope either way, so scattering into a larger gradient still
        // touches each of its elements exactly once.
        let upsampling_gradient = work(InterpolateProblem::Backward(InterpolateBackwardProblem {
            input_size: [4096, 4096],
            out_grad_shape: [1, 2048, 2048, 3],
            options: InterpolateOptions::new(nearest()),
        }));

        assert_eq!(upsampling_gradient.compute_ops, outputs);
        assert_eq!(upsampling_gradient.bytes, (grads + outputs) * F32.size());
    }
}
