//! A float routine of the matmul family at one of its settings ([`MatmulRoutineCandidate`]), as a
//! caller that measures the routines lists, launches and identifies it.

use crate::kernels::plan::MatmulCall;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::{
    ListedFor, MatmulBlueprint, MatmulRoutine, MatmulRoutinePrepared, MatmulRoutineProblem,
    MatmulTensors, MatmulTuneKey,
};
use cubecl::std::tensor::TensorHandle;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::{Routine, RoutineCandidate};

/// A float routine at one of its settings, prepared for the device it was listed for: what a
/// caller that measures the routines races, each problem prepared once by every clone.
///
/// These routines snap no cut to a divisor of the problem (a cube overhanging it is masked), so
/// the prepared launch's cube box is the block a setting asks for
/// ([`MatmulRoutinePrepared::cube_block`]).
#[derive(Debug, Clone)]
pub struct MatmulRoutineCandidate<R: Routine> {
    candidate: RoutineCandidate<R>,
}

impl<R: Routine> MatmulRoutineCandidate<R> {
    /// The setting this candidate launches the routine at, which names it.
    pub fn settings(&self) -> &R::Settings {
        self.candidate.settings()
    }
}

impl<R: Routine + Copy> MatmulRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        RoutineCandidate::list(routine, hw)
            .into_iter()
            .map(|candidate| Self { candidate })
            .collect()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.candidate.routine()
    }

    /// `call`'s product by this candidate's routine at its setting: into its destination or into
    /// an output the launch allocates, on the device the candidate was listed for, each problem
    /// prepared once.
    ///
    /// # Errors
    ///
    /// A packed input, which a float routine's settings are not listed for, what the tensors
    /// decline ([`MatmulTensors::writing_into`], [`MatmulTensors::new`]), and why the routine
    /// cannot run the product at this setting.
    pub fn launch<M: MatmulMicrokernel, SM>(&self, call: MatmulCall) -> Result<TensorHandle, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        ListedFor::Float
            .admits(call.product())
            .map_err(|decline| format!("{decline:?}"))?;
        let settings = self.settings();
        let destination = call.destination;
        let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
        let tensors = match &destination {
            Some(destination) => MatmulTensors::writing_into(
                &call.client,
                lhs,
                rhs,
                PlainTensor::of_handle(destination.clone()),
                R::ROWS_KIND,
            ),
            None => MatmulTensors::new(&call.client, lhs, rhs, R::ROWS_KIND),
        };
        let launched = tensors.and_then(|tensors| {
            let (blueprint, grid) = self.candidate.prepare(&tensors.problem())?;
            self.routine()
                .dispatch_into_buffer(&tensors, blueprint, grid)?;
            Ok(match destination {
                Some(destination) => destination,
                None => tensors
                    .into_out()
                    .expect("a launch allocating its output hands it back"),
            })
        });
        launched.map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
    }
}

impl<R, M: MatmulMicrokernel, SM> MatmulRoutineCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// key names a packed input, which a float routine does not read, or the routine declines the
    /// problem on the device it was listed for.
    ///
    /// # Guarantees
    ///
    /// Where it is `None`, [`launch`](Self::launch) declines the key's products before compiling
    /// anything.
    pub fn prepared(&self, key: &MatmulTuneKey) -> Option<MatmulRoutinePrepared<M, SM>> {
        if key.is_packed() {
            return None;
        }
        let problem = MatmulRoutineProblem::from(&key.shape());
        let (blueprint, cube_dim) = self.candidate.launch_of(&problem)?;
        Some(MatmulRoutinePrepared::new(blueprint, cube_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{
        CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, MatmulBatching, OperandKey,
        PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
    };
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::Dimension;
    use cubek_std::devices;
    use cubek_std::layout::BindingOrder;
    use cubek_std::matmul_dimensions::{K, N};

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the `16×16×16` `f16` instruction summed in `f32`.
    fn offering(hw: Hardware) -> Hardware {
        devices::offering_cmma(
            hw,
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: F32,
                m: 16,
                n: 16,
                k: 16,
            }],
        )
    }

    /// A key over an `f16` `m × k @ k × n` product of `batch` batches, row-major.
    fn key(batch: usize, m: usize, n: usize, k: usize) -> MatmulTuneKey {
        MatmulTuneKey::new(
            F16,
            k,
            n,
            m,
            batch,
            OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
            OperandKey::float(Some(N), MatmulBatching::new(&[batch])),
        )
    }

    /// A key over an `f16` product of `batch` batches of `m` rows against a `k × n` rhs whose
    /// buffer runs along `rhs`.
    fn rows_key(batch: usize, m: usize, n: usize, k: usize, rhs: Dimension) -> MatmulTuneKey {
        MatmulTuneKey::new(
            F16,
            k,
            n,
            m,
            batch,
            OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
            OperandKey::float(Some(rhs), MatmulBatching::Shared),
        )
    }

    #[test]
    fn a_candidate_per_setting_the_default_first() {
        let hw = offering(devices::strix_halo());
        let listed: Vec<_> = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw)
            .iter()
            .map(|candidate| *candidate.settings())
            .collect();
        assert_eq!(listed, CubeStagedMatmulRoutine::settings(&hw));
        let listed: Vec<_> = MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw)
            .iter()
            .map(|candidate| *candidate.settings())
            .collect();
        assert_eq!(listed, PlaneShuffleMatmulRoutine::settings(&hw));
    }

    /// A col rhs is read where it lies, so the key that names one prepares the launch its
    /// tensors will: the cube routine's cmma fragment and the unit routine's register block both
    /// bind it in its buffer's order, in vectors along `K`.
    #[test]
    fn a_col_rhs_key_prepares_the_launch_its_tensors_will() {
        let col = MatmulTuneKey::new(
            F16,
            256,
            256,
            256,
            1,
            OperandKey::float(Some(K), MatmulBatching::Shared),
            OperandKey::float(Some(K), MatmulBatching::Shared),
        );
        let hw = offering(devices::strix_halo());
        let cube = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
        let prepared = cube.prepared(&col).expect("runs here");
        let in_buffer_order = prepared.blueprint();
        assert_eq!(
            in_buffer_order.microkernel.orders().rhs,
            BindingOrder::InBufferOrder
        );
        assert!(in_buffer_order.widths.rhs > 1);

        let unit = MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw).swap_remove(0);
        let prepared = unit.prepared(&col).expect("runs here");
        let merged = prepared.blueprint();
        assert_eq!(merged.microkernel.orders().rhs, BindingOrder::InBufferOrder);
        assert!(merged.widths.rhs > 1);
    }

    /// On devices a dev machine cannot show: the unit routine, the family's floor, prepares
    /// everywhere. A CPU holds fewer units a cube than one cube routine's cube of planes, so there
    /// the cube routine declines, and planes of no single width hold no fragment.
    #[test]
    fn every_device_keeps_the_floor() {
        let key = key(1, 2048, 2048, 2048);
        for hw in [
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for unit in MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw) {
                assert!(unit.prepared(&key).is_some(), "{}", unit.settings());
            }
        }
        let cpu = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &devices::cpu_12())
            .swap_remove(0);
        assert!(cpu.prepared(&key).is_none());
        let rdna = offering(devices::rdna_range());
        let lead = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &rdna).swap_remove(0);
        assert!(lead.prepared(&key).is_none());
    }

    /// A batched key prepares a batched launch: the batch is distributed to cubes, not dropped.
    #[test]
    fn a_batched_key_distributes_its_batch() {
        let problem = MatmulRoutineProblem::from(&key(8, 256, 256, 256).shape());
        assert_eq!(problem.batches, [8]);
        let hw = offering(devices::strix_halo());
        let lead = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
        let single = lead.prepared(&key(1, 256, 256, 256)).expect("runs here");
        let batched = lead.prepared(&key(8, 256, 256, 256)).expect("runs here");
        assert_ne!(
            single.blueprint().partitioning,
            batched.blueprint().partitioning
        );
    }

    /// A key's batches are read as rows against one rhs, as the whole-rows tensors merge an
    /// lhs's leading dims.
    #[test]
    fn a_keys_batches_are_read_as_rows() {
        let plane = MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &devices::strix_halo())
            .swap_remove(0);
        let prepared = plane
            .prepared(&rows_key(2, 4, 2048, 2048, K))
            .expect("eight rows against a col rhs run here");
        assert_eq!(prepared.blueprint().microkernel.rows, 8);
    }

    /// A shuffle routine prepares no launch for a packed rhs, which a float setting does not read,
    /// for more rows than a work item holds, or on a device that commits to no plane width.
    #[test]
    fn a_shuffle_prepares_nothing_it_cannot_run() {
        use crate::kernels::select::problem::OperandKind;
        use cubecl::quant::scheme::{BlockScale, QuantScheme, QuantStore, QuantValue, ScaleDtype};
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q4S)
            .with_levels([BlockScale::per_block([32, 1], ScaleDtype::F16)])
            .with_store(QuantStore::PackedU32(1));
        let packed = MatmulTuneKey::new(
            F16,
            2048,
            2048,
            1,
            1,
            OperandKey::float(Some(K), MatmulBatching::Shared),
            OperandKey {
                kind: OperandKind::Quantized(scheme),
                ..OperandKey::float(Some(K), MatmulBatching::Shared)
            },
        );
        let cases = [
            (devices::strix_halo(), packed),
            (devices::strix_halo(), rows_key(1, 16, 2048, 2048, K)),
            (devices::wgpu_no_subgroups(), rows_key(1, 1, 2048, 2048, K)),
        ];
        for (hw, key) in cases {
            for plane in MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw) {
                assert!(plane.prepared(&key).is_none(), "{}", plane.settings());
            }
            for cube in MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, &hw) {
                assert!(cube.prepared(&key).is_none(), "{}", cube.settings());
            }
        }
    }
}
