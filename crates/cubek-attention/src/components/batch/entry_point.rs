use crate::components::{batch::base::BatchAttention, global::GlobalAttentionConfig};
use crate::components::{
    batch::{BatchAttentionConfig, BatchAttentionFamily},
    stage::StageAttentionConfig,
};
use crate::{
    definition::AttentionBlueprint, definition::AttentionElems, definition::CubeCountInput,
    launch::AttentionArgs, launch::TensorKey, launch::TensorMask, launch::TensorOutput,
    launch::TensorQuery, launch::TensorValue,
};
use cubecl;
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

type Input<Args, QG, KG, VG, MSK> = <Args as AttentionArgs>::Input<QG, KG, VG, MSK>;
type Output<Args, OG> = <Args as AttentionArgs>::Output<OG>;

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
/// Launches the attention kernel
pub(crate) fn attention<
    Args: AttentionArgs,
    QG: Float,
    QGS: Size,
    KG: Float,
    KGS: Size,
    VG: Float,
    VGS: Size,
    MSK: Numeric,
    MSKS: Size,
    OG: Float,
    OGS: Size,
    BMMF: BatchAttentionFamily<Blueprint = AttentionBlueprint>,
>(
    inputs: &Input<Args, (QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS)>,
    output: &mut Output<Args, (OG, OGS)>,
    cube_count_args: CubeCountInput,
    #[comptime] blueprint: AttentionBlueprint,
    #[comptime] dtypes: AttentionElems,
    #[define(QG, KG, VG, MSK, OG)] _elem_types: [StorageType; 5],
    #[define(QGS, KGS, VGS, MSKS, OGS)] _sizes: [usize; 5],
) {
    let device_props = comptime::device_properties();
    let config = comptime!(BMMF::expand_config(&device_props, blueprint, &dtypes));
    if config.is_err() {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = config.unwrap();

    let mut state = Args::init_state(inputs, output);

    let query =
        TensorQuery::<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>::new(&state);
    let query = VirtualTensor::<QG, QGS>::new::<
        TensorQuery<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>,
    >(&query);

    let key =
        TensorKey::<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>::new(&state);
    let key = VirtualTensor::<KG, KGS>::new::<
        TensorKey<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>,
    >(&key);

    let value =
        TensorValue::<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>::new(&state);
    let value = VirtualTensor::<VG, VGS>::new::<
        TensorValue<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>,
    >(&value);

    let has_mask = Args::has_mask(&state);
    let mask = has_mask.map(|_| {
        let mask = TensorMask::<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>::new(
            &state,
        );
        VirtualTensor::<MSK, MSKS>::new::<
            TensorMask<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>,
        >(&mask)
    });

    let mut out =
        TensorOutput::<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>::new(
            &mut state,
        );
    let out = VirtualTensor::<OG, OGS, ReadWrite>::new::<
        TensorOutput<(QG, QGS), (KG, KGS), (VG, VGS), (MSK, MSKS), (OG, OGS), Args>,
    >(&mut out);

    let stage_config = config.global_config().stage_config();
    let key_stage = stage_config.key_smem_config();
    let value_stage = stage_config.value_smem_config();
    let out_stage = stage_config.out_smem_config();

    let define!(QT) = dtypes.query_tile;
    let define!(KS) = key_stage.dtype;
    let size!(KSS) = key_stage.vector_size as usize;
    let define!(VS) = value_stage.dtype;
    let size!(VSS) = value_stage.vector_size as usize;
    let define!(KVT) = dtypes.key_value_tile;
    let define!(SM) = dtypes.softmax_acc;
    let define!(SML) = dtypes.softmax_lhs;
    let define!(ACC) = dtypes.accumulator;
    let define!(OS) = out_stage.dtype;
    let size!(OSS) = out_stage.vector_size as usize;

    BMMF::Attention::<(
        (QG, QGS, QT),
        (KG, KGS, KS, KSS),
        (VG, VGS, VS, VSS),
        KVT,
        SM,
        SML,
        ACC,
        MSK,
        MSKS,
        (OG, OGS, OS, OSS),
    )>::execute(query, key, value, mask, out, cube_count_args, config);
}
