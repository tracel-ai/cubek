//! What the suites share: the elements a device runs, and the reads that move values from a buffer
//! to the host at an element.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::server::Handle;
use cubecl::tile::launch::Refusal;

pub const F32: ElemType = ElemType::Float(FloatKind::F32);
pub const F16: ElemType = ElemType::Float(FloatKind::F16);
pub const BF16: ElemType = ElemType::Float(FloatKind::BF16);

/// The elements every case runs at, those this device's runtime runs.
pub fn elements(client: &Client) -> Vec<ElemType> {
    [F32, F16, BF16]
        .into_iter()
        .filter(|elem| {
            client
                .properties()
                .features
                .type_usage(*elem)
                .is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer)
        })
        .collect()
}

/// `value` rounded to what `elem` holds.
pub fn round_to(value: f32, elem: ElemType) -> f32 {
    match elem {
        F16 => half::f16::from_f32(value).to_f32(),
        BF16 => half::bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

/// The buffer `handle` as `f32`, from the `elem` it holds.
pub fn read(client: &Client, handle: Handle, elem: ElemType) -> Vec<f32> {
    let bytes = client.read_one(handle).expect("the buffer reads back");
    match elem {
        F16 => (bytes.as_chunks::<2>().0.iter())
            .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        BF16 => (bytes.as_chunks::<2>().0.iter())
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        F32 => (bytes.as_chunks::<4>().0.iter())
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        other => panic!("{other:?} is not read back"),
    }
}

/// `values` uploaded as `elem` holds them: each rounded to it.
pub fn upload(client: &Client, values: &[f32], elem: ElemType) -> Handle {
    let bytes: Vec<u8> = match elem {
        F16 => (values.iter())
            .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
            .collect(),
        BF16 => (values.iter())
            .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
            .collect(),
        F32 => values.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => panic!("{other:?} is not uploaded"),
    };
    client.create_from_slice(&bytes)
}

/// What a launch left, or `None` where this device declines it by the very refusal admission
/// names: a device short of bindings serves less, and a test of what it serves skips the rest.
pub fn served<T>(
    what: &str,
    launched: Result<T, Refusal>,
    admitted: impl FnOnce() -> Result<(), Refusal>,
) -> Option<T> {
    match launched {
        Ok(outputs) => Some(outputs),
        Err(why) => {
            eprintln!("skipped, the device declines {what}: {why:?}");
            assert_eq!(admitted(), Err(why), "{what}");
            None
        }
    }
}
