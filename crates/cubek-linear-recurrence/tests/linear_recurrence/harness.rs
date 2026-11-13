//! What the suites share: the elements a device runs, and the reads that move values from a buffer
//! to the host at an element.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;

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

/// A tensor of `shape` and `strides` over `buffer` uploaded as `elem`, its view starting `start`
/// elements in.
pub fn view(
    client: &Client,
    buffer: &[f32],
    start: usize,
    shape: &[usize],
    strides: &[usize],
    elem: ElemType,
) -> (Handle, PlainTensor) {
    let raw = upload(client, buffer, elem);
    let handle = raw.clone().offset_start((start * elem.size()) as u64);
    let tensor = PlainTensor::of_handle(TensorHandle::from_metadata(
        handle,
        Metadata::new(shape.to_vec(), strides.to_vec()),
        elem,
    ));
    (raw, tensor)
}

pub fn table(client: &Client, entries: Vec<u32>) -> PlainTensor {
    PlainTensor::of_u32s(client, &[entries.len()], &entries)
}

/// `entries` as a column of a `[rows, 2]` table, the other column holding `other`.
pub fn strided_table(client: &Client, entries: Vec<u32>, other: u32) -> PlainTensor {
    let rows = entries.len();
    let values: Vec<u32> = entries.iter().flat_map(|&entry| [entry, other]).collect();
    PlainTensor::of_u32s(client, &[rows, 2], &values).viewed(Metadata::new(vec![rows], vec![2]))
}

/// `len` values of a sine, rounded to `elem`.
pub fn wave(len: usize, seed: f32, scale: f32, elem: ElemType) -> Vec<f32> {
    (0..len)
        .map(|i| round_to(scale * (i as f32 * 0.37 + seed).sin(), elem))
        .collect()
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

/// The channels convolved and rounded as the kernel rounds them: the newest tap of `channel` reads
/// `newest(channel)`, tap `j` reads `past(j, channel)` or nothing where none is held, and `bias`
/// joins the sum before its first rounding.
pub fn convolve_channels(
    filters: &[f32],
    bias: Option<&[f32]>,
    taps: usize,
    newest: impl Fn(usize) -> f32,
    past: impl Fn(usize, usize) -> Option<f32>,
    elem: ElemType,
) -> Vec<f64> {
    let round = |x: f64| round_to(x as f32, elem) as f64;
    (0..filters.len() / taps)
        .map(|channel| {
            let tap = |j: usize| filters[channel * taps + j] as f64;
            let mut sum = newest(channel) as f64 * tap(taps - 1);
            for j in 0..taps - 1 {
                sum += past(j, channel).map_or(0.0, |x| x as f64 * tap(j));
            }
            sum += bias.map_or(0.0, |bias| bias[channel] as f64);
            let mixed = round(sum);
            round(mixed / (1.0 + (-mixed).exp()))
        })
        .collect()
}

/// How far an output may fall from the reference's, relative to one plus its magnitude: a sum an
/// ulp of the element apart rounds to neighbouring values.
pub fn tolerance(elem: ElemType) -> f64 {
    match elem {
        F32 => 1e-5,
        F16 => 6e-3,
        BF16 => 2e-2,
        _ => unreachable!("the conv is verified at f16, bf16 and f32"),
    }
}

pub fn compare(got: &[f32], want: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: the element count");
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got as f64 - want).abs() <= tolerance * (1.0 + want.abs()),
            "{what}: element {i} is {got}, the reference {want}"
        );
    }
}
