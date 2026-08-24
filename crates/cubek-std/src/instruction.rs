//! Choosing the accelerated instruction shape for a problem.

use crate::{MatmulProblemSize, TileSize};

/// The instruction shape to contract with, chosen from the problem's aspect ratio:
/// `32x8x16` for a tall problem, `8x32x16` for a wide one, `16x16x16` for a balanced
/// one, then `8x8x8`, then whatever the device supports at all.
///
/// `tm`/`tn`/`tk` force an axis when `Some`, and a forced axis filters every
/// candidate including the fallback — so forcing a shape the device cannot serve
/// yields `None` rather than silently picking another.
///
/// This is a heuristic over *shapes*, and takes nothing else: `is_supported` answers
/// whether one `m×n×k` may be used, `fallback_sizes` lists what to fall back to.
/// Element types, a compute client, a device — whatever a caller needs to answer
/// that, it closes over. Which is what lets the operand live beside the size types it
/// returns rather than inside a routine crate: matmul, convolution and attention all
/// pick an instruction the same way, and none of them needs the others to do it, nor
/// does a selector that plans against a hardware snapshot with no runtime in hand.
///
/// "Supported" is the caller's word, not the device's. A selector that also requires
/// the shape to divide its problem folds that into `is_supported` and into what
/// `fallback_sizes` lists, and the preference order then applies to exactly the
/// shapes it would accept — the operand never returns one the caller would reject.
///
/// `None` means no candidate was both supported and consistent with the forced
/// axes. Callers with a richer error map it to their own.
pub fn find_instruction_size<IsSupported, FallbackSizes>(
    problem_size: MatmulProblemSize,
    (tm, tn, tk): (Option<u32>, Option<u32>, Option<u32>),
    is_supported: IsSupported,
    fallback_sizes: FallbackSizes,
) -> Option<TileSize>
where
    IsSupported: Fn(u32, u32, u32) -> bool,
    FallbackSizes: Fn() -> Vec<TileSize>,
{
    let matches_forced = |m: u32, n: u32, k: u32| {
        tm.is_none_or(|v| m == v) && tn.is_none_or(|v| n == v) && tk.is_none_or(|v| k == v)
    };

    let try_candidate = |m: u32, n: u32, k: u32| {
        (is_supported(m, n, k) && matches_forced(m, n, k)).then(|| TileSize::from((m, n, k)))
    };

    let (m, n) = (problem_size.m, problem_size.n);

    if m >= 4 * n
        && let Some(ts) = try_candidate(32, 8, 16)
    {
        return Some(ts);
    }

    if n >= 4 * m
        && let Some(ts) = try_candidate(8, 32, 16)
    {
        return Some(ts);
    }

    if let Some(ts) = try_candidate(16, 16, 16) {
        return Some(ts);
    }

    if let Some(ts) = try_candidate(8, 8, 8) {
        return Some(ts);
    }

    fallback_sizes()
        .into_iter()
        .find(|ts| matches_forced(ts.m, ts.n, ts.k))
}
