//! On-disk format for [`HostData`].
//!
//! Used by the tuner to compare the output of two metal runs (current commit
//! vs. trusted reference) without materializing both at the same time. Files
//! are intended to live in a temp directory — there's no forward-compatibility
//! story beyond a version byte that lets a reader reject an unknown format.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//!   offset  size   field
//!   ------  ----   -----
//!     0     4      magic = "CKHD"
//!     4     1      version (currently 1)
//!     5     1      dtype tag (0=F32, 1=I32, 2=Bool)
//!     6     4      rank
//!    10     8*rank shape
//!    +     8*rank  strides
//!    +     8       element count
//!    +     n       packed element bytes
//! ```
//!
//! Booleans are written as one byte each (0/1). The element count is the
//! length of the packed data array (not `shape.product()` — strides may make
//! the physical extent larger than the logical one).
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use cubecl::zspace::{Shape, Strides};

use crate::test_tensor::host_data::{HostData, HostDataVec};

const MAGIC: &[u8; 4] = b"CKHD";
const VERSION: u8 = 1;

const TAG_F32: u8 = 0;
const TAG_I32: u8 = 1;
const TAG_BOOL: u8 = 2;

/// Write `data` to `path` in the binary format documented at the module level.
///
/// Truncates any existing file. Returns the number of bytes written so callers
/// can surface a "wrote N MiB" log line.
pub fn write_host_data(path: &Path, data: &HostData) -> io::Result<u64> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    w.write_all(&[VERSION])?;

    let (tag, elem_count) = match &data.data {
        HostDataVec::F32(v) => (TAG_F32, v.len()),
        HostDataVec::I32(v) => (TAG_I32, v.len()),
        HostDataVec::Bool(v) => (TAG_BOOL, v.len()),
    };
    w.write_all(&[tag])?;

    let rank = data.shape.as_slice().len();
    w.write_all(&(rank as u32).to_le_bytes())?;
    for d in data.shape.as_slice() {
        w.write_all(&(*d as u64).to_le_bytes())?;
    }
    let strides_slice: &[usize] = &data.strides;
    if strides_slice.len() != rank {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "strides rank {} != shape rank {}",
                strides_slice.len(),
                rank,
            ),
        ));
    }
    for s in strides_slice {
        w.write_all(&(*s as u64).to_le_bytes())?;
    }
    w.write_all(&(elem_count as u64).to_le_bytes())?;

    match &data.data {
        HostDataVec::F32(v) => w.write_all(bytemuck::cast_slice(v))?,
        HostDataVec::I32(v) => w.write_all(bytemuck::cast_slice(v))?,
        HostDataVec::Bool(v) => {
            // One byte per bool — keeps reads alignment-free and rare enough
            // not to be worth bit-packing.
            for b in v {
                w.write_all(&[u8::from(*b)])?;
            }
        }
    }

    w.flush()?;
    Ok(w.into_inner()
        .map_err(|e| e.into_error())?
        .metadata()?
        .len())
}

/// Read a [`HostData`] previously produced by [`write_host_data`].
///
/// Errors with `InvalidData` for any header/version/tag mismatch — these
/// usually mean the file came from a different cubek version and should be
/// regenerated.
pub fn read_host_data(path: &Path) -> io::Result<HostData> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("wrong magic — file is not a HostData blob"));
    }
    let version = read_u8(&mut r)?;
    if version != VERSION {
        return Err(invalid(format!(
            "unsupported HostData file version: {version} (expected {VERSION})"
        )));
    }
    let tag = read_u8(&mut r)?;
    let rank = read_u32(&mut r)? as usize;

    let mut shape_dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape_dims.push(read_u64(&mut r)? as usize);
    }
    let mut stride_dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        stride_dims.push(read_u64(&mut r)? as usize);
    }
    let elem_count = read_u64(&mut r)? as usize;

    let data = match tag {
        TAG_F32 => {
            let mut buf = vec![0u8; elem_count * std::mem::size_of::<f32>()];
            r.read_exact(&mut buf)?;
            // Guaranteed-aligned re-cast: build the Vec<f32> from the byte
            // chunks rather than transmuting the buffer in place.
            let mut v = Vec::with_capacity(elem_count);
            for chunk in buf.chunks_exact(4) {
                v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            HostDataVec::F32(v)
        }
        TAG_I32 => {
            let mut buf = vec![0u8; elem_count * std::mem::size_of::<i32>()];
            r.read_exact(&mut buf)?;
            let mut v = Vec::with_capacity(elem_count);
            for chunk in buf.chunks_exact(4) {
                v.push(i32::from_le_bytes(chunk.try_into().unwrap()));
            }
            HostDataVec::I32(v)
        }
        TAG_BOOL => {
            let mut buf = vec![0u8; elem_count];
            r.read_exact(&mut buf)?;
            HostDataVec::Bool(buf.into_iter().map(|b| b != 0).collect())
        }
        other => return Err(invalid(format!("unknown HostData dtype tag: {other}"))),
    };

    Ok(HostData {
        data,
        shape: Shape::from(shape_dims),
        strides: Strides::new(&stride_dims),
    })
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn invalid<E: Into<String>>(msg: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(label: &str, data: HostData) {
        let dir =
            std::env::temp_dir().join(format!("cubek-test-utils-iotest-{}", std::process::id(),));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("blob-{label}.bin"));
        write_host_data(&path, &data).unwrap();
        let read_back = read_host_data(&path).unwrap();
        assert_eq!(data.shape, read_back.shape);
        assert_eq!(data.strides, read_back.strides);
        match (&data.data, &read_back.data) {
            (HostDataVec::F32(a), HostDataVec::F32(b)) => assert_eq!(a, b),
            (HostDataVec::I32(a), HostDataVec::I32(b)) => assert_eq!(a, b),
            (HostDataVec::Bool(a), HostDataVec::Bool(b)) => assert_eq!(a, b),
            _ => panic!("dtype mismatch on round-trip"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn round_trip_f32() {
        round_trip(
            "f32",
            HostData {
                data: HostDataVec::F32(vec![1.0, -2.0, std::f32::consts::PI, 0.5, 0.0]),
                shape: Shape::from(vec![5]),
                strides: Strides::new(&[1]),
            },
        );
    }

    #[test]
    fn round_trip_i32_2d() {
        round_trip(
            "i32",
            HostData {
                data: HostDataVec::I32(vec![1, 2, 3, 4, 5, 6]),
                shape: Shape::from(vec![2, 3]),
                strides: Strides::new(&[3, 1]),
            },
        );
    }

    #[test]
    fn round_trip_bool() {
        round_trip(
            "bool",
            HostData {
                data: HostDataVec::Bool(vec![true, false, true, true, false]),
                shape: Shape::from(vec![5]),
                strides: Strides::new(&[1]),
            },
        );
    }
}
