use cubecl::{
    CubeElement, TestRuntime,
    client::ComputeClient,
    prelude::CubePrimitive,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
};

use crate::test_tensor::cast::copy_casted;

#[derive(Debug)]
pub struct HostData {
    pub data: HostDataVec,
    pub shape: Shape,
    pub strides: Strides,
}

#[derive(Eq, PartialEq, PartialOrd)]
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
}

impl HostData {
    pub fn from_tensor_handle(
        client: &ComputeClient<TestRuntime>,
        tensor_handle: TensorHandle<TestRuntime>,
        host_data_type: HostDataType,
    ) -> Self {
        let shape = tensor_handle.shape().clone();
        let strides = tensor_handle.strides().clone();

        let data = match host_data_type {
            HostDataType::F32 => {
                let handle = copy_casted(client, tensor_handle, f32::as_type_native_unchecked());
                let data = f32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::F32(data)
            }
            HostDataType::I32 => {
                let handle = copy_casted(client, tensor_handle, i32::as_type_native_unchecked());
                let data = i32::from_bytes(
                    &client.read_one_unchecked_tensor(handle.into_copy_descriptor()),
                )
                .to_owned();

                HostDataVec::I32(data)
            }
            HostDataType::Bool => {
                let handle = copy_casted(client, tensor_handle, u32::as_type_native_unchecked());
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

    fn strided_index(&self, index: &[usize]) -> usize {
        let mut i = 0usize;
        for (d, idx) in index.iter().enumerate() {
            i += idx * self.strides[d];
        }
        i
    }

    pub fn pretty_print(&self) -> String {
        assert!(
            self.shape.rank() == 2,
            "pretty_print only supports 2D tensors"
        );

        let [rows, cols] = self.shape.dims();

        pretty_print_table(rows, cols, |row, col| {
            let idx = self.strided_index(&[row, col]);

            match &self.data {
                HostDataVec::I32(_) => self.data.get_i32(idx).to_string(),
                HostDataVec::F32(_) => format!("{:.3}", self.data.get_f32(idx)),
                HostDataVec::Bool(_) => self.data.get_bool(idx).to_string(),
            }
        })
    }
}

pub fn pretty_print_zip(tensors: &[&HostData]) -> String {
    assert!(!tensors.is_empty(), "Need at least one tensor");

    let [rows, cols] = tensors[0].shape.dims();

    for t in tensors {
        assert_eq!(
            t.shape.dims(),
            [rows, cols],
            "All tensors must have same shape"
        );
    }

    pretty_print_table(rows, cols, |row, col| {
        let mut parts = Vec::with_capacity(tensors.len());

        for t in tensors {
            let idx = t.strided_index(&[row, col]);

            let val = match &t.data {
                HostDataVec::I32(_) => t.data.get_i32(idx).to_string(),
                HostDataVec::F32(_) => format!("{:.3}", t.data.get_f32(idx)),
                HostDataVec::Bool(_) => t.data.get_bool(idx).to_string(),
            };

            parts.push(val);
        }

        parts.join("/")
    })
}

fn pretty_print_table<F>(rows: usize, cols: usize, mut cell: F) -> String
where
    F: FnMut(usize, usize) -> String,
{
    let mut max_width = 0;

    for r in 0..rows {
        for c in 0..cols {
            max_width = max_width.max(cell(r, c).len());
        }
    }

    max_width = max_width.max(2);

    let mut s = String::new();

    // header
    s.push_str(&format!("{:>3} |", ""));
    for col in 0..cols {
        s.push_str(&format!(" {:>width$}", col, width = max_width));
    }
    s.push('\n');

    // separator
    s.push_str("----+");
    for _ in 0..cols {
        s.push_str(&"-".repeat(max_width + 1));
    }
    s.push('\n');

    // rows
    for row in 0..rows {
        s.push_str(&format!("{:>3} |", row));

        for col in 0..cols {
            let value = cell(row, col);
            s.push_str(&format!(" {:>width$}", value, width = max_width));
        }

        s.push('\n');
    }

    s
}
