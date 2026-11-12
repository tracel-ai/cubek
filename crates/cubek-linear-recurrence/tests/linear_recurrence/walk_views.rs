//! The walk reading its queries, keys and values as views into one buffer, as the conv leaves its
//! output: the values first, then the keys, then the queries, every position a row of the whole.

use super::harness::{F32, elements};
use super::walk::{Case, walk_or_skip};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::metadata::Metadata;
use cubek_linear_recurrence::kernels::walk::{CubeWalkRoutineSettings, RecurrenceWalkInputs};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_std::Hardware;

impl Case {
    /// The case's inputs with the values, keys and queries read in place from one buffer of `b, t`
    /// rows of `channels`: the values at 0, the keys after them, the queries after the keys.
    fn inputs_viewed(&self, client: &Client, element: ElemType) -> RecurrenceWalkInputs {
        let operands = self.operands();
        let RecurrenceHeads {
            key_heads,
            key_features,
            value_features,
            ..
        } = self.heads;
        let (rows, values, keys) = (
            self.batch * self.positions,
            self.value_heads() * value_features,
            key_heads * key_features,
        );
        let channels = values + 2 * keys;
        let mut buffer = vec![f32::NAN; rows * channels];
        for row in 0..rows {
            let at = row * channels;
            buffer[at..at + values].copy_from_slice(&operands.value[row * values..][..values]);
            buffer[at + values..at + values + keys]
                .copy_from_slice(&operands.key[row * keys..][..keys]);
            buffer[at + values + keys..at + channels]
                .copy_from_slice(&operands.query[row * keys..][..keys]);
        }
        let handle = client.create(Bytes::from_elems(buffer));
        let view = |start: usize, heads: usize, features: usize| {
            PlainTensor::of_handle(TensorHandle::from_metadata(
                handle
                    .clone()
                    .offset_start((start * size_of::<f32>()) as u64),
                Metadata::new(
                    vec![self.batch, self.positions, heads, features],
                    vec![self.positions * channels, channels, features, 1],
                ),
                F32,
            ))
        };
        RecurrenceWalkInputs {
            value: view(0, self.value_heads(), value_features),
            key: view(values, key_heads, key_features),
            query: view(values + keys, key_heads, key_features),
            ..self.inputs(client, &operands, element)
        }
    }
}

/// The values, keys and queries are views of one buffer, the keys and queries starting past the
/// values: the walk reads them where they lie, whatever its rule and the element it writes.
#[test]
fn the_walk_reads_values_keys_and_queries_as_views_of_one_buffer() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for rule in [UpdateRule::Delta, UpdateRule::Additive] {
        let case = Case {
            batch: 2,
            positions: 5,
            heads: RecurrenceHeads {
                key_heads: 2,
                heads_per_group: 3,
                key_features: 16,
                value_features: 20,
            },
            kept: 2,
            rule,
            masked: vec![3],
        };
        for element in elements(&client) {
            for settings in CubeWalkRoutineSettings::listed() {
                let inputs = case.inputs_viewed(&client, element);
                let what = format!("views {settings} {case:?}");
                if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                    case.check(&client, &case.operands(), walked, element, &what);
                }
            }
        }
    }
}
