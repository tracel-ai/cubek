use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecurrenceWalkOperand {
    /// `[batch, position, key_head, key_feature]`: the queries, normalized and scaled.
    Query,
    /// `[batch, position, key_head, key_feature]`: the keys, normalized.
    Key,
    /// `[batch, position, key_head, head_in_group, value_feature]`: the values.
    Value,
    /// `[batch, position, key_head, head_in_group, rate]`: the log of the factor the state decays
    /// by before each write at [`DECAY_RATE`], and the strength of the write at
    /// [`STRENGTH_RATE`].
    Rates,
    /// `[batch, key_head, head_in_group, key_feature, value_feature]`: the state the walk resumes from.
    /// In a store's slots the first axis is the store's, `[store_slot, ..]`, and the walk writes
    /// the state after its last position back to it.
    State,
    /// `[batch, position, key_head, head_in_group, value_feature]`: what each position reads of the state
    /// it wrote.
    Out,
    /// `[kept_state, batch, key_head, head_in_group, key_feature, value_feature]`: the states after the
    /// walk's last positions, oldest first.
    Kept,
}

impl RecurrenceWalkOperand {
    /// Every operand.
    pub const ALL: [Self; 7] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::Rates,
        Self::State,
        Self::Out,
        Self::Kept,
    ];

    /// The operands a line of value columns runs along.
    pub const COLUMNS: [Self; 4] = [Self::Value, Self::State, Self::Out, Self::Kept];

    /// What every microkernel of the walk requires of the operands, an owner holding `columns` of a
    /// head's value columns over `rows` of its key rows a unit: the columns and the rows are read
    /// in lines of one width, so the width divides both; the rates are read a scalar at a time.
    pub fn require(columns: usize, rows: usize, vectorization: &mut Vectorization<Self>) {
        for operand in Self::COLUMNS {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: columns,
            });
            vectorization.require(VectorRequirement::Together {
                operand: Self::Value,
                with: operand,
            });
        }
        for operand in [Self::Query, Self::Key] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: rows,
            });
            vectorization.require(VectorRequirement::Together {
                operand: Self::Value,
                with: operand,
            });
        }
        vectorization.require(VectorRequirement::Elementwise {
            operand: Self::Rates,
        });
    }
}

/// The index of a position's log decay along the [`Rates`](RecurrenceWalkOperand::Rates) axis.
pub const DECAY_RATE: usize = 0;
/// The index of a position's write strength along the [`Rates`](RecurrenceWalkOperand::Rates) axis.
pub const STRENGTH_RATE: usize = 1;

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecurrenceWalkOperandMap<T> {
    pub query: T,
    pub key: T,
    pub value: T,
    pub rates: T,
    pub state: T,
    pub out: T,
    pub kept: T,
}

impl<T: Copy> RecurrenceWalkOperandMap<T> {
    /// The same value for every operand.
    pub fn splat(value: T) -> Self {
        Self {
            query: value,
            key: value,
            value,
            rates: value,
            state: value,
            out: value,
            kept: value,
        }
    }

    /// The value `operand` holds.
    pub fn get(&self, operand: RecurrenceWalkOperand) -> T {
        match operand {
            RecurrenceWalkOperand::Query => self.query,
            RecurrenceWalkOperand::Key => self.key,
            RecurrenceWalkOperand::Value => self.value,
            RecurrenceWalkOperand::Rates => self.rates,
            RecurrenceWalkOperand::State => self.state,
            RecurrenceWalkOperand::Out => self.out,
            RecurrenceWalkOperand::Kept => self.kept,
        }
    }
}
