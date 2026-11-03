use cubecl::tile::launch::{VectorRequirement, Vectorization};

/// One of the operands of the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatedDeltaWalkOperand {
    /// `[batch, position, key_head, key_feature]`: the queries, normalized and scaled.
    Query,
    /// `[batch, position, key_head, key_feature]`: the keys, normalized.
    Key,
    /// `[batch, position, key_head, group, value_feature]`: the values.
    Value,
    /// `[batch, position, key_head, group]`: the log of the factor the state decays by before each
    /// write.
    Decay,
    /// `[batch, position, key_head, group]`: the strength of each write.
    Strength,
    /// `[batch, key_head, group, key_feature, value_feature]`: the state the walk resumes from.
    State,
    /// `[batch, position, key_head, group, value_feature]`: what each position reads of the state
    /// it wrote.
    Out,
    /// `[kept_state, batch, key_head, group, key_feature, value_feature]`: the states after the
    /// walk's last positions, oldest first.
    Kept,
}

impl GatedDeltaWalkOperand {
    /// Every operand.
    pub const ALL: [Self; 8] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::Decay,
        Self::Strength,
        Self::State,
        Self::Out,
        Self::Kept,
    ];

    /// The operands a line of value columns runs along.
    pub const COLUMNS: [Self; 4] = [Self::Value, Self::State, Self::Out, Self::Kept];

    /// What every microkernel of the walk requires of the operands, an owner holding `columns` of a
    /// head's value columns over `rows` of its key rows a unit: the columns and the rows are read
    /// in lines of one width, so the width divides both; the decay and the strength are one scalar
    /// a position.
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
        for operand in [Self::Decay, Self::Strength] {
            vectorization.require(VectorRequirement::Elementwise { operand });
        }
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GatedDeltaWalkOperandMap<T> {
    pub query: T,
    pub key: T,
    pub value: T,
    pub decay: T,
    pub strength: T,
    pub state: T,
    pub out: T,
    pub kept: T,
}

impl<T: Copy> GatedDeltaWalkOperandMap<T> {
    /// The same value for every operand.
    pub fn splat(value: T) -> Self {
        Self {
            query: value,
            key: value,
            value,
            decay: value,
            strength: value,
            state: value,
            out: value,
            kept: value,
        }
    }

    /// The value `operand` holds.
    pub fn get(&self, operand: GatedDeltaWalkOperand) -> T {
        match operand {
            GatedDeltaWalkOperand::Query => self.query,
            GatedDeltaWalkOperand::Key => self.key,
            GatedDeltaWalkOperand::Value => self.value,
            GatedDeltaWalkOperand::Decay => self.decay,
            GatedDeltaWalkOperand::Strength => self.strength,
            GatedDeltaWalkOperand::State => self.state,
            GatedDeltaWalkOperand::Out => self.out,
            GatedDeltaWalkOperand::Kept => self.kept,
        }
    }
}
