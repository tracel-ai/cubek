/// What a pass does to each cell of the held state before the cell is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowUpdate {
    /// `cell = cell · factor`.
    Decay,
    /// `cell = cell + key · write`.
    Write,
    /// `cell = cell · factor + key · write`.
    DecayWrite,
}

/// Which window of the position a pass reads each updated cell against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowRead {
    Keys,
    Queries,
}
