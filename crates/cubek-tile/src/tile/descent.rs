//! What a tile has been descended with: the [`Level`]s its ancestors' `at`s were handed, in
//! nesting order. Recorded as the kernel walks, read back by the drain, which has to find each
//! fragment's window in an output that was never walked.
//!
//! One record is shared by every tile of one descent (the handle is cloned, not the levels), and
//! each tile knows its own depth in it. A tile descended twice at the same depth must be given
//! the same level: the loops are the only statement of the nest, and this is where two of them
//! would disagree.

use std::{cell::RefCell, hash::Hash, rc::Rc};

use crate::Level;

#[derive(Clone)]
pub struct Descent {
    levels: Rc<RefCell<Vec<Level>>>,
    depth: usize,
}

impl Descent {
    /// A tile nothing has descended yet.
    pub(crate) fn root() -> Self {
        Descent {
            levels: Rc::new(RefCell::new(Vec::new())),
            depth: 0,
        }
    }

    /// The child's descent under `level`: recorded at this depth the first time, checked against
    /// the record every time after.
    pub(crate) fn under(&self, level: &Level) -> Descent {
        {
            let mut levels = self.levels.borrow_mut();
            if levels.len() == self.depth {
                levels.push(level.clone());
            } else {
                assert!(
                    levels[self.depth] == *level,
                    "Tile::at: this tile was descended at depth {} with two different levels, \
                     {:?} and then {:?}; the loops that walk it disagree with each other",
                    self.depth,
                    levels[self.depth],
                    level
                );
            }
        }
        Descent {
            levels: Rc::clone(&self.levels),
            depth: self.depth + 1,
        }
    }

    /// The levels recorded below this tile, outermost first.
    pub(crate) fn below(&self) -> Vec<Level> {
        self.levels.borrow()[self.depth..].to_vec()
    }
}

impl PartialEq for Descent {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && *self.levels.borrow() == *other.levels.borrow()
    }
}

impl Eq for Descent {}

impl Hash for Descent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.levels.borrow().hash(state);
    }
}

impl std::fmt::Debug for Descent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Descent")
            .field("depth", &self.depth)
            .field("levels", &*self.levels.borrow())
            .finish()
    }
}
