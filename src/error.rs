use std::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PositionsError {
    Unsorted,
    OutOfBound,
    Duplicated,
}

impl fmt::Display for PositionsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PositionsError::Unsorted => "some positions are not sorted".fmt(f),
            PositionsError::OutOfBound => "some positions are out of bound".fmt(f),
            PositionsError::Duplicated => "some positions are duplicated".fmt(f),
        }
    }
}

impl std::error::Error for PositionsError {}
