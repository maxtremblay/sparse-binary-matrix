use is_sorted::IsSorted;
use itertools::Itertools;
use std::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
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

pub(crate) fn validate_positions(length: usize, positions: &[usize]) -> Result<(), PositionsError> {
    for position in positions.iter() {
        if *position >= length {
            return Result::Err(PositionsError::OutOfBound);
        }
    }
    if !IsSorted::is_sorted(&mut positions.iter()) {
        return Result::Err(PositionsError::Unsorted);
    }
    if positions.iter().unique().count() != positions.len() {
        return Result::Err(PositionsError::Duplicated);
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct IncompatibleDimensions<DL, DR> {
    left_dimensions: DL,
    right_dimensions: DR,
}

impl<DL, DR> IncompatibleDimensions<DL, DR> {
    pub fn new(left_dimensions: DL, right_dimensions: DR) -> Self {
        Self {
            left_dimensions,
            right_dimensions,
        }
    }
}

pub type VecVecIncompatibleDimensions = IncompatibleDimensions<usize, usize>;
pub type MatVecIncompatibleDimensions = IncompatibleDimensions<(usize, usize), usize>;
pub type VecMatIncompatibleDimensions = IncompatibleDimensions<usize, (usize, usize)>;
pub type MatMatIncompatibleDimensions = IncompatibleDimensions<(usize, usize), (usize, usize)>;

macro_rules! impl_dim_error {
    ($dl:ty, $dr:ty) => {
        impl fmt::Display for IncompatibleDimensions<$dl, $dr> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                format!(
                    "incompatible dimensions {:?} and {:?}",
                    self.left_dimensions, self.right_dimensions
                )
                .fmt(f)
            }
        }

        impl std::error::Error for IncompatibleDimensions<$dl, $dr> {}
    };
}

impl_dim_error!(usize, usize); // Vec - Vec
impl_dim_error!((usize, usize), usize); // Mat - Vec
impl_dim_error!(usize, (usize, usize)); // Vec - Mat
impl_dim_error!((usize, usize), (usize, usize)); // Mat - Mat
