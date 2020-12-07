//! Error types for matrix and vector operations.
//!
//! This contains two kind of errors.
//! [`InvalidPositions`](InvalidPositions) represents errors
//! when building a vector or matrix with invalid positions.
//! [`IncompatibleDimensions`](IncompatibleDimensions) represents errors
//! when two objects have incompatible dimensions for a given operations
//! such as addition or multiplication.

use is_sorted::IsSorted;
use itertools::Itertools;
use std::fmt;

/// An error to represent invalid positions in a vector or matrix.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum InvalidPositions {
    Unsorted,
    OutOfBound,
    Duplicated,
}

impl fmt::Display for InvalidPositions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InvalidPositions::Unsorted => "some positions are not sorted".fmt(f),
            InvalidPositions::OutOfBound => "some positions are out of bound".fmt(f),
            InvalidPositions::Duplicated => "some positions are duplicated".fmt(f),
        }
    }
}

impl std::error::Error for InvalidPositions {}

pub(crate) fn validate_positions(
    length: usize,
    positions: &[usize],
) -> Result<(), InvalidPositions> {
    for position in positions.iter() {
        if *position >= length {
            return Result::Err(InvalidPositions::OutOfBound);
        }
    }
    if !IsSorted::is_sorted(&mut positions.iter()) {
        return Result::Err(InvalidPositions::Unsorted);
    }
    if positions.iter().unique().count() != positions.len() {
        return Result::Err(InvalidPositions::Duplicated);
    }
    Ok(())
}

/// An error to represent incompatible dimensions
/// in matrix and vector operations.
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
