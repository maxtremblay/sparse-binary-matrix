use crate::error::{validate_positions, IncompatibleDimensions, InvalidPositions};
use crate::BinaryNumber;
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Deref, Mul};

mod bitwise_operations;
use bitwise_operations::BitwiseZipIter;

/// A sparse binary vector.
///
/// There are two main variants of a vector,
/// the owned one, [`SparseBinVec`](crate::SparseBinVec), and the borrowed one,
/// [`SparseBinSlice`](crate::SparseBinSlice).
/// Most of the time, you want to create a owned version.
/// However, some iterators, such as those defined on [`SparseBinMat`](crate::SparseBinMat)
/// returns the borrowed version.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SparseBinVecBase<T> {
    positions: T,
    length: usize,
}

pub type SparseBinVec = SparseBinVecBase<Vec<usize>>;
pub type SparseBinSlice<'a> = SparseBinVecBase<&'a [usize]>;

impl SparseBinVec {
    /// Creates a vector fill with zeros of the given length.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::zeros(3);
    ///
    /// assert_eq!(vector.len(), 3);
    /// assert_eq!(vector.weight(), 0);
    /// ```
    pub fn zeros(length: usize) -> Self {
        Self {
            length,
            positions: Vec::new(),
        }
    }

    /// Creates an empty vector.
    ///
    /// This allocate minimally, so it is a good placeholder.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::empty();
    ///
    /// assert_eq!(vector.len(), 0);
    /// assert_eq!(vector.weight(), 0);
    /// ```
    pub fn empty() -> Self {
        Self {
            length: 0,
            positions: Vec::new(),
        }
    }

    /// Converts the sparse binary vector to a `Vec` of
    /// the non trivial positions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.to_positions_vec(), vec![0, 2]);
    /// ```
    pub fn to_positions_vec(self) -> Vec<usize> {
        self.positions
    }
}

impl<T: Deref<Target = [usize]>> SparseBinVecBase<T> {
    /// Creates a new vector with the given length and list of non trivial positions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// use sparse_bin_mat::error::InvalidPositions;
    ///
    /// let vector = SparseBinVec::new(5, vec![0, 2]);
    ///
    /// assert_eq!(vector.len(), 5);
    /// assert_eq!(vector.weight(), 2);
    /// ```
    pub fn new(length: usize, positions: T) -> Self {
        Self::try_new(length, positions).unwrap()
    }

    /// Creates a new vector with the given length and list of non trivial positions
    /// or returns as error if the positions are unsorted, greater or equal to length
    /// or contain duplicates.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// use sparse_bin_mat::error::InvalidPositions;
    ///
    /// let vector = SparseBinVec::try_new(5, vec![0, 2]);
    /// assert_eq!(vector, Ok(SparseBinVec::new(5, vec![0, 2])));
    ///
    /// let vector = SparseBinVec::try_new(5, vec![2, 0]);
    /// assert_eq!(vector, Err(InvalidPositions::Unsorted));
    ///
    /// let vector = SparseBinVec::try_new(5, vec![0, 10]);
    /// assert_eq!(vector, Err(InvalidPositions::OutOfBound));
    ///
    /// let vector = SparseBinVec::try_new(5, vec![0, 0]);
    /// assert_eq!(vector, Err(InvalidPositions::Duplicated));
    /// ```
    pub fn try_new(length: usize, positions: T) -> Result<Self, InvalidPositions> {
        validate_positions(length, &positions)?;
        Ok(Self { positions, length })
    }

    // Positions should be sorted, in bound and all unique.
    pub(crate) fn new_unchecked(length: usize, positions: T) -> Self {
        Self { positions, length }
    }

    /// Returns the length (number of elements) of the vector.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns the number of elements with value 1 in the vector.
    pub fn weight(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if the length of the vector is 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if all the element in the vector are 0.
    pub fn is_zero(&self) -> bool {
        self.weight() == 0
    }

    /// Returns the value at the given position
    /// or None if the position is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.get(0), Some(1));
    /// assert_eq!(vector.get(1), Some(0));
    /// assert_eq!(vector.get(2), Some(1));
    /// assert_eq!(vector.get(3), None);
    /// ```
    pub fn get(&self, position: usize) -> Option<BinaryNumber> {
        if position < self.len() {
            if self.positions.contains(&position) {
                Some(1)
            } else {
                Some(0)
            }
        } else {
            None
        }
    }

    /// Returns true if the value at the given position is 0
    /// or None if the position is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.is_zero_at(0), Some(false));
    /// assert_eq!(vector.is_zero_at(1), Some(true));
    /// assert_eq!(vector.is_zero_at(2), Some(false));
    /// assert_eq!(vector.is_zero_at(3), None);
    /// ```
    pub fn is_zero_at(&self, position: usize) -> Option<bool> {
        self.get(position).map(|value| value == 0)
    }

    /// Returns true if the value at the given position is 1
    /// or None if the position is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.is_one_at(0), Some(true));
    /// assert_eq!(vector.is_one_at(1), Some(false));
    /// assert_eq!(vector.is_one_at(2), Some(true));
    /// assert_eq!(vector.is_one_at(3), None);
    /// ```
    pub fn is_one_at(&self, position: usize) -> Option<bool> {
        self.get(position).map(|value| value == 1)
    }

    /// Returns an iterator over all positions where the value is 1.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(5, vec![0, 1, 3]);
    /// let mut iter = vector.non_trivial_positions();
    ///
    /// assert_eq!(iter.next(), Some(0));
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn non_trivial_positions<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        self.positions.iter().cloned()
    }

    /// Returns the concatenation of two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let left_vector = SparseBinVec::new(3, vec![0, 1]);
    /// let right_vector = SparseBinVec::new(4, vec![2, 3]);
    ///
    /// let concatened = left_vector.concat(&right_vector);
    ///
    /// let expected = SparseBinVec::new(7, vec![0, 1, 5, 6]);
    ///
    /// assert_eq!(concatened, expected);
    /// ```
    pub fn concat(&self, other: &Self) -> SparseBinVec {
        let positions = self
            .non_trivial_positions()
            .chain(other.non_trivial_positions().map(|pos| pos + self.len()))
            .collect();
        SparseBinVec::new_unchecked(self.len() + other.len(), positions)
    }

    /// Returns a new vector keeping only the given positions or an error
    /// if the positions are unsorted, out of bound or contain deplicate.
    ///
    /// Positions are relabeled to the fit new number of positions.
    ///
    /// # Example
    ///
    /// ```
    /// use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(5, vec![0, 2, 4]);
    /// let truncated = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.keep_only_positions(&[0, 1, 2]), Ok(truncated));
    /// assert_eq!(vector.keep_only_positions(&[1, 2]).map(|vec| vec.len()), Ok(2));
    /// ```
    pub fn keep_only_positions(
        &self,
        positions: &[usize],
    ) -> Result<SparseBinVec, InvalidPositions> {
        validate_positions(self.length, positions)?;
        let old_to_new_positions_map = positions
            .iter()
            .enumerate()
            .map(|(new, old)| (old, new))
            .collect::<HashMap<_, _>>();
        let new_positions = self
            .non_trivial_positions()
            .filter_map(|position| old_to_new_positions_map.get(&position).cloned())
            .collect();
        Ok(SparseBinVec::new_unchecked(positions.len(), new_positions))
    }

    /// Returns a truncated vector where the given positions are remove or an error
    /// if the positions are unsorted or out of bound.
    ///
    /// Positions are relabeled to fit the new number of positions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(5, vec![0, 2, 4]);
    /// let truncated = SparseBinVec::new(3, vec![0, 2]);
    ///
    /// assert_eq!(vector.without_positions(&[3, 4]), Ok(truncated));
    /// assert_eq!(vector.without_positions(&[1, 2]).map(|vec| vec.len()), Ok(3));
    /// ```
    pub fn without_positions(&self, positions: &[usize]) -> Result<SparseBinVec, InvalidPositions> {
        let to_keep: Vec<usize> = (0..self.len()).filter(|x| !positions.contains(x)).collect();
        self.keep_only_positions(&to_keep)
    }

    /// Returns a view over the vector.
    pub fn as_view(&self) -> SparseBinSlice {
        SparseBinSlice {
            length: self.length,
            positions: &self.positions,
        }
    }

    /// Returns a slice of the non trivial positions.
    pub fn as_slice(&self) -> &[usize] {
        self.positions.as_ref()
    }

    /// Returns an owned version of the vector.
    pub fn to_owned(self) -> SparseBinVec {
        SparseBinVec {
            length: self.length,
            positions: self.positions.to_owned(),
        }
    }

    /// Returns the dot product of two vectors or an
    /// error if the vectors have different length.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let first = SparseBinVec::new(4, vec![0, 1, 2]);
    /// let second = SparseBinVec::new(4, vec![1, 2, 3]);
    /// let third = SparseBinVec::new(4, vec![0, 3]);
    ///
    /// assert_eq!(first.dot_with(&second), Ok(0));
    /// assert_eq!(first.dot_with(&third), Ok((1)));
    /// ```
    pub fn dot_with<S: Deref<Target = [usize]>>(
        &self,
        other: &SparseBinVecBase<S>,
    ) -> Result<BinaryNumber, IncompatibleDimensions<usize, usize>> {
        if self.len() != other.len() {
            return Err(IncompatibleDimensions::new(self.len(), other.len()));
        }
        Ok(BitwiseZipIter::new(self.as_view(), other.as_view())
            .fold(0, |sum, x| sum ^ x.first_row_value * x.second_row_value))
    }

    /// Returns the bitwise xor of two vectors or an
    /// error if the vectors have different length.
    ///
    /// Use the Add (+) operator if you want a version
    /// that panics instead or returning an error.
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let first = SparseBinVec::new(4, vec![0, 1, 2]);
    /// let second = SparseBinVec::new(4, vec![1, 2, 3]);
    /// let third = SparseBinVec::new(4, vec![0, 3]);
    ///
    /// assert_eq!(first.bitwise_xor_with(&second), Ok(third));
    /// ```
    pub fn bitwise_xor_with<S: Deref<Target = [usize]>>(
        &self,
        other: &SparseBinVecBase<S>,
    ) -> Result<SparseBinVec, IncompatibleDimensions<usize, usize>> {
        if self.len() != other.len() {
            return Err(IncompatibleDimensions::new(self.len(), other.len()));
        }
        let positions = BitwiseZipIter::new(self.as_view(), other.as_view())
            .filter_map(|x| {
                if x.first_row_value ^ x.second_row_value == 1 {
                    Some(x.position)
                } else {
                    None
                }
            })
            .collect();
        Ok(SparseBinVec::new_unchecked(self.len(), positions))
    }
}

impl<S, T> Add<&SparseBinVecBase<S>> for &SparseBinVecBase<T>
where
    S: Deref<Target = [usize]>,
    T: Deref<Target = [usize]>,
{
    type Output = SparseBinVec;

    fn add(self, other: &SparseBinVecBase<S>) -> Self::Output {
        self.bitwise_xor_with(other).expect(&format!(
            "vector of length {} can't be added to vector of length {}",
            self.len(),
            other.len()
        ))
    }
}

impl<S, T> Mul<&SparseBinVecBase<S>> for &SparseBinVecBase<T>
where
    S: Deref<Target = [usize]>,
    T: Deref<Target = [usize]>,
{
    type Output = BinaryNumber;

    fn mul(self, other: &SparseBinVecBase<S>) -> Self::Output {
        self.dot_with(other).expect(&format!(
            "vector of length {} can't be dotted to vector of length {}",
            self.len(),
            other.len()
        ))
    }
}

impl<T: Deref<Target = [usize]>> fmt::Display for SparseBinVecBase<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.positions.deref())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn addition() {
        let first_vector = SparseBinVec::new_unchecked(6, vec![0, 2, 4]);
        let second_vector = SparseBinVec::new_unchecked(6, vec![0, 1, 2]);
        let sum = SparseBinVec::new_unchecked(6, vec![1, 4]);
        assert_eq!(&first_vector + &second_vector, sum);
    }

    #[test]
    fn panics_on_addition_if_different_length() {
        let vector_6 = SparseBinVec::new_unchecked(6, vec![0, 2, 4]);
        let vector_2 = SparseBinVec::new_unchecked(2, vec![0]);

        let result = std::panic::catch_unwind(|| &vector_6 + &vector_2);
        assert!(result.is_err());
    }
}
