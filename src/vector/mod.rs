use crate::BinaryNumber;
use is_sorted::IsSorted;
use std::collections::HashMap;
use std::ops::{Add, Deref};

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
    /// Creates a new vector with the given length
    /// and list of non trivial positions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(5, vec![0, 2]);
    ///
    /// assert_eq!(vector.len(), 5);
    /// assert_eq!(vector.weight(), 2);
    ///
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if a position is greater or equal to the length.
    ///
    /// ```should_panic
    /// # use sparse_bin_mat::SparseBinVec;
    /// let vector = SparseBinVec::new(2, vec![1, 3]);
    /// ```
    pub fn new(length: usize, mut positions: Vec<usize>) -> Self {
        for position in positions.iter() {
            if *position >= length {
                panic!(
                    "position {} is out of bound for length {}",
                    position, length
                );
            }
        }
        positions.sort();
        Self { positions, length }
    }

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
        Self::new(length, Vec::new())
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

    pub(crate) fn take_inner_vec(self) -> Vec<usize> {
        self.positions
    }
}

impl<'a> SparseBinSlice<'a> {
    /// Creates a new vector with the given length
    /// and list of non trivial positions.
    ///
    /// This take a mutable reference to the positions in order to sort them.
    /// If you know the positions are sorted, you can instead use
    /// [`new_from_sorted`](SparseBinSlice::new_from_sorted).
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinSlice;
    /// let mut positions = vec![2, 0];
    /// let vector = SparseBinSlice::new(5, &mut positions);
    ///
    /// assert_eq!(vector.len(), 5);
    /// assert_eq!(vector.weight(), 2);
    /// assert_eq!(vector.non_trivial_positions().collect::<Vec<_>>(), vec![0, 2]);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if a position is greater or equal to the length.
    ///
    /// ```should_panic
    /// # use sparse_bin_mat::SparseBinSlice;
    /// let vector = SparseBinSlice::new(2, &mut [1, 3]);
    /// ```
    pub fn new(length: usize, positions: &'a mut [usize]) -> Self {
        for position in positions.iter() {
            if *position >= length {
                panic!(
                    "position {} is out of bound for length {}",
                    position, length
                );
            }
        }
        positions.sort();
        Self { positions, length }
    }

    /// Creates a new vector with the given length and a sorted list of non trivial positions.
    ///
    //// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinSlice;
    /// let mut positions = vec![0, 2];
    /// let vector = SparseBinSlice::new_from_sorted(5, &positions);
    ///
    /// assert_eq!(vector.len(), 5);
    /// assert_eq!(vector.weight(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the list of positions is unsorted or if a position is greater or equal to the
    /// length.
    ///
    /// ```should_panic
    /// # use sparse_bin_mat::SparseBinSlice;
    /// let mut positions = vec![2, 0];
    /// let vector = SparseBinSlice::new_from_sorted(5, &positions);
    /// ```
    pub fn new_from_sorted(length: usize, positions: &'a [usize]) -> Self {
        for position in positions.iter() {
            if *position >= length {
                panic!(
                    "position {} is out of bound for length {}",
                    position, length
                );
            }
        }
        // Waiting for the is_sorted API to stabilize in std.
        // https://github.com/rust-lang/rust/issues/53485
        if !IsSorted::is_sorted(&mut positions.iter()) {
            panic!("positions are unsorted");
        }
        Self { length, positions }
    }
}

impl<T: Deref<Target = [usize]>> SparseBinVecBase<T> {
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

    /// Returns true if the value at the given position 0
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

    /// Returns true if the value at the given position 1
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
        SparseBinVec::new(self.len() + other.len(), positions)
    }

    /// Returns a new vector keeping only the given positions.
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
    /// assert_eq!(vector.keep_only_positions(&[0, 1, 2]), truncated);
    /// assert_eq!(vector.keep_only_positions(&[1, 2]).len(), 2);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some positions are out of bound.
    pub fn keep_only_positions(&self, positions: &[usize]) -> SparseBinVec {
        for position in positions {
            if *position >= self.len() {
                panic!(
                    "position {} is out of bound for length {}",
                    position,
                    self.len()
                );
            }
        }
        let old_to_new_positions_map = positions
            .iter()
            .enumerate()
            .map(|(new, old)| (old, new))
            .collect::<HashMap<_, _>>();
        let new_positions = self
            .non_trivial_positions()
            .filter_map(|position| old_to_new_positions_map.get(&position).cloned())
            .collect();
        SparseBinVec::new(positions.len(), new_positions)
    }

    /// Returns a truncated vector where the given positions are removed.
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
    /// assert_eq!(vector.without_positions(&[3, 4]), truncated);
    /// assert_eq!(vector.without_positions(&[1, 2]).len(), 3);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some positions are out of bound.
    pub fn without_positions(&self, positions: &[usize]) -> SparseBinVec {
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

    /// Returns the dot product of two vectors.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinVec;
    /// let first = SparseBinVec::new(4, vec![0, 1, 2]);
    /// let second = SparseBinVec::new(4, vec![1, 2, 3]);
    /// let third = SparseBinVec::new(4, vec![0, 3]);
    ///
    /// assert_eq!(first.dot_with(&second), 0);
    /// assert_eq!(first.dot_with(&third), 1);
    /// ```
    pub fn dot_with<S: Deref<Target = [usize]>>(
        &self,
        other: &SparseBinVecBase<S>,
    ) -> BinaryNumber {
        BitwiseZipIter::new(self.as_view(), other.as_view())
            .fold(0, |sum, x| sum ^ x.first_row_value * x.second_row_value)
    }
}

impl<S, T> Add<&SparseBinVecBase<S>> for &SparseBinVecBase<T>
where
    S: Deref<Target = [usize]>,
    T: Deref<Target = [usize]>,
{
    type Output = SparseBinVec;

    fn add(self, other: &SparseBinVecBase<S>) -> SparseBinVec {
        if self.len() != other.len() {
            panic!(
                "vector of length {} can't be added to vector of length {}",
                self.len(),
                other.len()
            );
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
        SparseBinVec::new(self.len(), positions)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn positions_are_sorted_on_construction() {
        let vector = SparseBinVec::new(4, vec![3, 0, 2]);

        assert_eq!(
            vector.non_trivial_positions().collect::<Vec<_>>(),
            vec![0, 2, 3]
        )
    }

    #[test]
    fn addition() {
        let first_vector = SparseBinVec::new(6, vec![0, 2, 4]);
        let second_vector = SparseBinVec::new(6, vec![0, 1, 2]);
        let sum = SparseBinVec::new(6, vec![1, 4]);
        assert_eq!(&first_vector + &second_vector, sum);
    }

    #[test]
    fn panics_on_addition_if_different_length() {
        let vector_6 = SparseBinVec::new(6, vec![0, 2, 4]);
        let vector_2 = SparseBinVec::new(2, vec![0]);

        let result = std::panic::catch_unwind(|| &vector_6 + &vector_2);
        assert!(result.is_err());
    }
}
