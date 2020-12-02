use crate::BinaryNumber;
use crate::SparseBinSlice;
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::{Add, Mul};

mod concat;
use concat::{concat_horizontally, concat_vertically};

mod constructor_utils;
use constructor_utils::{assert_rows_are_inbound, initialize_from};

mod gauss_jordan;
use gauss_jordan::GaussJordan;

mod nullspace;
use nullspace::nullspace;

mod rows;
pub use self::rows::Rows;

mod transpose;
use transpose::transpose;

/// A sparse binary matrix optimized for row operations.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SparseBinMat {
    row_ranges: Vec<usize>,
    column_indices: Vec<usize>,
    number_of_columns: usize,
}

impl SparseBinMat {
    /// Creates a new matrix with the given number of columns
    /// and list of rows.
    ///
    /// A row is a list of the positions where the elements have value 1.
    /// All rows are sorted during insertion.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(4, vec![vec![0, 1, 2], vec![0, 2, 3]]);
    ///
    /// assert_eq!(matrix.number_of_rows(), 2);
    /// assert_eq!(matrix.number_of_columns(), 4);
    /// assert_eq!(matrix.number_of_elements(), 8);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if a position in a row is greater or equal to
    /// the number of columns.
    ///
    /// ```should_panic
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(2, vec![vec![1, 2], vec![3, 0]]);
    /// ```
    pub fn new(number_of_columns: usize, rows: Vec<Vec<usize>>) -> Self {
        assert_rows_are_inbound(number_of_columns, &rows);
        let (row_ranges, column_indices) = initialize_from(rows);
        Self {
            row_ranges,
            column_indices,
            number_of_columns,
        }
    }

    /// Creates an identity matrix of the given length.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::identity(5);
    ///
    /// let identity_rows = (0..5).map(|x| vec![x]).collect();
    /// let identity_matrix = SparseBinMat::new(5, identity_rows);
    ///
    /// assert_eq!(matrix, identity_matrix);
    /// ```
    pub fn identity(length: usize) -> Self {
        Self {
            column_indices: (0..length).collect(),
            row_ranges: (0..length + 1).collect(),
            number_of_columns: length,
        }
    }

    /// Creates a matrix fill with zeros of the given dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::zeros(2, 3);
    ///
    /// assert_eq!(matrix.number_of_rows(), 2);
    /// assert_eq!(matrix.number_of_columns(), 3);
    /// assert_eq!(matrix.number_of_zeros(), 6);
    /// assert_eq!(matrix.number_of_ones(), 0);
    /// ```
    pub fn zeros(number_of_rows: usize, number_of_columns: usize) -> Self {
        Self::new(number_of_columns, vec![Vec::new(); number_of_rows])
    }

    /// Creates an empty matrix.
    ///
    /// This allocate minimally, so it is a good placeholder.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::empty();
    ///
    /// assert_eq!(matrix.number_of_rows(), 0);
    /// assert_eq!(matrix.number_of_columns(), 0); assert_eq!(matrix.number_of_elements(), 0);
    /// // Note that these are not equal since new preallocate some space
    /// // to store the data.
    /// assert_ne!(SparseBinMat::new(0, Vec::new()), SparseBinMat::empty());
    ///
    /// // To test for emptyness, you should prefer the following.
    /// assert!(matrix.is_empty());
    /// ```
    pub fn empty() -> Self {
        Self {
            column_indices: Vec::new(),
            row_ranges: Vec::new(),
            number_of_columns: 0,
        }
    }

    /// Returns the number of columns in the matrix.
    pub fn number_of_columns(&self) -> usize {
        self.number_of_columns
    }

    /// Returns the number of rows in the matrix
    pub fn number_of_rows(&self) -> usize {
        match self.row_ranges.len() {
            0 => 0,
            n => n - 1,
        }
    }

    /// Returns the number of rows and columns in the matrix.
    pub fn dimension(&self) -> (usize, usize) {
        (self.number_of_rows(), self.number_of_columns())
    }

    /// Returns the number of elements in the matrix.
    pub fn number_of_elements(&self) -> usize {
        self.number_of_rows() * self.number_of_columns()
    }

    /// Returns the number of elements with value 0 in the matrix.
    pub fn number_of_zeros(&self) -> usize {
        self.number_of_elements() - self.number_of_ones()
    }

    /// Returns the number of elements with value 1 in the matrix.
    pub fn number_of_ones(&self) -> usize {
        self.column_indices.len()
    }

    /// Returns true if the number of elements in the matrix is 0.
    pub fn is_empty(&self) -> bool {
        self.number_of_elements() == 0
    }

    /// Returns the value at the given row and column
    /// or None if one of the index is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let rows = vec![vec![0, 1], vec![1, 2]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// assert_eq!(matrix.get(0, 0), Some(1));
    /// assert_eq!(matrix.get(1, 0), Some(0));
    /// assert_eq!(matrix.get(2, 0), None);
    /// ```
    pub fn get(&self, row: usize, column: usize) -> Option<BinaryNumber> {
        if column < self.number_of_columns() {
            self.row(row).and_then(|row| row.get(column))
        } else {
            None
        }
    }

    /// Returns true if the value at the given row and column is 0
    /// or None if one of the index is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let rows = vec![vec![0, 1], vec![1, 2]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// assert_eq!(matrix.is_zero_at(0, 0), Some(false));
    /// assert_eq!(matrix.is_zero_at(1, 0), Some(true));
    /// assert_eq!(matrix.is_zero_at(2, 0), None);
    /// ```
    pub fn is_zero_at(&self, row: usize, column: usize) -> Option<bool> {
        self.get(row, column).map(|value| value == 0)
    }

    /// Returns true if the value at the given row and column is 1
    /// or None if one of the index is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let rows = vec![vec![0, 1], vec![1, 2]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// assert_eq!(matrix.is_one_at(0, 0), Some(true));
    /// assert_eq!(matrix.is_one_at(1, 0), Some(false));
    /// assert_eq!(matrix.is_one_at(2, 0), None);
    /// ```
    pub fn is_one_at(&self, row: usize, column: usize) -> Option<bool> {
        self.get(row, column).map(|value| value == 1)
    }

    /// Returns a reference to the given row of the matrix
    /// or None if the row index is out of bound.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::{SparseBinSlice, SparseBinMat};
    /// let rows = vec![vec![0, 1], vec![1, 2]];
    /// let matrix = SparseBinMat::new(3, rows.clone());
    ///
    /// assert_eq!(matrix.row(0), Some(SparseBinSlice::new_from_sorted(3, &rows[0])));
    /// assert_eq!(matrix.row(1), Some(SparseBinSlice::new_from_sorted(3, &rows[1])));
    /// assert_eq!(matrix.row(2), None);
    /// ```
    pub fn row(&self, row: usize) -> Option<SparseBinSlice> {
        let row_start = self.row_ranges.get(row)?;
        let row_end = self.row_ranges.get(row + 1)?;
        Some(SparseBinSlice::new_from_sorted(
            self.number_of_columns(),
            &self.column_indices[*row_start..*row_end],
        ))
    }

    /// Returns an iterator yielding the rows of the matrix
    /// as slice of non zero positions.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::{SparseBinSlice, SparseBinMat};
    /// let rows = vec![vec![0, 1, 2, 5], vec![1, 3, 4], vec![2, 4, 5], vec![0, 5]];
    /// let matrix = SparseBinMat::new(7, rows.clone());
    ///
    /// let mut iter = matrix.rows();
    ///
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new_from_sorted(7, &rows[0])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new_from_sorted(7, &rows[1])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new_from_sorted(7, &rows[2])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new_from_sorted(7, &rows[3])));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn rows(&self) -> Rows {
        Rows::from(self)
    }

    /// Returns an iterator yielding the number
    /// of non zero elements in each row of the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let rows = vec![vec![0, 1, 2, 5], vec![1, 3, 4], vec![2, 4, 5], vec![0, 5]];
    /// let matrix = SparseBinMat::new(7, rows);
    ///
    /// assert_eq!(matrix.row_weights().collect::<Vec<usize>>(), vec![4, 3, 3, 2]);
    /// ```
    pub fn row_weights<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        self.rows().map(|row| row.weight())
    }

    /// Gets the transposed version of the matrix
    /// by swapping the columns with the rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    ///
    /// let rows = vec![vec![0, 1, 2], vec![1, 3], vec![0, 2, 3]];
    /// let matrix = SparseBinMat::new(4, rows);
    ///
    /// let transposed_matrix = matrix.transposed();
    ///
    /// let expected_rows = vec![vec![0, 2], vec![0, 1], vec![0, 2], vec![1, 2]];
    /// let expected_matrix = SparseBinMat::new(3, expected_rows);
    ///
    /// assert_eq!(transposed_matrix, expected_matrix);
    /// ```
    pub fn transposed(&self) -> Self {
        transpose(self)
    }

    /// Computes the rank of the matrix.
    /// That is, the number of linearly independent rows or columns.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    ///
    /// let rows = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// assert_eq!(matrix.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        GaussJordan::new(self).rank()
    }

    /// Returns an echeloned version of the matrix.
    ///
    /// A matrix in echelon form as the property that no
    /// rows any given row have a 1 in the first non trivial
    /// position of that row. Also, all rows are linearly
    /// independent.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let rows = vec![vec![0, 1, 2], vec![0], vec![1, 2], vec![0, 2]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// let expected = SparseBinMat::new(3, vec![vec![0, 1, 2], vec![1], vec![2]]);
    ///
    /// assert_eq!(matrix.echelon_form(), expected);
    /// ```
    pub fn echelon_form(&self) -> Self {
        GaussJordan::new(self).echelon_form()
    }

    /// Returns a matrix for which the rows are the generators
    /// of the nullspace of the original matrix.
    ///
    /// The nullspace of a matrix M is the set of vectors N such that
    /// Mx = 0 for all x in N.
    /// Therefore, if N is the nullspace matrix obtain from this function,
    /// we have that M * N^T = 0.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(
    ///     6,
    ///     vec![vec![0, 1, 3, 5], vec![2, 3, 4], vec![2, 5], vec![0, 1, 3]],
    /// );
    ///
    /// let expected = SparseBinMat::new(6, vec![vec![0, 3, 4,], vec![0, 1]]);
    /// let nullspace = matrix.nullspace();
    ///
    /// assert_eq!(nullspace, expected);
    /// assert_eq!(&matrix * &nullspace.transposed(), SparseBinMat::zeros(4, 2));
    /// ```
    pub fn nullspace(&self) -> Self {
        nullspace(self)
    }

    /// Returns the horizontal concatenation of two matrices.
    ///
    /// If the matrix have different number of rows, the smallest
    /// one is padded with empty rows.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let left_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let right_matrix = SparseBinMat::new(4, vec![vec![1, 2, 3], vec![0, 1], vec![2, 3]]);
    ///
    /// let concatened = left_matrix.horizontal_concat_with(&right_matrix);
    ///
    /// let expected = SparseBinMat::new(7, vec![vec![0, 1, 4, 5, 6], vec![1, 2, 3, 4], vec![5, 6]]);
    ///
    /// assert_eq!(concatened, expected);
    /// ```
    pub fn horizontal_concat_with(&self, other: &SparseBinMat) -> SparseBinMat {
        concat_horizontally(self, other)
    }

    /// Returns the vertical concatenation of two matrices.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let left_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let right_matrix = SparseBinMat::identity(3);
    ///
    /// let concatened = left_matrix.vertical_concat_with(&right_matrix);
    ///
    /// let expected = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0], vec![1], vec![2]]);
    ///
    /// assert_eq!(concatened, expected);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if the matrices have a different number of columns.
    pub fn vertical_concat_with(&self, other: &SparseBinMat) -> SparseBinMat {
        if self.number_of_columns() != other.number_of_columns() {
            panic!(
                "{} and {} matrices can't be concatenated vertically",
                dimension_to_string(self.dimension()),
                dimension_to_string(other.dimension()),
            );
        }
        concat_vertically(self, other)
    }

    /// Returns a new matrix keeping only the given rows.
    ///
    /// # Example
    ///
    /// ```
    /// use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![2, 3, 4],
    ///     vec![0, 2, 4],
    ///     vec![1, 3],
    /// ]);
    ///
    /// let truncated = SparseBinMat::new(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 2, 4],
    /// ]);
    ///
    /// assert_eq!(matrix.keep_only_rows(&[0, 2]), truncated);
    /// assert_eq!(matrix.keep_only_rows(&[0, 2, 3]).number_of_rows(), 3);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some rows are out of bound.
    pub fn keep_only_rows(&self, rows: &[usize]) -> Self {
        self.assert_rows_are_inbound(rows);
        let rows = self
            .rows()
            .enumerate()
            .filter(|(index, _)| rows.contains(index))
            .map(|(_, row)| row.non_trivial_positions().collect())
            .collect();
        Self::new(self.number_of_columns(), rows)
    }

    /// Returns a truncated matrix where the given rows are removed.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![2, 3, 4],
    ///     vec![0, 2, 4],
    ///     vec![1, 3],
    /// ]);
    ///
    /// let truncated = SparseBinMat::new(5, vec![
    ///     vec![2, 3, 4],
    ///     vec![1, 3],
    /// ]);
    ///
    /// assert_eq!(matrix.without_rows(&[0, 2]), truncated);
    /// assert_eq!(matrix.without_rows(&[1, 2, 3]).number_of_rows(), 1);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some rows are out of bound.
    pub fn without_rows(&self, rows: &[usize]) -> Self {
        let to_keep: Vec<usize> = (0..self.number_of_rows())
            .filter(|x| !rows.contains(x))
            .collect();
        self.keep_only_rows(&to_keep)
    }

    fn assert_rows_are_inbound(&self, rows: &[usize]) {
        for row in rows {
            if *row >= self.number_of_columns() {
                panic!(
                    "row {} is out of bound for {} matrix",
                    row,
                    dimension_to_string(self.dimension())
                );
            }
        }
    }
    /// Returns a new matrix keeping only the given columns.
    ///
    /// Columns are relabeled to the fit new number of columns.
    ///
    /// # Example
    ///
    /// ```
    /// use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![2, 3, 4],
    ///     vec![0, 2, 4],
    ///     vec![1, 3],
    /// ]);
    ///
    /// let truncated = SparseBinMat::new(3, vec![
    ///     vec![0, 1],
    ///     vec![2],
    ///     vec![0, 2],
    ///     vec![1],
    /// ]);
    ///
    /// assert_eq!(matrix.keep_only_columns(&[0, 1, 4]), truncated);
    /// assert_eq!(matrix.keep_only_columns(&[1, 2]).number_of_columns(), 2);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some columns are out of bound.
    pub fn keep_only_columns(&self, columns: &[usize]) -> Self {
        self.assert_columns_are_inbound(columns);
        let old_to_new_column_map = columns
            .iter()
            .enumerate()
            .map(|(new, old)| (old, new))
            .collect::<HashMap<_, _>>();
        let rows = self
            .rows()
            .map(|row| {
                row.non_trivial_positions()
                    .filter_map(|column| old_to_new_column_map.get(&column).cloned())
                    .collect()
            })
            .collect();
        Self::new(columns.len(), rows)
    }

    /// Returns a truncated matrix where the given columns are removed.
    ///
    /// Columns are relabeled to fit the new number of columns.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(5, vec![
    ///     vec![0, 1, 2],
    ///     vec![2, 3, 4],
    ///     vec![0, 2, 4],
    ///     vec![1, 3],
    /// ]);
    ///
    /// let truncated = SparseBinMat::new(3, vec![
    ///     vec![0],
    ///     vec![1, 2],
    ///     vec![2],
    ///     vec![0, 1],
    /// ]);
    ///
    /// assert_eq!(matrix.without_columns(&[0, 2]), truncated);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if some columns are out of bound.
    pub fn without_columns(&self, columns: &[usize]) -> Self {
        let to_keep: Vec<usize> = (0..self.number_of_columns)
            .filter(|x| !columns.contains(x))
            .collect();
        self.keep_only_columns(&to_keep)
    }

    fn assert_columns_are_inbound(&self, columns: &[usize]) {
        for column in columns {
            if *column >= self.number_of_columns() {
                panic!(
                    "column {} is out of bound for {} matrix",
                    column,
                    dimension_to_string(self.dimension())
                );
            }
        }
    }
}

impl Add<&SparseBinMat> for &SparseBinMat {
    type Output = SparseBinMat;

    fn add(self, other: &SparseBinMat) -> SparseBinMat {
        if self.dimension() != other.dimension() {
            panic!(
                "{} and {} matrices can't be added",
                dimension_to_string(self.dimension()),
                dimension_to_string(other.dimension()),
            );
        }
        let rows = self
            .rows()
            .zip(other.rows())
            .map(|(row, other_row)| &row + &other_row)
            .map(|sum| sum.non_trivial_positions().collect_vec())
            .collect();
        SparseBinMat::new(self.number_of_columns(), rows)
    }
}

impl Mul<&SparseBinMat> for &SparseBinMat {
    type Output = SparseBinMat;

    fn mul(self, other: &SparseBinMat) -> SparseBinMat {
        if self.number_of_columns() != other.number_of_rows() {
            panic!(
                "{} and {} matrices can't be multiplied",
                dimension_to_string(self.dimension()),
                dimension_to_string(other.dimension()),
            );
        }
        let other_transposed = other.transposed();
        let rows = self
            .rows()
            .map(|row| {
                other_transposed
                    .rows()
                    .positions(|column| row.dot_with(&column) == 1)
                    .collect()
            })
            .collect();
        SparseBinMat::new(other.number_of_columns(), rows)
    }
}

fn dimension_to_string(dimension: (usize, usize)) -> String {
    format!("({} x {})", dimension.0, dimension.1)
}

//impl std::fmt::Display for SparseBinMat {
//fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//for row in self.rows() {
//write!(f, "[ ")?;
//for column in row.iter() {
//write!(f, "{} ", column)?;
//}
//write!(f, "]")?;
//}
//Ok(())
//}
//}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rows_are_sorted_on_construction() {
        let rows = vec![vec![1, 0], vec![0, 2, 1], vec![1, 2, 3]];
        let matrix = SparseBinMat::new(4, rows);

        assert_eq!(matrix.row(0).unwrap().as_slice(), &[0, 1]);
        assert_eq!(matrix.row(1).unwrap().as_slice(), &[0, 1, 2]);
        assert_eq!(matrix.row(2).unwrap().as_slice(), &[1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn panics_on_construction_if_rows_are_out_of_bound() {
        let rows = vec![vec![0, 1, 5], vec![2, 3, 4]];
        SparseBinMat::new(5, rows);
    }

    #[test]
    fn addition() {
        let first_matrix = SparseBinMat::new(6, vec![vec![0, 2, 4], vec![1, 3, 5]]);
        let second_matrix = SparseBinMat::new(6, vec![vec![0, 1, 2], vec![3, 4, 5]]);
        let sum = SparseBinMat::new(6, vec![vec![1, 4], vec![1, 4]]);
        assert_eq!(&first_matrix + &second_matrix, sum);
    }

    #[test]
    fn panics_on_addition_if_different_dimensions() {
        let matrix_6_2 = SparseBinMat::new(6, vec![vec![0, 2, 4], vec![1, 3, 5]]);
        let matrix_6_3 = SparseBinMat::new(6, vec![vec![0, 1, 2], vec![3, 4, 5], vec![0, 3]]);
        let matrix_2_2 = SparseBinMat::new(2, vec![vec![0], vec![1]]);

        let result = std::panic::catch_unwind(|| &matrix_6_2 + &matrix_6_3);
        assert!(result.is_err());

        let result = std::panic::catch_unwind(|| &matrix_6_2 + &matrix_2_2);
        assert!(result.is_err());

        let result = std::panic::catch_unwind(|| &matrix_6_3 + &matrix_2_2);
        assert!(result.is_err());
    }

    #[test]
    fn multiplication_with_other_matrix() {
        let first_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
        let second_matrix = SparseBinMat::new(5, vec![vec![0, 2], vec![1, 3], vec![2, 4]]);
        let product = SparseBinMat::new(5, vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]]);
        assert_eq!(&first_matrix * &second_matrix, product);
    }

    #[test]
    fn panics_on_matrix_multiplication_if_wrong_dimension() {
        let matrix_6_3 = SparseBinMat::new(6, vec![vec![0, 1, 2], vec![3, 4, 5], vec![0, 3]]);
        let matrix_2_2 = SparseBinMat::new(2, vec![vec![0], vec![1]]);
        let result = std::panic::catch_unwind(|| &matrix_6_3 * &matrix_2_2);
        assert!(result.is_err());
    }
}
