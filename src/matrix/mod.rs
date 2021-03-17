use crate::error::{
    validate_positions, InvalidPositions, MatMatIncompatibleDimensions,
    MatVecIncompatibleDimensions,
};
use crate::BinaryNumber;
use crate::{SparseBinSlice, SparseBinVec, SparseBinVecBase};
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::{Add, Deref, Mul};

mod concat;
use concat::{concat_horizontally, concat_vertically};

mod constructor_utils;
use constructor_utils::initialize_from;

mod gauss_jordan;
use gauss_jordan::GaussJordan;

mod inplace_operations;
use inplace_operations::{insert_one_at, remove_one_at};

mod kronecker;
use kronecker::kronecker_product;

mod non_trivial_elements;
pub use self::non_trivial_elements::NonTrivialElements;

mod nullspace;
use nullspace::nullspace;

mod rows;
pub use self::rows::Rows;

mod transpose;
use transpose::transpose;

mod ser_de;

/// A sparse binary matrix optimized for row operations.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct SparseBinMat {
    row_ranges: Vec<usize>,
    column_indices: Vec<usize>,
    number_of_columns: usize,
}

impl SparseBinMat {
    /// Creates a new matrix with the given number of columns
    /// and list of rows .
    ///
    /// A row is a list of the positions where the elements have value 1.
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
    /// Panics if a position in a row is greater or equal the number of columns,
    /// a row is unsorted or a row contains duplicate.
    pub fn new(number_of_columns: usize, rows: Vec<Vec<usize>>) -> Self {
        Self::try_new(number_of_columns, rows).unwrap()
    }

    /// Creates a new matrix with the given number of columns
    /// and list of rows or returns an error if a position in a
    /// row is greater or equal the number of columns, a row is unsorted
    /// or a row contains duplicate.
    ///
    /// A row is a list of the positions where the elements have value 1.
    /// All rows are sorted during insertion.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let first_matrix = SparseBinMat::try_new(4, vec![vec![0, 1, 2], vec![0, 2, 3]]);
    /// let second_matrix = SparseBinMat::new(4, vec![vec![0, 1, 2], vec![0, 2, 3]]);
    /// assert_eq!(first_matrix, Ok(second_matrix));
    /// ```
    pub fn try_new(
        number_of_columns: usize,
        rows: Vec<Vec<usize>>,
    ) -> Result<Self, InvalidPositions> {
        for row in rows.iter() {
            validate_positions(number_of_columns, row)?;
        }
        Ok(Self::new_unchecked(number_of_columns, rows))
    }

    // Assumes rows are sorted, all unique and inbound.
    pub(crate) fn new_unchecked(number_of_columns: usize, rows: Vec<Vec<usize>>) -> Self {
        let (row_ranges, column_indices) = initialize_from(rows, None);
        Self {
            row_ranges,
            column_indices,
            number_of_columns,
        }
    }

    /// Creates a new matrix with the given number of columns,
    /// capacity and list of rows.
    ///
    /// A row is a list of the positions where the elements have value 1.
    ///
    /// The capacity is used to pre-allocate enough memory to store that
    /// amount of 1s in the matrix.
    /// This is mostly useful in combination with inplace operations modifying
    /// the number of 1s in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::with_capacity(4, 7, vec![vec![0, 1, 2], vec![0, 2, 3]]);
    ///
    /// assert_eq!(matrix.number_of_rows(), 2);
    /// assert_eq!(matrix.number_of_columns(), 4);
    /// assert_eq!(matrix.number_of_elements(), 8);
    /// assert_eq!(matrix.capacity(), 7);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if a position in a row is greater or equal the number of columns,
    /// a row is unsorted or a row contains duplicate.
    pub fn with_capacity(number_of_columns: usize, capacity: usize, rows: Vec<Vec<usize>>) -> Self {
        Self::try_with_capacity(number_of_columns, capacity, rows).expect("[Error]")
    }

    /// Creates a new matrix with the given number of columns,
    /// capacity and list of rows
    /// Returns an error if a position in a
    /// row is greater or equal the number of columns, a row is unsorted
    /// or a row contains duplicate.
    ///
    /// A row is a list of the positions where the elements have value 1.
    /// All rows are sorted during insertion.
    ///
    /// The capacity is used to pre-allocate enough memory to store that
    /// amount of 1s in the matrix.
    /// This is mostly useful in combination with inplace operations modifying
    /// the number of 1s in the matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(4, vec![vec![0, 1, 2], vec![0, 2, 3]]);
    /// let try_matrix = SparseBinMat::try_with_capacity(4, 7, vec![vec![0, 1 ,2], vec![0, 2, 3]]);
    ///
    /// assert_eq!(try_matrix, Ok(matrix));
    /// assert_eq!(try_matrix.unwrap().capacity(), 7);
    /// ```
    pub fn try_with_capacity(
        number_of_columns: usize,
        capacity: usize,
        rows: Vec<Vec<usize>>,
    ) -> Result<Self, InvalidPositions> {
        for row in rows.iter() {
            validate_positions(number_of_columns, row)?;
        }
        let (row_ranges, column_indices) = initialize_from(rows, Some(capacity));
        Ok(Self {
            row_ranges,
            column_indices,
            number_of_columns,
        })
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
        Self::new_unchecked(number_of_columns, vec![Vec::new(); number_of_rows])
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

    /// Returns the maximum number of 1s the matrix can
    /// store before reallocating.
    pub fn capacity(&self) -> usize {
        self.column_indices.capacity()
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

    /// Returns true if the all elements of the matrix are 0.
    pub fn is_zero(&self) -> bool {
        self.number_of_ones() == 0
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

    /// Inserts the given value at the given row and column.
    ///
    /// This operation is perform in place. That is, it take ownership of the matrix
    /// and returns an updated matrix using the same memory.
    ///
    /// To avoid having to reallocate new memory,
    /// it is reccommended to construct the matrix using
    /// [`with_capacity`](SparseBinMat::with_capacity)
    /// or
    /// [`try_with_capacity`](SparseBinMat::try_with_capacity).
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    ///
    /// let mut matrix = SparseBinMat::with_capacity(3, 4, vec![vec![0], vec![1, 2]]);
    ///
    /// // This doesn't change the matrix
    /// matrix = matrix.emplace_at(1, 0, 0);
    /// assert_eq!(matrix.number_of_ones(), 3);
    ///
    /// // Add a 1 in the first row.
    /// matrix = matrix.emplace_at(1, 0, 2);
    /// let expected = SparseBinMat::new(3, vec![vec![0, 2], vec![1, 2]]);
    /// assert_eq!(matrix, expected);
    ///
    /// // Remove a 1 in the second row.
    /// matrix = matrix.emplace_at(0, 1, 1);
    /// let expected = SparseBinMat::new(3, vec![vec![0, 2], vec![2]]);
    /// assert_eq!(matrix, expected);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if either the row or column is out of bound.
    pub fn emplace_at(self, value: BinaryNumber, row: usize, column: usize) -> Self {
        if !(value == 0 || value == 1) {
            panic!("value must be 0 or 1")
        }
        match (self.get(row, column), value) {
            (None, _) => panic!(
                "position ({}, {}) is out of bound for {} matrix",
                row,
                column,
                dimension_to_string(self.dimension())
            ),
            (Some(0), 1) => insert_one_at(self, row, column),
            (Some(1), 0) => remove_one_at(self, row, column),
            _ => self,
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
    /// assert_eq!(matrix.row(0), Some(SparseBinSlice::new(3, &rows[0])));
    /// assert_eq!(matrix.row(1), Some(SparseBinSlice::new(3, &rows[1])));
    /// assert_eq!(matrix.row(2), None);
    /// ```
    pub fn row(&self, row: usize) -> Option<SparseBinSlice> {
        let row_start = self.row_ranges.get(row)?;
        let row_end = self.row_ranges.get(row + 1)?;
        Some(
            SparseBinSlice::try_new(
                self.number_of_columns(),
                &self.column_indices[*row_start..*row_end],
            )
            .unwrap(),
        )
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
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new(7, &rows[0])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new(7, &rows[1])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new(7, &rows[2])));
    /// assert_eq!(iter.next(), Some(SparseBinSlice::new(7, &rows[3])));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn rows(&self) -> Rows {
        Rows::from(self)
    }

    /// Returns an iterator yielding the positions of the non
    /// trivial elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::{SparseBinSlice, SparseBinMat};
    /// let rows = vec![vec![0, 2], vec![1], vec![0, 2], vec![1]];
    /// let matrix = SparseBinMat::new(3, rows);
    ///
    /// let mut iter = matrix.non_trivial_elements();
    ///
    /// assert_eq!(iter.next(), Some((0, 0)));
    /// assert_eq!(iter.next(), Some((0, 2)));
    /// assert_eq!(iter.next(), Some((1, 1)));
    /// assert_eq!(iter.next(), Some((2, 0)));
    /// assert_eq!(iter.next(), Some((2, 2)));
    /// assert_eq!(iter.next(), Some((3, 1)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn non_trivial_elements(&self) -> NonTrivialElements {
        NonTrivialElements::new(self)
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
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1, 2], vec![0], vec![1, 2], vec![0, 2]]);
    /// let expected = SparseBinMat::new(3, vec![vec![0, 1, 2], vec![1], vec![2]]);
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
    /// If the matrix have different number of columns, the smallest
    /// one is padded with empty columns.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let left_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let right_matrix = SparseBinMat::identity(3);
    ///
    /// let concatened = left_matrix.vertical_concat_with(&right_matrix);
    /// let expected = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0], vec![1], vec![2]]);
    ///
    /// assert_eq!(concatened, expected);
    /// ```
    pub fn vertical_concat_with(&self, other: &SparseBinMat) -> Self {
        concat_vertically(self, other)
    }

    /// Returns the dot product between a matrix and a vector or an error
    /// if the number of columns in the matrix is not equal to the length of
    /// the vector.
    ///
    /// Use the Mul (*) operator for a version that panics instead of
    /// returning a `Result`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::{SparseBinMat, SparseBinVec};
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let vector = SparseBinVec::new(3, vec![0, 1]);
    /// let result = SparseBinVec::new(2, vec![1]);
    ///
    /// assert_eq!(matrix.dot_with_vector(&vector), Ok(result));
    /// ```
    pub fn dot_with_vector<T>(
        &self,
        vector: &SparseBinVecBase<T>,
    ) -> Result<SparseBinVec, MatVecIncompatibleDimensions>
    where
        T: std::ops::Deref<Target = [usize]>,
    {
        if self.number_of_columns() != vector.len() {
            return Err(MatVecIncompatibleDimensions::new(
                self.dimension(),
                vector.len(),
            ));
        }
        let positions = self
            .rows()
            .map(|row| row.dot_with(vector).unwrap())
            .positions(|product| product == 1)
            .collect();
        Ok(SparseBinVec::new_unchecked(
            self.number_of_rows(),
            positions,
        ))
    }

    /// Returns the dot product between two matrices or an error
    /// if the number of columns in the first matrix is not equal
    /// to the number of rows in the second matrix.
    ///
    /// Use the Mul (*) operator for a version that panics instead of
    /// returning a `Result`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let other_matrix = SparseBinMat::new(4, vec![vec![1], vec![2], vec![3]]);
    /// let result = SparseBinMat::new(4, vec![vec![1, 2], vec![2, 3]]);
    ///
    /// assert_eq!(matrix.dot_with_matrix(&other_matrix), Ok(result));
    /// ```
    pub fn dot_with_matrix(
        &self,
        other: &Self,
    ) -> Result<SparseBinMat, MatMatIncompatibleDimensions> {
        if self.number_of_columns() != other.number_of_rows() {
            return Err(MatMatIncompatibleDimensions::new(
                self.dimension(),
                other.dimension(),
            ));
        }
        let transposed = other.transposed();
        let rows = self
            .rows()
            .map(|row| {
                transposed
                    .rows()
                    .positions(|column| row.dot_with(&column).unwrap() == 1)
                    .collect()
            })
            .collect();
        Ok(Self::new_unchecked(other.number_of_columns, rows))
    }

    /// Returns the bitwise xor sum of two matrices or an error
    /// if the matrices have different dimensions.
    ///
    /// Use the Add (+) operator for a version that panics instead
    /// of returning a `Result`.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::{SparseBinMat, SparseBinVec};
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let other_matrix = SparseBinMat::new(3, vec![vec![0, 1, 2], vec![0, 1, 2]]);
    /// let result = SparseBinMat::new(3, vec![vec![2], vec![0]]);
    ///
    /// assert_eq!(matrix.bitwise_xor_with(&other_matrix), Ok(result));
    /// ```
    pub fn bitwise_xor_with(&self, other: &Self) -> Result<Self, MatMatIncompatibleDimensions> {
        if self.dimension() != other.dimension() {
            return Err(MatMatIncompatibleDimensions::new(
                self.dimension(),
                other.dimension(),
            ));
        }
        let rows = self
            .rows()
            .zip(other.rows())
            .map(|(row, other_row)| row.bitwise_xor_with(&other_row).unwrap().to_positions_vec())
            .collect();
        Ok(SparseBinMat::new_unchecked(self.number_of_columns(), rows))
    }

    /// Returns a new matrix keeping only the given rows or an error
    /// if rows are out of bound, unsorted or not unique.
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
    /// assert_eq!(matrix.keep_only_rows(&[0, 2]), Ok(truncated));
    /// assert_eq!(matrix.keep_only_rows(&[0, 2, 3]).unwrap().number_of_rows(), 3);
    /// ```
    pub fn keep_only_rows(&self, rows: &[usize]) -> Result<Self, InvalidPositions> {
        validate_positions(self.number_of_rows(), rows)?;
        let rows = self
            .rows()
            .enumerate()
            .filter(|(index, _)| rows.contains(index))
            .map(|(_, row)| row.non_trivial_positions().collect())
            .collect();
        Ok(Self::new_unchecked(self.number_of_columns(), rows))
    }

    /// Returns a truncated matrix where the given rows are removed or an error
    /// if rows are out of bound or unsorted.
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
    /// assert_eq!(matrix.without_rows(&[0, 2]), Ok(truncated));
    /// assert_eq!(matrix.without_rows(&[1, 2, 3]).unwrap().number_of_rows(), 1);
    /// ```
    pub fn without_rows(&self, rows: &[usize]) -> Result<Self, InvalidPositions> {
        let to_keep: Vec<usize> = (0..self.number_of_rows())
            .filter(|x| !rows.contains(x))
            .collect();
        self.keep_only_rows(&to_keep)
    }

    /// Returns a new matrix keeping only the given columns or an error
    /// if columns are out of bound, unsorted or not unique.
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
    /// assert_eq!(matrix.keep_only_columns(&[0, 1, 4]), Ok(truncated));
    /// assert_eq!(matrix.keep_only_columns(&[1, 2]).unwrap().number_of_columns(), 2);
    /// ```
    pub fn keep_only_columns(&self, columns: &[usize]) -> Result<Self, InvalidPositions> {
        validate_positions(self.number_of_columns(), columns)?;
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
        Ok(Self::new_unchecked(columns.len(), rows))
    }

    /// Returns a truncated matrix where the given columns are removed or
    /// an error if columns are out of bound or unsorted.
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
    /// assert_eq!(matrix.without_columns(&[0, 2]), Ok(truncated));
    /// ```
    pub fn without_columns(&self, columns: &[usize]) -> Result<Self, InvalidPositions> {
        let to_keep: Vec<usize> = (0..self.number_of_columns)
            .filter(|x| !columns.contains(x))
            .collect();
        self.keep_only_columns(&to_keep)
    }

    /// Returns the Kronecker product of two matrices.
    ///
    /// # Example
    ///
    /// ```
    /// # use sparse_bin_mat::SparseBinMat;
    /// let left_matrix = SparseBinMat::new(2, vec![vec![1], vec![0]]);
    /// let right_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    ///
    /// let product = left_matrix.kron_with(&right_matrix);
    /// let expected = SparseBinMat::new(6, vec![vec![3, 4], vec![4, 5], vec![0, 1], vec![1, 2]]);
    ///
    /// assert_eq!(product, expected);
    /// ```
    pub fn kron_with(&self, other: &Self) -> Self {
        kronecker_product(self, other)
    }

    /// Returns a json string for the matrix.
    pub fn as_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

impl Add<&SparseBinMat> for &SparseBinMat {
    type Output = SparseBinMat;

    fn add(self, other: &SparseBinMat) -> SparseBinMat {
        self.bitwise_xor_with(other).expect(&format!(
            "{} and {} matrices can't be added",
            dimension_to_string(self.dimension()),
            dimension_to_string(other.dimension()),
        ))
    }
}

impl Mul<&SparseBinMat> for &SparseBinMat {
    type Output = SparseBinMat;

    fn mul(self, other: &SparseBinMat) -> SparseBinMat {
        self.dot_with_matrix(other).expect(&format!(
            "{} and {} matrices can't be multiplied",
            dimension_to_string(self.dimension()),
            dimension_to_string(other.dimension()),
        ))
    }
}

impl<T: Deref<Target = [usize]>> Mul<&SparseBinVecBase<T>> for &SparseBinMat {
    type Output = SparseBinVec;

    fn mul(self, other: &SparseBinVecBase<T>) -> SparseBinVec {
        self.dot_with_vector(other).expect(&format!(
            "{} matrix can't be multiplied with vector of length {}",
            dimension_to_string(self.dimension()),
            other.len()
        ))
    }
}

fn dimension_to_string(dimension: (usize, usize)) -> String {
    format!("({} x {})", dimension.0, dimension.1)
}

impl std::fmt::Display for SparseBinMat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.rows() {
            writeln!(f, "{}", row)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
