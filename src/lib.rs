//! A sparse implementation of a binary matrix optimized for row operations.
//!
//! The main objects of this crate are [`matrices`](crate::SparseBinMat) and [`vectors`](crate::SparseBinVecBase).
//! All elements in a binary matrix or vector are element of the binary field GF2.
//! That is, they are either 0 or 1 and addition is modulo 2.
//!
//! # Quick start
//!
//! To instanciate a matrix, you need to specify the number of columns as well
//! as the position of 1 in each rows.
//!
//! ```
//! use sparse_bin_mat::SparseBinMat;
//!
//! // This is the matrix
//! // 1 0 1 0 1
//! // 0 1 0 1 0
//! // 0 0 1 0 0
//! let matrix = SparseBinMat::new(5, vec![vec![0, 2, 4], vec![1, 3], vec![2]]);
//! ```
//!
//! It is easy to access elements or rows of a matrix. However,
//! since the matrix are optimized for row operations, you need
//! to transpose the matrix if you want to perform column operations.
//!
//! ```
//! # use sparse_bin_mat::SparseBinMat;
//! let matrix = SparseBinMat::new(5, vec![vec![0, 2, 4], vec![1, 3], vec![2]]);
//! assert_eq!(matrix.row(1).unwrap().as_slice(), [1, 3].as_ref());
//! assert_eq!(matrix.get(0, 0), Some(1));
//! assert_eq!(matrix.get(0, 1), Some(0));
//! // The element (0, 7) is out of bound for a 3 x 5 matrix.
//! assert_eq!(matrix.get(0, 7), None);
//! ```
//!
//! Adition and multiplication are implemented between matrix references.
//!
//! ```
//! # use sparse_bin_mat::SparseBinMat;
//! let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
//! let identity = SparseBinMat::identity(3);
//!
//! let sum = SparseBinMat::new(3, vec![vec![1], vec![2], vec![0]]);
//! assert_eq!(&matrix + &identity, sum);
//!
//! assert_eq!(&matrix * &identity, matrix);
//! ```
//!
//! Many useful operations and decompositions are implemented.
//! These include, but are not limited to
//! - [`rank`](SparseBinMat::rank),
//! - [`echelon form`](SparseBinMat::echelon_form),
//! - [`nullspace`](SparseBinMat::nullspace),
//! - [`tranposition`](SparseBinMat::transposed),
//! - [`horizontal`](SparseBinMat::horizontal_concat_with) and
//! [`vertical`](SparseBinMat::vertical_concat_with) concatenations,
//! - and more ...
//!
//! Operations are implemented as I need them,
//! feel welcome to raise an issue if you need a new functionnality.

mod matrix;
pub use matrix::{Rows, SparseBinMat};

mod vector;
pub use vector::{SparseBinSlice, SparseBinVec, SparseBinVecBase};

type BinaryNumber = u8;
