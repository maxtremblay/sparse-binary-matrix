use super::SparseBinMat;

pub(super) fn transpose(matrix: &SparseBinMat) -> SparseBinMat {
    if matrix.is_empty() {
        println!("{}",  matrix.number_of_columns());
        return SparseBinMat::new(0, vec![vec![]; matrix.number_of_columns()]);
    }
    let mut transposed = vec![Vec::new(); matrix.number_of_columns()];
    for (row, positions) in matrix.rows().enumerate() {
        insert_positions_into(row, positions.as_slice(), &mut transposed);
    }
    SparseBinMat::new(matrix.number_of_rows(), transposed)
}

fn insert_positions_into(row: usize, positions: &[usize], transpose: &mut Vec<Vec<usize>>) {
    positions
        .iter()
        .for_each(|position| transpose[*position].push(row));
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transposition_of_empty_matrix() {
        let transposed = SparseBinMat::empty().transposed();
        assert!(transposed.is_empty());
    }

    #[test]
    fn transposition_of_general_matrix() {
        let matrix = SparseBinMat::new(
            5,
            vec![vec![0, 1, 4], vec![2, 3], vec![1, 3, 4], vec![0, 2]],
        );

        let transposed = SparseBinMat::new(
            4,
            vec![vec![0, 3], vec![0, 2], vec![1, 3], vec![1, 2], vec![0, 2]],
        );

        assert_eq!(matrix.transposed(), transposed);
    }

    #[test]
    fn tranposition_with_some_empty_transposed_rows() {
        let matrix = SparseBinMat::new(5, vec![vec![0, 1, 4], vec![2, 4], vec![0, 1, 2]]);

        let transposed = SparseBinMat::new(
            3,
            vec![vec![0, 2], vec![0, 2], vec![1, 2], vec![], vec![0, 1]],
        );

        assert_eq!(matrix.transposed(), transposed);
    }

    #[test]
    fn transpose_matrix_without_rows() {
        let matrix = SparseBinMat::new(5, vec![]);
        let expected = SparseBinMat::new(0, vec![vec![]; 5]);
        assert_eq!(matrix.transposed(), expected)
    }
}
