use super::SparseBinMat;
use crate::SparseBinSlice;

pub(super) fn kronecker_product(
    left_matrix: &SparseBinMat,
    right_matrix: &SparseBinMat,
) -> SparseBinMat {
    let rows = left_matrix
        .rows()
        .flat_map(|row| kron_row(row, right_matrix))
        .collect();
    let number_of_columns = left_matrix.number_of_columns * right_matrix.number_of_columns();
    SparseBinMat::new(number_of_columns, rows)
}

fn kron_row<'a>(
    left_row: SparseBinSlice<'a>,
    right_matrix: &'a SparseBinMat,
) -> impl Iterator<Item = Vec<usize>> + 'a {
    right_matrix.rows().map(move |right_row| {
        left_row
            .non_trivial_positions()
            .flat_map(|position| pad_row(position * right_row.len(), &right_row))
            .collect()
    })
}

fn pad_row<'a>(pad: usize, row: &'a SparseBinSlice<'a>) -> impl Iterator<Item = usize> + 'a {
    row.non_trivial_positions()
        .map(move |position| position + pad)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn left_kron_with_identity() {
        let matrix = SparseBinMat::new(4, vec![vec![0, 2], vec![1, 3]]);
        let product = matrix.kron_with(&SparseBinMat::identity(2));
        let expected = SparseBinMat::new(8, vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]);
        assert_eq!(product, expected);
    }

    #[test]
    fn right_kron_with_identity() {
        let matrix = SparseBinMat::new(4, vec![vec![0, 2], vec![1, 3]]);
        let product = SparseBinMat::identity(2).kron_with(&matrix);
        let expected = SparseBinMat::new(8, vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]);
        assert_eq!(product, expected);
    }

    #[test]
    fn kron_with_itself() {
        let matrix = SparseBinMat::new(4, vec![vec![0, 2], vec![1, 3]]);
        let product = matrix.kron_with(&matrix);
        let expected = SparseBinMat::new(
            16,
            vec![
                vec![0, 2, 8, 10],
                vec![1, 3, 9, 11],
                vec![4, 6, 12, 14],
                vec![5, 7, 13, 15],
            ],
        );
        assert_eq!(product, expected);
    }
}
