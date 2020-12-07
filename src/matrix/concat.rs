use super::SparseBinMat;
use super::SparseBinSlice;
use itertools::EitherOrBoth;
use itertools::Itertools;

pub(super) fn concat_horizontally(
    left_matrix: &SparseBinMat,
    right_matrix: &SparseBinMat,
) -> SparseBinMat {
    let number_of_columns = left_matrix.number_of_columns() + right_matrix.number_of_columns();
    let rows = get_rows_of_horizontal_concat(left_matrix, right_matrix);
    SparseBinMat::new(number_of_columns, rows)
}

fn get_rows_of_horizontal_concat(
    left_matrix: &SparseBinMat,
    right_matrix: &SparseBinMat,
) -> Vec<Vec<usize>> {
    left_matrix
        .rows()
        .zip_longest(right_matrix.rows())
        .map(|rows| concat_rows_horizontally(left_matrix.number_of_columns(), rows))
        .collect()
}

fn concat_rows_horizontally(
    pad: usize,
    rows: EitherOrBoth<SparseBinSlice, SparseBinSlice>,
) -> Vec<usize> {
    match rows {
        EitherOrBoth::Both(left_row, right_row) => left_row.concat(&right_row).to_positions_vec(),
        EitherOrBoth::Left(row) => row.to_owned().to_positions_vec(),
        EitherOrBoth::Right(row) => pad_right_row(pad, row.as_slice()),
    }
}

fn pad_right_row(pad: usize, row: &[usize]) -> Vec<usize> {
    row.iter().map(|position| position + pad).collect()
}

// This assume that both matrix have the same number of columns.
pub(super) fn concat_vertically(
    top_matrix: &SparseBinMat,
    bottom_matrix: &SparseBinMat,
) -> SparseBinMat {
    let rows = top_matrix
        .rows()
        .chain(bottom_matrix.rows())
        .map(|row| row.to_owned().to_positions_vec())
        .collect();
    let number_of_columns = std::cmp::max(
        top_matrix.number_of_columns(),
        bottom_matrix.number_of_columns(),
    );
    SparseBinMat::new(number_of_columns, rows)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn horizontal_concat_with_empty_matrix() {
        let left_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let right_matrix = SparseBinMat::empty();
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        assert_eq!(concatened, left_matrix);
    }

    #[test]
    fn horizontal_concat_from_empty_matrix() {
        let left_matrix = SparseBinMat::empty();
        let right_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        assert_eq!(concatened, right_matrix);
    }

    #[test]
    fn horizontal_concat_with_smaller_left_matrix() {
        let left_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let right_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected =
            SparseBinMat::new(7, vec![vec![0, 1, 4, 5], vec![1, 2, 3, 5, 6], vec![4, 6]]);
        assert_eq!(concatened, expected);
    }

    #[test]
    fn horizontal_concat_with_smaller_right_matrix() {
        let left_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let right_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected =
            SparseBinMat::new(7, vec![vec![0, 1, 3, 4], vec![1, 2, 4, 5, 6], vec![0, 2]]);
        assert_eq!(concatened, expected);
    }

    #[test]
    fn horizontal_concat_with_equal_length_matrices() {
        let left_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
        let right_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3], vec![0, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected = SparseBinMat::new(
            8,
            vec![vec![0, 1, 4, 5], vec![1, 2, 5, 6, 7], vec![2, 3, 4, 6, 7]],
        );
        assert_eq!(concatened, expected);
    }

    #[test]
    fn vertical_concat_with_smaller_bottom_matrix() {
        let top_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let bottom_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let concatened = concat_vertically(&top_matrix, &bottom_matrix);
        let expected = SparseBinMat::new(
            4,
            vec![
                vec![0, 1],
                vec![1, 2, 3],
                vec![0, 1],
                vec![1, 2],
                vec![0, 2],
            ],
        );
        assert_eq!(concatened, expected);
    }

    #[test]
    fn vertical_concat_with_smaller_top_matrix() {
        let left_matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let right_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let concatened = concat_vertically(&left_matrix, &right_matrix);
        let expected = SparseBinMat::new(
            4,
            vec![
                vec![0, 1],
                vec![1, 2],
                vec![0, 2],
                vec![0, 1],
                vec![1, 2, 3],
            ],
        );
        assert_eq!(concatened, expected);
    }

    #[test]
    fn vertical_concatwith_equal_length_matrices() {
        let top_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let bottom_matrix = SparseBinMat::new(4, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let concatened = concat_vertically(&top_matrix, &bottom_matrix);
        let expected = SparseBinMat::new(
            4,
            vec![
                vec![0, 1],
                vec![1, 2, 3],
                vec![0, 1],
                vec![1, 2],
                vec![0, 2],
            ],
        );
        assert_eq!(concatened, expected);
    }
}
