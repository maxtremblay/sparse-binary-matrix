use super::Matrix;
use itertools::EitherOrBoth;
use itertools::Itertools;

pub(super) fn concat_horizontally(left_matrix: &Matrix, right_matrix: &Matrix) -> Matrix {
    let number_of_columns = left_matrix.number_of_columns() + right_matrix.number_of_columns();
    let rows = get_rows_of_horizontal_concat(left_matrix, right_matrix);
    Matrix::new(number_of_columns, rows)
}

fn get_rows_of_horizontal_concat(left_matrix: &Matrix, right_matrix: &Matrix) -> Vec<Vec<usize>> {
    left_matrix
        .rows()
        .zip_longest(right_matrix.rows())
        .map(|rows| {
            rows.map_left(|row| row.to_vec())
                .map_right(|row| pad_right_row(left_matrix.number_of_columns(), row))
        })
        .map(|rows| concat_rows_horizontally(rows))
        .collect()
}

fn concat_rows_horizontally(rows: EitherOrBoth<Vec<usize>, Vec<usize>>) -> Vec<usize> {
    match rows {
        EitherOrBoth::Both(left_row, right_row) => concat(left_row, right_row),
        EitherOrBoth::Left(row) => row,
        EitherOrBoth::Right(row) => row,
    }
}

fn concat(mut left_row: Vec<usize>, mut right_row: Vec<usize>) -> Vec<usize> {
    left_row.append(&mut right_row);
    left_row
}

fn pad_right_row(pad: usize, row: &[usize]) -> Vec<usize> {
    row.iter().map(|position| position + pad).collect()
}

// This assume that both matrix have the same number of columns.
pub(super) fn concat_vertically(top_matrix: &Matrix, bottom_matrix: &Matrix) -> Matrix {
    let rows = top_matrix
        .rows()
        .chain(bottom_matrix.rows())
        .map(|row| row.to_vec())
        .collect();
    Matrix::new(top_matrix.number_of_columns(), rows)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn horizontal_concat_with_empty_matrix() {
        let left_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let right_matrix = Matrix::empty();
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        assert_eq!(concatened, left_matrix);
    }

    #[test]
    fn horizontal_concat_from_empty_matrix() {
        let left_matrix = Matrix::empty();
        let right_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        assert_eq!(concatened, right_matrix);
    }

    #[test]
    fn horizontal_concat_with_smaller_left_matrix() {
        let left_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let right_matrix = Matrix::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected = Matrix::new(7, vec![vec![0, 1, 4, 5], vec![1, 2, 3, 5, 6], vec![4, 6]]);
        assert_eq!(concatened, expected);
    }

    #[test]
    fn horizontal_concat_with_smaller_right_matrix() {
        let left_matrix = Matrix::new(3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let right_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected = Matrix::new(7, vec![vec![0, 1, 3, 4], vec![1, 2, 4, 5, 6], vec![0, 2]]);
        assert_eq!(concatened, expected);
    }

    #[test]
    fn horizontal_concat_with_equal_length_matrices() {
        let left_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
        let right_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3], vec![0, 2, 3]]);
        let concatened = concat_horizontally(&left_matrix, &right_matrix);
        let expected = Matrix::new(
            8,
            vec![vec![0, 1, 4, 5], vec![1, 2, 5, 6, 7], vec![2, 3, 4, 6, 7]],
        );
        assert_eq!(concatened, expected);
    }

    #[test]
    fn vertical_concat() {
        let top_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2, 3]]);
        let bottom_matrix = Matrix::new(4, vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let concatened = concat_vertically(&top_matrix, &bottom_matrix);
        let expected = Matrix::new(
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

    //#[test]
    //fn diagonal_concat_with_empty_matrix() {
    //let left_matrix =
    //Matrix::with_block_size(4).with_checks(vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
    //let right_matrix = Matrix::new();

    //let concatened = Concatener::from(&left_matrix, &right_matrix).concat_diagonally();

    //assert_eq!(concatened, left_matrix);
    //}

    //#[test]
    //fn diagonal_concat_from_empty_matrix() {
    //let left_matrix = Matrix::new();

    //let right_matrix =
    //Matrix::with_block_size(4).with_checks(vec![vec![0, 1], vec![1, 2], vec![2, 3]]);

    //let concatened = Concatener::from(&left_matrix, &right_matrix).concat_diagonally();

    //assert_eq!(concatened, right_matrix);
    //}

    //#[test]
    //fn diagonal_concat() {
    //let left_matrix =
    //Matrix::with_block_size(4).with_checks(vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
    //let right_matrix = Matrix::with_block_size(4).with_checks(vec![vec![0, 1], vec![1, 2, 3]]);

    //let concatened = Concatener::from(&left_matrix, &right_matrix).concat_diagonally();
    //let expected = Matrix::with_block_size(8).with_checks(vec![
    //vec![0, 1],
    //vec![1, 2],
    //vec![2, 3],
    //vec![4, 5],
    //vec![5, 6, 7],
    //]);

    //assert_eq!(concatened, expected);
    //}
}
