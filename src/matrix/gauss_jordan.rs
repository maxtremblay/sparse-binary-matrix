use super::SparseBinMat;
use crate::SparseBinSlice;

pub(super) struct GaussJordan {
    number_of_columns: usize,
    active_column: usize,
    rows: Vec<Vec<usize>>,
}

impl GaussJordan {
    pub(super) fn new(matrix: &SparseBinMat) -> Self {
        Self {
            number_of_columns: matrix.number_of_columns(),
            active_column: 0,
            rows: matrix
                .rows()
                .map(|row| row.non_trivial_positions().collect())
                .collect(),
        }
    }

    pub(super) fn rank(self) -> usize {
        self.unsorted_echeloned_rows().len()
    }

    pub(super) fn echelon_form(self) -> SparseBinMat {
        let number_of_columns = self.number_of_columns;
        let mut rows = self.unsorted_echeloned_rows();
        rows.sort_by_key(|row| row[0]);
        SparseBinMat::new(number_of_columns, rows)
    }

    fn unsorted_echeloned_rows(mut self) -> Vec<Vec<usize>> {
        while self.is_not_in_echelon_form() {
            self.pivot_active_column();
            self.go_to_next_column();
        }
        self.rows
    }

    fn is_not_in_echelon_form(&self) -> bool {
        self.active_column < self.number_of_columns
    }

    fn pivot_active_column(&mut self) {
        if let Some(pivot) = self.find_and_remove_pivot() {
            self.pivot_rows_that_start_in_active_column_with(&pivot);
            self.rows.push(pivot);
        }
    }

    fn find_and_remove_pivot(&mut self) -> Option<Vec<usize>> {
        let mut row_index = 0;
        while row_index < self.rows.len() {
            if self.row_at_index_start_at_active_column(row_index) {
                let row = self.get_and_remove_row_at_index(row_index);
                return Some(row);
            }
            row_index += 1;
        }
        None
    }

    fn row_at_index_start_at_active_column(&self, index: usize) -> bool {
        self.rows[index]
            .first()
            .map(|column| *column == self.active_column)
            .unwrap_or(false)
    }

    fn get_and_remove_row_at_index(&mut self, index: usize) -> Vec<usize> {
        self.rows.swap_remove(index)
    }

    fn pivot_rows_that_start_in_active_column_with(&mut self, pivot: &[usize]) {
        let mut row_index = 0;
        while row_index < self.rows.len() {
            if self.row_at_index_start_at_active_column(row_index) {
                self.rows[row_index] = (&SparseBinSlice::new(self.number_of_columns, pivot)
                    + &SparseBinSlice::new(self.number_of_columns, &self.rows[row_index]))
                    .to_positions_vec();
                if self.rows[row_index].is_empty() {
                    self.rows.swap_remove(row_index);
                    continue;
                }
            }
            row_index += 1;
        }
    }

    fn go_to_next_column(&mut self) {
        self.active_column += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ranks() {
        let matrix = SparseBinMat::empty();
        let rank = GaussJordan::new(&matrix).rank();
        assert_eq!(rank, 0);

        let rows = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4], vec![0, 4]];
        let matrix = SparseBinMat::new(5, rows);
        let rank = GaussJordan::new(&matrix).rank();
        assert_eq!(rank, 4);

        let rows = vec![vec![0, 1], vec![1, 2], vec![0, 1, 3]];
        let matrix = SparseBinMat::new(4, rows);
        let rank = GaussJordan::new(&matrix).rank();
        assert_eq!(rank, 3);

        let rows = vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![0, 3],
            vec![3, 4, 5],
            vec![0, 4, 6],
            vec![5, 6],
        ];
        let matrix = SparseBinMat::new(7, rows);
        let rank = GaussJordan::new(&matrix).rank();
        assert_eq!(rank, 4);
    }

    #[test]
    fn do_nothing_if_already_in_echelon_form() {
        let rows = vec![vec![0, 1, 2], vec![1, 2, 3], vec![3, 4, 5], vec![5, 6]];
        let matrix = SparseBinMat::new(7, rows);
        let echelon_form = GaussJordan::new(&matrix).echelon_form();
        assert_eq!(echelon_form, matrix);
    }

    #[test]
    fn compute_the_good_echelon_form() {
        let rows = vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![0, 3],
            vec![3, 4, 5],
            vec![0, 4, 6],
            vec![5, 6],
        ];
        let matrix = SparseBinMat::new(7, rows);
        let echelon_form = GaussJordan::new(&matrix).echelon_form();
        let expected = SparseBinMat::new(
            7,
            vec![vec![0, 1, 2], vec![1, 2, 3], vec![3, 4, 6], vec![5, 6]],
        );
        assert_eq!(echelon_form, expected);
    }
}
