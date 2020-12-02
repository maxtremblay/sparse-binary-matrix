use super::BinaryNumber;
use std::cmp::Ordering;

pub(super) fn rows_bitwise_sum(first_row: &[usize], second_row: &[usize]) -> Vec<usize> {
    RowBitwiseZipIter::new(first_row, second_row)
        .filter_map(|x| {
            if x.first_row_value ^ x.second_row_value == 1 {
                Some(x.position)
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn rows_dot_product(first_row: &[usize], second_row: &[usize]) -> BinaryNumber {
    RowBitwiseZipIter::new(first_row, second_row)
        .fold(0, |sum, x| sum ^ x.first_row_value * x.second_row_value)
}

struct RowBitwiseZipIter<'a> {
    first_row: &'a [usize],
    second_row: &'a [usize],
    first_row_index: usize,
    second_row_index: usize,
}

impl<'a> Iterator for RowBitwiseZipIter<'a> {
    type Item = PositionAndValues;

    fn next(&mut self) -> Option<Self::Item> {
        match (
            self.next_position_in_first_row(),
            self.next_position_in_second_row(),
        ) {
            (Some(p), Some(q)) => match p.cmp(&q) {
                Ordering::Equal => {
                    self.advance_both_rows();
                    Some(PositionAndValues::both_rows_at(p))
                }
                Ordering::Less => {
                    self.advance_first_row();
                    Some(PositionAndValues::only_first_row_at(p))
                }
                Ordering::Greater => {
                    self.advance_second_row();
                    Some(PositionAndValues::only_second_row_at(q))
                }
            },
            (Some(p), None) => {
                self.advance_first_row();
                Some(PositionAndValues::only_first_row_at(p))
            }
            (None, Some(q)) => {
                self.advance_second_row();
                Some(PositionAndValues::only_second_row_at(q))
            }
            _ => None,
        }
    }
}

impl<'a> RowBitwiseZipIter<'a> {
    fn new(first_row: &'a [usize], second_row: &'a [usize]) -> Self {
        Self {
            first_row,
            second_row,
            first_row_index: 0,
            second_row_index: 0,
        }
    }

    fn next_position_in_first_row(&self) -> Option<usize> {
        self.first_row.get(self.first_row_index).cloned()
    }

    fn next_position_in_second_row(&self) -> Option<usize> {
        self.second_row.get(self.second_row_index).cloned()
    }

    fn advance_both_rows(&mut self) {
        self.advance_first_row();
        self.advance_second_row();
    }

    fn advance_first_row(&mut self) {
        self.first_row_index += 1;
    }

    fn advance_second_row(&mut self) {
        self.second_row_index += 1;
    }
}

struct PositionAndValues {
    position: usize,
    first_row_value: BinaryNumber,
    second_row_value: BinaryNumber,
}

impl PositionAndValues {
    fn both_rows_at(position: usize) -> Self {
        Self {
            position,
            first_row_value: 1,
            second_row_value: 1,
        }
    }

    fn only_first_row_at(position: usize) -> Self {
        Self {
            position,
            first_row_value: 1,
            second_row_value: 0,
        }
    }

    fn only_second_row_at(position: usize) -> Self {
        Self {
            position,
            first_row_value: 0,
            second_row_value: 1,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot_product() {
        let left_row = vec![0, 1, 3, 5, 8];
        let right_row = vec![1, 2, 5, 6, 8];
        assert_eq!(rows_dot_product(&left_row, &right_row), 1);

        let left_row = vec![0, 1, 5, 6, 8];
        let right_row = vec![1, 2, 5, 6, 8];
        assert_eq!(rows_dot_product(&left_row, &right_row), 0);

        let left_row = vec![0, 1, 3, 5, 8];
        let right_row = vec![];
        assert_eq!(rows_dot_product(&left_row, &right_row), 0);
    }

    #[test]
    fn bitwise_sum() {
        let left_row = vec![0, 1, 3, 5, 8];
        let right_row = vec![1, 2, 5, 6, 8];
        assert_eq!(rows_bitwise_sum(&left_row, &right_row), vec![0, 2, 3, 6]);

        let left_row = vec![0, 1, 5, 6, 8];
        let right_row = vec![1, 2, 5, 6, 8];
        assert_eq!(rows_bitwise_sum(&left_row, &right_row), vec![0, 2]);

        let left_row = vec![0, 1, 3, 5, 8];
        let right_row = vec![];
        assert_eq!(rows_bitwise_sum(&left_row, &right_row), vec![0, 1, 3, 5, 8]);
    }
}
