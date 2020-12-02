use super::SparseBinSlice;
use crate::BinaryNumber;
use std::cmp::Ordering;

pub(super) struct BitwiseZipIter<'a> {
    first: &'a [usize],
    second: &'a [usize],
    first_index: usize,
    second_index: usize,
}

impl<'a> Iterator for BitwiseZipIter<'a> {
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

impl<'a> BitwiseZipIter<'a> {
    pub(super) fn new(first: SparseBinSlice<'a>, second: SparseBinSlice<'a>) -> Self {
        Self {
            first: first.positions,
            second: second.positions,
            first_index: 0,
            second_index: 0,
        }
    }

    fn next_position_in_first_row(&self) -> Option<usize> {
        self.first.get(self.first_index).cloned()
    }

    fn next_position_in_second_row(&self) -> Option<usize> {
        self.second.get(self.second_index).cloned()
    }

    fn advance_both_rows(&mut self) {
        self.advance_first_row();
        self.advance_second_row();
    }

    fn advance_first_row(&mut self) {
        self.first_index += 1;
    }

    fn advance_second_row(&mut self) {
        self.second_index += 1;
    }
}

pub(super) struct PositionAndValues {
    pub(super) position: usize,
    pub(super) first_row_value: BinaryNumber,
    pub(super) second_row_value: BinaryNumber,
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
