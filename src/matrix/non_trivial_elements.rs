use super::SparseBinMat;

/// An iterator over the coordinates of non trivial elements.
///
/// See the [`non_trivial_elements`](SparseBinMat::non_trivial_elements) method.
#[derive(Debug, Clone)]
pub struct NonTrivialElements<'a> {
    matrix: &'a SparseBinMat,
    row_index: usize,
    column_index: usize,
}

impl<'a> NonTrivialElements<'a> {
    pub(super) fn new(matrix: &'a SparseBinMat) -> Self {
        Self {
            matrix,
            row_index: 0,
            column_index: 0,
        }
    }

    fn next_element(&mut self) -> Option<(usize, usize)> {
        self.matrix
            .row(self.row_index)
            .and_then(|row| row.as_slice().get(self.column_index).cloned())
            .map(|column| (self.row_index, column))
    }

    fn move_to_next_row(&mut self) {
        self.row_index += 1;
        self.column_index = 0;
    }

    fn move_to_next_column(&mut self) {
        self.column_index += 1;
    }

    fn is_done(&self) -> bool {
        self.row_index >= self.matrix.number_of_rows()
    }
}

impl<'a> Iterator for NonTrivialElements<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done() {
            None
        } else {
            match self.next_element() {
                Some(element) => {
                    self.move_to_next_column();
                    Some(element)
                }
                None => {
                    self.move_to_next_row();
                    self.next()
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn non_trivial_elements_of_small_matrix() {
        let matrix = SparseBinMat::new(3, vec![vec![1], vec![0, 2], vec![0, 1, 2]]);
        let mut iter = NonTrivialElements::new(&matrix);

        assert_eq!(iter.next(), Some((0, 1)));
        assert_eq!(iter.next(), Some((1, 0)));
        assert_eq!(iter.next(), Some((1, 2)));
        assert_eq!(iter.next(), Some((2, 0)));
        assert_eq!(iter.next(), Some((2, 1)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.next(), None);
    }
}
