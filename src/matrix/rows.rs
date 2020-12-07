use super::SparseBinMat;
use crate::SparseBinSlice;

/// An iterator over the rows of matrix.
///
/// See the [`rows`](SparseBinMat::rows) method.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows<'a> {
    matrix: &'a SparseBinMat,
    front: usize,
    back: usize,
}

impl<'a> Rows<'a> {
    pub(super) fn from(matrix: &'a SparseBinMat) -> Self {
        Self {
            matrix,
            front: 0,
            back: matrix.number_of_rows(),
        }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = SparseBinSlice<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let row = self.matrix.row(self.front);
            self.front += 1;
            row
        } else {
            None
        }
    }
}

impl<'a> DoubleEndedIterator for Rows<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.back > self.front {
            self.back -= 1;
            self.matrix.row(self.back)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn checks_iterator() {
        let rows = vec![vec![0, 1, 2], vec![2, 3], vec![0, 3], vec![1, 2, 3]];
        let matrix = SparseBinMat::new(4, rows.clone());
        let mut iter = Rows::from(&matrix);

        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[0])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[1])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[2])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[3])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn checks_iterator_for_empty_matrix() {
        let matrix = SparseBinMat::empty();
        let mut iter = Rows::from(&matrix);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn reverse_checks_iterator() {
        let rows = vec![vec![0, 1, 2], vec![2, 3], vec![0, 3], vec![1, 2, 3]];
        let matrix = SparseBinMat::new(4, rows.clone());
        let mut iter = Rows::from(&matrix).rev();

        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[3])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[2])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[1])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[0])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn bothway_checks_iterator() {
        let rows = vec![vec![0, 1, 2], vec![2, 3], vec![0, 3], vec![1, 2, 3]];
        let matrix = SparseBinMat::new(4, rows.clone());
        let mut iter = Rows::from(&matrix);

        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[0])));
        assert_eq!(iter.next_back(), Some(SparseBinSlice::new(4, &rows[3])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[1])));
        assert_eq!(iter.next(), Some(SparseBinSlice::new(4, &rows[2])));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
    }
}
