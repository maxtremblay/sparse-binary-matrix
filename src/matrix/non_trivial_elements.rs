use super::SparseBinMat;
use crate::SparseBinSlice;

/// An iterator over the coordinates of non trivial elements.
///
/// See the [`non_trivial_elements`](SparseBinMat::non_trivial_elements) method.
#[derive(Debug, Clone)]
pub struct NonTrivialElements<'a> {}

impl<'a> NonTrivialElements<'a> {
    pub(super) fn new(matrix: &'a SparseBinMat) -> Self {}

    fn move_to_next_row(&mut self) {
        todo!()
    }
}

impl<'a> Iterator for NonTrivialElements<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {}
}

#[cfg(test)]
mod test {
    use super::*;
}
