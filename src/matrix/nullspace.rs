use super::SparseBinMat;
use itertools::Itertools;

pub(super) fn nullspace(matrix: &SparseBinMat) -> SparseBinMat {
    let echelon_form = matrix.echelon_form();
    let (normal_form, permutation) = normal_form_from_echelon_form(&echelon_form);
    let nullspace = normal_form
        .without_columns(&(0..normal_form.number_of_rows()).collect_vec())
        .unwrap()
        .vertical_concat_with(&SparseBinMat::identity(
            normal_form.number_of_columns() - normal_form.number_of_rows(),
        ));
    permute_columns(&nullspace.transposed(), &inverse_permutation(&permutation))
}

fn normal_form_from_echelon_form(matrix: &SparseBinMat) -> (SparseBinMat, Vec<usize>) {
    let (swapped_echelon_form, permutation) = swap_echelon_form(matrix);
    (reduce_to_normal_form(&swapped_echelon_form), permutation)
}

fn swap_echelon_form(matrix: &SparseBinMat) -> (SparseBinMat, Vec<usize>) {
    let mut permutation = (0..matrix.number_of_columns()).collect::<Vec<_>>();
    for (index, row) in matrix.rows().enumerate() {
        let column = row.as_slice()[0];
        if column != index {
            permutation.swap(index, column);
        }
    }
    (permute_columns(&matrix, &permutation), permutation)
}

fn reduce_to_normal_form(matrix: &SparseBinMat) -> SparseBinMat {
    let rows = matrix
        .rows()
        .map(|row| {
            let mut r = row.to_owned();
            while r.weight() > 1 && r.as_slice()[1] < matrix.number_of_rows() {
                r = &r + &matrix.row(r.as_slice()[1]).unwrap();
            }
            r.to_positions_vec()
        })
        .collect();
    SparseBinMat::new(matrix.number_of_columns(), rows)
}

fn permute_columns(matrix: &SparseBinMat, permutation: &[usize]) -> SparseBinMat {
    let inverse = inverse_permutation(&permutation);
    let rows = matrix
        .rows()
        .map(|row| {
            row.non_trivial_positions()
                .map(|column| inverse[column])
                .sorted()
                .collect()
        })
        .collect();
    SparseBinMat::new(matrix.number_of_columns(), rows)
}

fn inverse_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; permutation.len()];
    for (index, position) in permutation.iter().enumerate() {
        inverse[*position] = index;
    }
    inverse
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn nullspace_of_rank_1_matrix() {
        let matrix = SparseBinMat::new(4, vec![vec![0, 1, 2, 3]]);
        let expected = SparseBinMat::new(4, vec![vec![0, 1], vec![0, 2], vec![0, 3]]);
        assert_eq!(nullspace(&matrix), expected);
    }

    #[test]
    fn normal_form() {
        let matrix = SparseBinMat::new(6, vec![vec![0, 1, 2], vec![1, 4], vec![3, 4, 5], vec![5]]);
        let expected = SparseBinMat::new(6, vec![vec![0, 4, 5], vec![1, 4], vec![2, 4], vec![3]]);
        let permutation = vec![0, 1, 3, 5, 4, 2];
        assert_eq!(
            normal_form_from_echelon_form(&matrix),
            (expected, permutation)
        )
    }

    #[test]
    fn permutation() {
        let permutation = vec![0, 2, 4, 1, 3, 5, 6];
        let inverse = vec![0, 3, 1, 4, 2, 5, 6];
        assert_eq!(inverse_permutation(&permutation), inverse);
    }

    #[test]
    fn column_permutation() {
        let matrix = SparseBinMat::new(
            6,
            vec![
                vec![0, 1, 2],
                vec![3, 4, 5],
                vec![0, 2, 4],
                vec![1, 3, 5],
                vec![0, 5],
            ],
        );
        let permutation = vec![1, 0, 2, 4, 5, 3];
        let expected = SparseBinMat::new(
            6,
            vec![
                vec![0, 1, 2],
                vec![3, 4, 5],
                vec![1, 2, 3],
                vec![0, 4, 5],
                vec![1, 4],
            ],
        );
        assert_eq!(permute_columns(&matrix, &permutation), expected);
    }
}
