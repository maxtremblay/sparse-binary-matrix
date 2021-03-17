use super::SparseBinMat;

// This assume that the value at the given position is zero before the update
// and that row and column are inbound.
pub(super) fn insert_one_at(mut matrix: SparseBinMat, row: usize, column: usize) -> SparseBinMat {
    matrix = update_column_indices_to_insert_one_at(matrix, row, column);
    increase_row_ranges_after(matrix, row)
}

fn update_column_indices_to_insert_one_at(
    mut matrix: SparseBinMat,
    row: usize,
    column: usize,
) -> SparseBinMat {
    let row_start = matrix.row_ranges[row];
    let row_end = matrix.row_ranges[row + 1];
    let offset = &matrix.column_indices[row_start..row_end]
        .iter()
        .position(|col| *col > column)
        .unwrap_or(row_end);
    let position = row_start + offset;
    matrix.column_indices.insert(position, column);
    matrix
}

fn increase_row_ranges_after(mut matrix: SparseBinMat, row: usize) -> SparseBinMat {
    for row_range in matrix.row_ranges[row + 1..].iter_mut() {
        *row_range += 1;
    }
    matrix
}

// This assume that the value at the given position is one before the update
// and that row and column are inbound.
pub(super) fn remove_zero_at(mut matrix: SparseBinMat, row: usize, column: usize) -> SparseBinMat {
    matrix = update_column_indices_to_remove_zero_at(matrix, row, column);
    decrease_row_ranges_after(matrix, row)
}

fn update_column_indices_to_remove_zero_at(
    mut matrix: SparseBinMat,
    row: usize,
    column: usize,
) -> SparseBinMat {
    let row_start = matrix.row_ranges[row];
    let row_end = matrix.row_ranges[row + 1];
    let offset = &matrix.column_indices[row_start..row_end]
        .iter()
        .position(|col| *col == column)
        .expect("[Bug - Please report]");
    let position = row_start + offset;
    matrix.column_indices.remove(position);
    matrix
}

fn decrease_row_ranges_after(mut matrix: SparseBinMat, row: usize) -> SparseBinMat {
    for row_range in matrix.row_ranges[row + 1..].iter_mut() {
        *row_range -= 1;
    }
    matrix
}
