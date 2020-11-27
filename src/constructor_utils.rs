pub(crate) fn assert_rows_are_inbound(number_of_columns: usize, rows: &[Vec<usize>]) {
    for row in rows {
        if row_is_out_of_bound(number_of_columns, row) {
            panic!(
                "row {:?} is out of bound for {} columns",
                row, number_of_columns
            );
        }
    }
}

fn row_is_out_of_bound(number_of_columns: usize, row: &[usize]) -> bool {
    row.iter().any(|position| *position >= number_of_columns)
}

pub(crate) fn initialize_from(rows: Vec<Vec<usize>>) -> (Vec<usize>, Vec<usize>) {
    let mut row_ranges = init_row_ranges(&rows);
    let mut column_indices = init_column_indices(&rows);
    for row in rows {
        add_row(row, &mut row_ranges, &mut column_indices);
    }
    (row_ranges, column_indices)
}

fn init_column_indices(rows: &[Vec<usize>]) -> Vec<usize> {
    let capacity = rows.iter().map(|row| row.len()).sum();
    Vec::with_capacity(capacity)
}

fn init_row_ranges(rows: &[Vec<usize>]) -> Vec<usize> {
    let mut row_ranges = Vec::with_capacity(rows.len() + 1);
    row_ranges.push(0);
    row_ranges
}

fn add_row(mut row: Vec<usize>, row_ranges: &mut Vec<usize>, column_indices: &mut Vec<usize>) {
    let elements_before = column_indices.len();
    row_ranges.push(elements_before + row.len());
    row.sort();
    column_indices.append(&mut row);
}
