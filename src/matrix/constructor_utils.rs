pub(crate) fn initialize_from(
    rows: Vec<Vec<usize>>,
    capacity: Option<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let mut row_ranges = init_row_ranges(&rows);
    let mut column_indices = init_column_indices(&rows, capacity);
    for row in rows {
        add_row(row, &mut row_ranges, &mut column_indices);
    }
    (row_ranges, column_indices)
}

fn init_column_indices(rows: &[Vec<usize>], capacity: Option<usize>) -> Vec<usize> {
    let capacity = match capacity {
        Some(cap) => cap,
        None => rows.iter().map(|row| row.len()).sum(),
    };
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
    column_indices.append(&mut row);
}
