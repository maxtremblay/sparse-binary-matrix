use sparse_bin_mat::SparseBinMat;

fn main() {
    let a = SparseBinMat::new(5, vec![
        vec![0, 4],
        vec![0, 1],
        vec![1, 2, 4],
        vec![0, 2],
    ]);
    
    let b = SparseBinMat::new(2, vec![
        vec![0],
        vec![1],
        vec![],
        vec![1],
    ]);
    let solution = a.solve(&b).unwrap();
    println!("{solution}");
}
