pub fn minhash(permutations: &[Vec<usize>], set: &[usize], num_items: usize) -> Vec<usize> {

    let mut hashes = Vec::with_capacity(num_items);

    for permutation in permutations {
        'search: for index in permutation {
            if set.contains(index) {
                hashes.push(*index);
                break 'search;
            }
        }
    }
    assert_eq!(hashes.len(), permutations.len());
    hashes
}