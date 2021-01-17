
pub fn read_interactions(dataset_file: &'static str, num_users: usize) -> Vec<(u32, Vec<u32>)> {

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_path(dataset_file)
        .expect("Unable to read input file");

    let mut interactions: Vec<(u32, Vec<u32>)> = (0 .. num_users)
        .map(|index| (index as u32, Vec::new())).collect();

    reader.deserialize()
        .for_each(|result| {
            if result.is_ok() {
                let (user, item): (u32, u32) = result.unwrap();
                let user_idx = user - 1;
                let item_idx = item - 1;

                if interactions[user_idx as usize].1.len() < 500 {
                    interactions[user_idx as usize].1.push(item_idx);
                }
            }
        });

    interactions
}