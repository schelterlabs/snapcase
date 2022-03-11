extern crate csv;

use std::cmp::max;
use std::path::Path;
use path_absolutize::Absolutize;

// pub fn users_from_baskets_partitioned(
//     baskets: &Vec<(u32, usize, Vec<usize>)>,
//     partition: usize,
//     num_partitions: usize,
// ) -> Vec<u32> {
//     let all_user_ids = users_from_baskets(baskets);
//
//     all_user_ids.into_iter()
//         .filter(|user_id| *user_id as usize % num_partitions == partition)
//         .collect()
// }

pub fn users_from_baskets(baskets: &Vec<(u32, usize, Vec<usize>)>) -> Vec<u32> {
    let mut user_ids: Vec<u32> = baskets.iter()
        .map(|(user, _, _)| *user)
        .collect();

    user_ids.sort_unstable();
    user_ids.dedup();

    user_ids
}

// pub fn baskets_from_file_partitioned(
//     path_to_file: &str,
//     partition: usize,
//     num_partitions: usize,
// ) -> Vec<(u32, usize, Vec<usize>)> {
//     let (all_baskets, _) = baskets_from_file(path_to_file);
//
//     all_baskets.iter()
//         .filter(|(user_id, _, _)| *user_id as usize % num_partitions == partition)
//         .map(|(user_id, basket, items)| (*user_id, *basket, items.clone()))
//         .collect()
// }

pub fn baskets_from_file(path_to_file: &str) -> (Vec<(u32, usize, Vec<usize>)>, usize) {

    let path: &Path = Path::new(path_to_file);
    println!("# Trying to read path: {:?}", path.absolutize().unwrap());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_path(path)
        .unwrap();

    let mut current_user: Option<u32> = None;
    let mut current_basket: Option<usize> = None;
    let mut baskets: Vec<(u32, usize, Vec<usize>)> = Vec::new();
    let mut buffer: Vec<usize> = Vec::new();

    let mut num_items = 0;

    for result in reader.deserialize() {
        let raw: (u32, usize, usize) = result.unwrap();

        let user = raw.0;
        let basket = raw.1;
        let item = raw.2 - 1;

        num_items = max(num_items, raw.2);

        if current_user.is_none() {
            current_user = Some(user);
        }
        if current_basket.is_none() {
            current_basket = Some(basket);
        }

        if user == current_user.unwrap() {
            if basket == current_basket.unwrap() {
                buffer.push(item);
            } else {
                baskets.push(
                    (current_user.unwrap().clone(),
                     current_basket.unwrap().clone(),
                     buffer.clone())
                );
                buffer.clear();
                current_basket = Some(basket);
                buffer.push(item);
            }
        } else {
            baskets.push(
                (current_user.unwrap().clone(),
                 current_basket.unwrap().clone(),
                 buffer.clone())
            );
            buffer.clear();
            current_user = Some(user);
            current_basket = Some(basket);
            buffer.push(item);
        }
    }

    if !buffer.is_empty() {
        baskets.push(
            (current_user.unwrap().clone(),
             current_basket.unwrap().clone(),
             buffer.clone())
        );
    }

    baskets.iter_mut()
        .for_each(|(_user, _basket, items)| {
            items.sort()
        });

    println!("# Found {} baskets for {} items", baskets.len(), num_items);

    (baskets, num_items)
}
