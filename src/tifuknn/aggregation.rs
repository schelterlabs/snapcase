use crate::tifuknn::types::{Basket, DiscretisedItemVector, SparseItemVector};

fn index(multiplicity: isize, bucket_size: isize) -> isize {
    let offset = multiplicity % bucket_size;
    if offset == 0 {
        bucket_size
    } else {
        offset
    }
}

pub fn group_vector(
    group: usize,
    baskets_and_multiplicities: &[(&Basket, isize)],
    group_size: isize,
    r: f64,
) -> DiscretisedItemVector {

    // The last group might not be filled, so we have to "shift" the baskets to the right
    let correction_offset = group_size - baskets_and_multiplicities.len() as isize;

    let mut group_vector = SparseItemVector::new();

    let num_baskets_in_group = baskets_and_multiplicities.len() as f64;

    for (basket, multiplicity) in baskets_and_multiplicities {

        let index = index(*multiplicity, group_size) + correction_offset;
        //println!("GROUP: {:?} {:?} {:?} - {:?}", r, group_size, index, basket.items);
        let decay = r.powi((&group_size - index) as i32);
        let contribution = decay / num_baskets_in_group;

        for item in &basket.items {
            group_vector.plus_at(*item, contribution);
        }
    }

    DiscretisedItemVector::new(group, group_vector)
}

pub fn user_vector(
    user: u32,
    group_vectors_and_multiplicities: &[(&DiscretisedItemVector, isize)],
    r: f64,
) -> DiscretisedItemVector {

    let mut user_vector = SparseItemVector::new();

    let num_groups = group_vectors_and_multiplicities.len() as isize;

    for (group_vector, multiplicity) in group_vectors_and_multiplicities {

        let group_index = *multiplicity;
        let index = index(group_index, num_groups);

        let m_minus_i = (&num_groups - index) as i32;
        let r_g_m_minus_i = r.powi(m_minus_i);
        let multiplier = r_g_m_minus_i / num_groups as f64;

        //println!("USER: {:?} {:?} {:?} - {:?}", r, num_groups, index, group_vector);

        user_vector.plus_mult(multiplier, group_vector);
    }

    DiscretisedItemVector::new(user as usize, user_vector)
}

pub fn top_k_neighbors_by_jaccard(
    user_vector: &DiscretisedItemVector,
    neighbor_vectors: &[(&DiscretisedItemVector, isize)],
    k: usize
) -> Vec<usize> {
    // TODO We might turn this into a hashset for fast look ups
    let items = &user_vector.indices;
    // TODO identify top-k neighbors, this should use a heap to reduce memory
    let mut similarities = Vec::with_capacity(neighbor_vectors.len());

    for (index, (neighbor_vector, _mult)) in neighbor_vectors.iter().enumerate() {
        let neighbor_items = &neighbor_vector.indices;
        let mut intersection = 0;
        for item in items {
            if neighbor_items.contains(&item) {
                intersection += 1;
            }
        }
        let jaccard_similarity = intersection as f64 /
            (items.len() + neighbor_items.len() - intersection) as f64;
        similarities.push((index, jaccard_similarity))
    }

    // We like to live dangerous
    similarities.sort_by(|(_, sim_a), (_, sim_b)| sim_b.partial_cmp(sim_a).unwrap());
    // TODO can we avoid the collect?
    similarities.iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect()
}

pub fn recommendation(
    top_k_neighbor_indices: Vec<usize>,
    user_vector: &DiscretisedItemVector,
    neighbor_vectors: &[(&DiscretisedItemVector, isize)],
    alpha: f64,
) -> DiscretisedItemVector {

    let num_neighbors = top_k_neighbor_indices.len();
    let mut sum_of_neighbors = SparseItemVector::new();

    for index in top_k_neighbor_indices {
        let (other_user_vector, _multiplicity) = neighbor_vectors.get(index).unwrap();
        sum_of_neighbors.plus(other_user_vector);
    }

    let neighbor_factor = (1.0 - alpha) * (1.0 / num_neighbors as f64);
    sum_of_neighbors.mult(neighbor_factor);
    sum_of_neighbors.plus_mult(alpha, &user_vector);

    DiscretisedItemVector::new(user_vector.id, sum_of_neighbors)
}
