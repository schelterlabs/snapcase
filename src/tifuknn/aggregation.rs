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
