use crate::tifuknn::types::Basket;

fn index(multiplicity: isize, bucket_size: isize) -> isize {
    let offset = multiplicity % bucket_size;
    if offset == 0 {
        bucket_size
    } else {
        offset
    }
}

pub fn group_vector(
    baskets_and_multiplicities: &[(&Basket, isize)],
    group_size: isize,
    r: f64,
) -> (u64, u64, u64, u64) {

    let mut group_vector = (0, 0, 0, 0);

    // The last group might not be filled, so we have to "shift" the baskets to the right
    let correction_offset = group_size - baskets_and_multiplicities.len() as isize;

    for (basket, multiplicity) in baskets_and_multiplicities {
        assert!(*multiplicity > 0);

        let index = index(*multiplicity, group_size) + correction_offset;
        let decay = r.powi((&group_size - index) as i32);

        group_vector.0 += ((basket.items.0 as f64 * decay / group_size as f64) * 100.0) as u64;
        group_vector.1 += ((basket.items.1 as f64 * decay / group_size as f64) * 100.0) as u64;
        group_vector.2 += ((basket.items.2 as f64 * decay / group_size as f64) * 100.0) as u64;
        group_vector.3 += ((basket.items.3 as f64 * decay / group_size as f64) * 100.0) as u64;
    }

    group_vector
}

pub fn user_vector(
    group_vectors_and_multiplicities: &[(&(u64, u64, u64, u64), isize)],
    r: f64,
) -> (u64, u64, u64, u64) {

    let mut user_vector = (0, 0, 0, 0);

    let num_groups = group_vectors_and_multiplicities.len() as isize;

    for (group_vector, multiplicity) in group_vectors_and_multiplicities {
        assert!(*multiplicity > 0);

        let index = index(*multiplicity, num_groups);

        let m_minus_i = (&num_groups - index) as i32;
        let r_g_m_minus_i = r.powi(m_minus_i);

        user_vector.0 += (group_vector.0 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
        user_vector.1 += (group_vector.1 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
        user_vector.2 += (group_vector.2 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
        user_vector.3 += (group_vector.3 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
    }

    user_vector
}
