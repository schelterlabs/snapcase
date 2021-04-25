use crate::tifuknn::types::{Basket, Embedding};
use super::sprs::CsVec;
use std::ops::{Add, MulAssign};


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
) -> Embedding {

    // The last group might not be filled, so we have to "shift" the baskets to the right
    let correction_offset = group_size - baskets_and_multiplicities.len() as isize;

    let mut embedding: CsVec<f64> =  CsVec::empty(4);

    for (basket, multiplicity) in baskets_and_multiplicities {
        assert!(*multiplicity > 0);

        let index = index(*multiplicity, group_size) + correction_offset;
        let decay = r.powi((&group_size - index) as i32);
        let contribution = decay / group_size as f64;

        let basket_as_vector = CsVec::new(
            4,
            basket.items.clone(),
            vec![contribution; basket.items.len()]
        );
        // Sprs might reallocate here...
        embedding = embedding.add(basket_as_vector);
    }

    println!("{:?}", embedding);
    Embedding::new(group, embedding)
}

pub fn user_vector(
    user: u32,
    group_vectors_and_multiplicities: &[(&Embedding, isize)],
    r: f64,
) -> Embedding {

    let mut embedding: CsVec<f64> =  CsVec::empty(4);

    let num_groups = group_vectors_and_multiplicities.len() as isize;

    for (group_vector, multiplicity) in group_vectors_and_multiplicities {
        assert!(*multiplicity > 0);

        let index = index(*multiplicity, num_groups);

        let m_minus_i = (&num_groups - index) as i32;
        let r_g_m_minus_i = r.powi(m_minus_i);
        let multiplier = r_g_m_minus_i / num_groups as f64;

        // TODO get rid of the copy here
        let mut group_embedding: CsVec<f64> = (*group_vector).clone().into_sparse_vector();

        group_embedding.mul_assign(multiplier);
        embedding = embedding + &group_embedding;
    }

    Embedding::new(user as usize, embedding)
}
