use blas::*;

use crate::tifuknn::{NUM_FEATURES, NUM_HASH_DIMENSIONS};
use crate::tifuknn::types::Embedding;

pub fn random_projection(
    random_projection_matrix: &Vec<f64>,
    user_vector: &Embedding
) -> u32 {

    let mut projection: Vec<f64> = vec![0.0; NUM_HASH_DIMENSIONS];

    let vector = user_vector.clone_into_dense_vector();

    // Random projection of user vector
    unsafe {
        dgemv(
            b'T',
            NUM_FEATURES as i32,
            NUM_HASH_DIMENSIONS as i32,
            1.0,
            random_projection_matrix,
            NUM_FEATURES as i32,
            &vector,
            1,
            0.0,
            &mut projection,
            1);
    }

    // TODO this should only be used for cosine distance, for euclidean distance,
    // TODO we need something along the lines of
    // TODO https://people.scs.carleton.ca/~maheshwa/courses/5703COMP/Notes/LSH/5703-lsh-slides.pdf
    // Signs of the result of the random projection give us the bucket key
    let mut key = 0u32;
    for (dimension, value) in projection.iter().enumerate() {
        if *value > 0.0 {
            key |= 1u32 << dimension as u32;
        }
    }

    key
}
