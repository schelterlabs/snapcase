use crate::tifuknn::types::{Basket, DiscretisedItemVector, BucketKey, HyperParams};
use crate::tifuknn::aggregation::{group_vector, user_vector, top_k_neighbors_by_jaccard, recommendation};

use timely::dataflow::Scope;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::Collection;
use differential_dataflow::operators::{Join, Reduce};
use differential_dataflow::operators::arrange::ArrangeByKey;

use datasketch_minhash_lsh::{MinHash, LshParams, Weights};

pub (crate) fn user_vectors<G: Scope>(
    baskets: &Collection<G, (u32, Basket), isize>,
    params: HyperParams,
) -> Collection<G, (u32, DiscretisedItemVector), isize>
    where G::Timestamp: Lattice+Ord
{
    let group_vectors = baskets
        .reduce(move |_user, baskets_and_multiplicities, out| {
            for (basket, multiplicity) in baskets_and_multiplicities {
                // TODO write a test for this...
                let group = if *multiplicity % params.group_size == 0 {
                    *multiplicity / params.group_size
                } else {
                    (*multiplicity + (params.group_size - (*multiplicity % params.group_size)))
                        / params.group_size
                };

                assert_ne!(group, 0);

                out.push(((group, (*basket).clone()), *multiplicity));
            }
        })
        .map(|(user, (group, basket))| ((user, group), basket))
        .reduce(move |(_user, group), baskets_and_multiplicities, out| {
            let group_vector = group_vector(
                *group as usize,
                baskets_and_multiplicities,
                params.group_size,
                params.r_basket,
            );

            out.push((group_vector, *group));
        })
        .map(|((user, _), group_vector)| (user, group_vector));
    //.inspect(|x| println!("Group vector {:?}", x));

    let user_vectors = group_vectors
        .reduce(move |user, vectors_and_multiplicities, out| {
            let user_vector = user_vector(*user, vectors_and_multiplicities, params.r_group);
            //println!("USER-{}-{}", user, user_vector.print());
            out.push((user_vector, 1))
        });

    user_vectors
}

pub (crate) fn lsh_recommendations<G: Scope>(
    user_vectors: &Collection<G, (u32, DiscretisedItemVector), isize>,
    params: HyperParams,
) -> Collection<G, (u32, DiscretisedItemVector), isize>
    where G::Timestamp: Lattice+Ord
{
    let LshParams { b: bands, r: bucket_key_length } =
        LshParams::find_optimal_params(
            params.jaccard_threshold,
            params.num_permutation_functions,
            &Weights(0.5, 0.5)
        );

    let bucketed_user_vectors = user_vectors
        .flat_map(move |(user, user_vector)| {

            let mut hasher =
                MinHash::new(params.num_permutation_functions, Some(params.random_seed));

            for index in user_vector.indices.iter() {
                hasher.update(index);
            }

            let hashes = hasher.hash_values.0;

            (0..bands).map(move |band_index| {
                let start_index = band_index * bucket_key_length;
                let end_index = start_index + bucket_key_length;
                let hashes_for_bucket = hashes[start_index..end_index].to_vec();

                let key = BucketKey::new(band_index, hashes_for_bucket);
                (key, user.clone())
            })
        });

    let cooccurring_users = bucketed_user_vectors
        .join_map(&bucketed_user_vectors, |_bucket_key, user_a, user_b| (*user_a, *user_b))
        .filter(|(user_a, user_b)| user_a < user_b);

    // Manual arrangement due to use in multiple joins, maybe we can use delta joins here?
    let arranged_user_vectors = user_vectors.arrange_by_key();

    let cooccurring_users_with_left_user_vectors = arranged_user_vectors
        .join_map(&cooccurring_users, |user_a, user_vector_a, user_b| {
            (*user_b, (*user_a, user_vector_a.clone()))
        });

    let cooccurring_users_with_user_vectors = arranged_user_vectors
        .join_map(&cooccurring_users_with_left_user_vectors,
                  |user_b, user_vector_b, (user_a, user_vector_a)| {
                      ((*user_a, user_vector_a.clone()), (*user_b, user_vector_b.clone()))
                  })
        .flat_map(|((user_a, user_vector_a), (user_b, user_vector_b))| {
            [((user_a, user_vector_a.clone()), user_vector_b.clone()),
                ((user_b, user_vector_b.clone()), user_vector_a.clone())]
        });

    let recommendations = cooccurring_users_with_user_vectors
        .reduce(move |(_user, user_vector), neighbor_vectors, out| {

            // TODO can we avoid having to allocate a Vec for this?
            let top_k_neighbors: Vec<usize> = if neighbor_vectors.len() > params.k {
                top_k_neighbors_by_jaccard(user_vector, neighbor_vectors, params.k)
            } else {
                (0..neighbor_vectors.len()).collect()
            };

            // let num_neighbors = top_k_neighbors.len();
            // let mut sum_of_neighbors = SparseItemVector::new();
            //
            // for index in top_k_neighbors {
            //     let (other_user_vector, _multiplicity) = neighbor_vectors.get(index).unwrap();
            //     sum_of_neighbors.plus(other_user_vector);
            // }
            //
            // let neighbor_factor = (1.0 - params.alpha) * (1.0 / num_neighbors as f64);
            // sum_of_neighbors.mult(neighbor_factor);
            // sum_of_neighbors.plus_mult(params.alpha, user_vector);
            //
            // let recommendations =
            //     DiscretisedItemVector::new(*user as usize, sum_of_neighbors);

            let recommendations = recommendation(
                top_k_neighbors,
                user_vector,
                neighbor_vectors,
                params.alpha
            );

            //println!("RECO-{}-{}", user, recommendations.print());

            out.push((recommendations, 1));
        })
        .map(move |((user, _), recommendations)| (user, recommendations));

    recommendations
}