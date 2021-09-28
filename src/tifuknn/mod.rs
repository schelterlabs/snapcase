extern crate timely;
extern crate differential_dataflow;
extern crate datasketch_minhash_lsh;

pub mod types;
pub mod aggregation;

use crate::tifuknn::types::{Basket, BucketKey, DiscretisedItemVector, SparseItemVector};

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::operators::arrange::ArrangeByKey;

use differential_dataflow::operators::{Reduce, Join};

use crate::tifuknn::aggregation::{group_vector, user_vector};

use std::cmp;

use datasketch_minhash_lsh::MinHash;
use self::differential_dataflow::operators::Threshold;
use datasketch_minhash_lsh::{LshParams, Weights};

// TODO these should not be hardcoded, we need a params object and must also include the r's
const GROUP_SIZE: isize = 7;
const R_GROUP: f64 = 0.7;
const R_USER: f64 = 0.9;
const RANDOM_SEED: u64 = 42;
const K: usize = 300;
const ALPHA: f64 = 0.7;
const NUM_PERMUTATION_FUNCS: usize = 1280;
const JACCARD_THRESHOLD: f64 = 0.1;
const LSH_WEIGHTS: Weights = Weights(0.5, 0.5);

// TODO refactor this into several submodules
pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
)   -> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    worker.dataflow(|scope| {
        let LshParams { b: bands, r: bucket_key_length } = LshParams::find_optimal_params(
            JACCARD_THRESHOLD, NUM_PERMUTATION_FUNCS, &LSH_WEIGHTS);

        let baskets = baskets_input.to_collection(scope);
        let group_vectors = baskets
            .reduce(|_user, baskets_and_multiplicities, out| {
                for (basket, multiplicity) in baskets_and_multiplicities {
                    // TODO write a test for this...
                    let group = if *multiplicity % GROUP_SIZE == 0 {
                        *multiplicity / GROUP_SIZE
                    } else {
                        (*multiplicity + (GROUP_SIZE - (*multiplicity % GROUP_SIZE))) / GROUP_SIZE
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
                    GROUP_SIZE,
                    R_GROUP,
                );

                out.push((group_vector, *group));
            })
            .map(|((user, _), group_vector)| (user, group_vector));
            //.inspect(|x| println!("Group vector {:?}", x));

        let user_vectors = group_vectors
            .reduce(move |user, vectors_and_multiplicities, out| {
                let user_vector = user_vector(*user, vectors_and_multiplicities, R_USER);
                println!("USER-{}-{}", user, user_vector.print());
                out.push((user_vector, 1))
            });
            //.inspect(|x| println!("User vector {:?}", x));

        let bucketed_user_vectors = user_vectors
            .flat_map(move |(user, user_vector)| {

                let mut hasher = MinHash::new(NUM_PERMUTATION_FUNCS, Some(RANDOM_SEED));

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
            .join(&bucketed_user_vectors)
            .flat_map(|(_bucket_key, (user_a, user_b))| { // TODO maybe we should do this later?
                [(user_a, user_b), (user_b, user_a)]
            })
            .distinct();

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
                      });

        let recommendations = cooccurring_users_with_user_vectors
            .reduce(move |(user_a, user_vector_a), users_b_vectors_mults, out| {

                let indices_a = &user_vector_a.indices;
                // TODO identify top-k neighbors, this should use a heap to reduce memory
                let mut similarities = Vec::with_capacity(users_b_vectors_mults.len());

                for (index, ((_user_b, user_vector_b), _)) in users_b_vectors_mults.iter().enumerate() {
                    let indices_b = &user_vector_b.indices;
                    let mut intersection = 0;
                    for index_a in indices_a {
                        if indices_b.contains(&index_a) {
                            intersection += 1;
                        }
                    }
                    let jaccard_similarity = intersection as f64 /
                        (indices_a.len() + indices_b.len() - intersection) as f64;
                    similarities.push((index, jaccard_similarity))
                }

                // We like to live dangerous
                similarities.sort_by(|(_, sim_a), (_, sim_b)| sim_b.partial_cmp(sim_a).unwrap());

                // TODO skip neighbor identification if we have <= K other vectors here!
                let num_neighbors = cmp::min(similarities.len(), K);

                let mut sum_of_neighbors = SparseItemVector::new();

                for (index, _similarity) in similarities.iter().take(num_neighbors) {
                    let ((_, other_user_vector), _) = users_b_vectors_mults.get(*index).unwrap();
                    sum_of_neighbors.plus(other_user_vector);
                }

                let neighbor_factor = (1.0 - ALPHA) * (1.0 / num_neighbors as f64);
                sum_of_neighbors.mult(neighbor_factor);
                sum_of_neighbors.plus_mult(ALPHA, user_vector_a);

                let recommendations =
                    DiscretisedItemVector::new(*user_a as usize, sum_of_neighbors);

                out.push((recommendations, 1));
            })
            .map(move |((user, _), recommendations)| {
                println!("RECO-{}-{}", user, recommendations.print());
                (user, recommendations)
            });

        recommendations
            //.inspect(|x| println!("RECO {:?}", x))
            .probe()
    })
}