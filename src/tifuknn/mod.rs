extern crate timely;
extern crate differential_dataflow;
extern crate sprs;

pub mod types;
pub mod aggregation;
pub mod minhash;

use crate::tifuknn::types::{Basket, BucketKey, Embedding};

//use timely::dataflow::operators::probe::Handle;
//use timely::dataflow::operators::Probe;
use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
//use differential_dataflow::operators::arrange::ArrangeByKey;

use differential_dataflow::operators::{Reduce};//,CountTotal};

//use rand::thread_rng;
use crate::tifuknn::aggregation::{group_vector, user_vector};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;

use self::sprs::CsVec;
use std::ops::{Add, MulAssign};

const GROUP_SIZE: isize = 2;
const BUCKEY_KEY_LENGTH: usize = 1;
const BANDS: usize = 6;

pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
    num_items: usize,
//) -> (ProbeHandle<T>, Trace<u32, (usize, (u64, u64, u64, u64)), T, isize>)
)   -> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    worker.dataflow(|scope| {

        let num_items = num_items.clone();

        let baskets = baskets_input.to_collection(scope);
        let group_vectors = baskets
            .reduce(|_user, baskets_and_multiplicities, out| {
                for (basket, multiplicity) in baskets_and_multiplicities {
                    assert!(*multiplicity > 0);
                    let group = (*multiplicity + (*multiplicity % GROUP_SIZE)) / GROUP_SIZE;
                    out.push(((group, (*basket).clone()), *multiplicity));
                }
            })
            .map(|(user, (group, basket))| ((user, group), basket))
            .reduce(move |(_user, group), baskets_and_multiplicities, out| {
                let group_vector = group_vector(
                    *group as usize,
                    baskets_and_multiplicities,
                    GROUP_SIZE,
                    0.7,
                    num_items.clone(),
                );

                out.push((group_vector, *group));
            })
            .map(|((user, _), group_vector)| (user, group_vector))
            .inspect(|x| println!("Group vector {:?}", x));

        let user_vectors = group_vectors
            .reduce(move |user, vectors_and_multiplicities, out| {
                let user_vector = user_vector(
                    *user,
                    vectors_and_multiplicities,
                    0.9,
                    num_items.clone()
                );

                out.push((user_vector, 1))
            })
            .inspect(|x| println!("User vector {:?}", x));

        let permutations: Vec<Vec<usize>> = (0..(BANDS * BUCKEY_KEY_LENGTH))
            .map(|_| {
                let mut permutation: Vec<usize> = (0..num_items).collect();
                let mut rng = StdRng::seed_from_u64(42);
                // TODO we must consistently seed this rng once we run with multiple workers
                //permutation.shuffle(&mut thread_rng());
                permutation.shuffle(&mut rng);
                permutation
            })
            .collect();

        let bucketed_user_vectors = user_vectors
            .flat_map(move |(user, user_vector)| {
                let hashes = minhash::minhash(
                    &permutations,
                    &user_vector.indices,
                    num_items.clone()
                );

                (0..BANDS).map(move |band| {
                    let start_index = band * BUCKEY_KEY_LENGTH;
                    let end_index = start_index + BUCKEY_KEY_LENGTH;
                    let hashes_for_bucket = hashes[start_index..end_index].to_vec();

                    let key = BucketKey::new(hashes_for_bucket);
                    (key, (user.clone(), user_vector.clone()))
                })
            });

        // TODO now we can proceed to compute the actual recommendations

        // STEP 1: reduce over buckets, emit user vector + sum of bucket
        // STEP 2: group by user, compute recommendation vector

        let pre_aggregated_vectors = bucketed_user_vectors
            .reduce(move |_key, users_and_user_vectors, out| {

                let mut sum_of_bucket: CsVec<f64> =  CsVec::empty(num_items);

                for ((_, user_vector), _) in users_and_user_vectors {
                    // Remove copy & allocations here
                    sum_of_bucket = sum_of_bucket.add((*user_vector).clone().into_sparse_vector(num_items));
                }

                // This is a little bit of a hack, could be dangerous
                let embedding_sum = Embedding::new(users_and_user_vectors.len(), sum_of_bucket);

                for ((user, user_vector), _) in users_and_user_vectors {
                    out.push(((user.clone(), user_vector.clone(), embedding_sum.clone()), 1));
                }
            })
            .map(|(_bucket_key, (user, user_vector, bucket_sum))| {
                (user, (user_vector, bucket_sum))
            })
            .inspect(|x| println!("PREAGG {:?}", x));

        // Might make sense to join use_vectors with bucket sums instead of replicating the user vectors
        let recommendations = pre_aggregated_vectors
            .reduce(move |user, user_vectors_and_bucket_sums, out| {

                let mut neighbor_vector: CsVec<f64> =  CsVec::empty(num_items);

                // Weighted sum of bucket vectors
                for ((_, bucket_sum ), _) in user_vectors_and_bucket_sums {
                    // Hack
                    let weight = 1.0 / bucket_sum.id as f64;
                    let mut bucket_contribution = bucket_sum.clone().into_sparse_vector(num_items);
                    bucket_contribution.mul_assign(weight);
                    neighbor_vector = neighbor_vector + &bucket_contribution;
                }


                // TODO Subtract user vectors for correction
                // TODO Final recommendation as linear combination of user vector and neighbor vector

                let recommendation = Embedding::new(*user as usize, neighbor_vector);

                out.push((recommendation, 1));
            });

        // bucketed_user_vectors
        //     .map(|(key, _)| key)
        //     .count_total()
        //     .inspect(|x| println!("BUCKET {:?}", x))
        //     .probe()

        recommendations
            .inspect(|x| println!("RECO {:?}", x))
            .probe()

        // bucketed_user_vectors
        //     .inspect(|x| println!("BUCKETING {:?}", x))
        //     .probe()


        /*
        // let mut probe = Handle::new();
        //
        // let arranged_bucketed_user_vectors = bucketed_user_vectors
        //     .inspect(|x| println!("BUCKETING {:?}", x))
        //     .arrange_by_key();
        //
        // arranged_bucketed_user_vectors.stream.probe_with(&mut probe);
        //
        // (probe, arranged_bucketed_user_vectors.trace) */
    })
}