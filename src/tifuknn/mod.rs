extern crate timely;
extern crate differential_dataflow;
extern crate sprs;

pub mod types;
pub mod aggregation;
pub mod minhash;

use crate::tifuknn::types::{Basket, BucketKey};

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

use differential_dataflow::operators::{Reduce,CountTotal};

use rand::thread_rng;
use crate::tifuknn::aggregation::{group_vector, user_vector};

use rand::seq::SliceRandom;

const GROUP_SIZE: isize = 7;
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
            ;//.inspect(|x| println!("Group vector {:?}", x));

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
            ;//.inspect(|x| println!("User vector {:?}", x));

        let permutations: Vec<Vec<usize>> = (0..(BANDS * BUCKEY_KEY_LENGTH))
            .map(|_| {
                let mut permutation: Vec<usize> = (0..num_items).collect();
                // TODO we must consistently seed this rng once we run with multiple workers
                permutation.shuffle(&mut thread_rng());
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


        bucketed_user_vectors
            .map(|(key, _)| key)
            .count_total()
            .inspect(|x| println!("BUCKET {:?}", x))
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