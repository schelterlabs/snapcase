extern crate timely;
extern crate differential_dataflow;
extern crate sprs;

pub mod types;
pub mod aggregation;
pub mod projection;

use crate::tifuknn::types::Basket;

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

use differential_dataflow::operators::Reduce;

use rand::distributions::Normal;
use rand::{thread_rng, Rng};
use crate::tifuknn::aggregation::{group_vector, user_vector};

const NUM_HASH_DIMENSIONS: usize = 5;
const NUM_FEATURES: usize = 4;
const GROUP_SIZE: isize = 2;

pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
//) -> (ProbeHandle<T>, Trace<u32, (usize, (u64, u64, u64, u64)), T, isize>)
)   -> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    //let mut rng = thread_rng();
    //let distribution = Normal::new(0.0, 1.0 / NUM_HASH_DIMENSIONS as f64);


    worker.dataflow(|scope| {

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
            .reduce(|(_user, group), baskets_and_multiplicities, out| {
                let group_vector = group_vector(*group as usize, baskets_and_multiplicities, GROUP_SIZE, 0.9);
                // TODO we might have to enforce group vectors to be unique
                out.push((group_vector, *group));
            })
            .map(|((user, _group), group_vector)| (user, group_vector))
            .inspect(|x| println!("Group vector {:?}", x));

        let user_vectors = group_vectors
            .reduce(|user, vectors_and_multiplicities, out| {
                let user_vector = user_vector(*user, vectors_and_multiplicities, 0.5);
                out.push((user_vector, 1))
            })
            .inspect(|x| println!("User vector {:?}", x));

        let mut random_projection_matrix: Vec<f64> =
            Vec::with_capacity(NUM_HASH_DIMENSIONS * NUM_FEATURES);

        // TODO we must consistently seed this rng once we run with multiple workers
        let mut rng = thread_rng();
        let distribution = Normal::new(0.0, 1.0 / NUM_HASH_DIMENSIONS as f64);

        for _ in 0..(NUM_HASH_DIMENSIONS * NUM_FEATURES) {
            let sampled: f64 = rng.sample(distribution);
            random_projection_matrix.push(sampled);
        }

        let bucketed_user_vectors = user_vectors
            .map(move |(user, user_vector)| {
                // TODO we might have to normalise the vectors for this to give us valid results
                let key = projection::random_projection(&random_projection_matrix, &user_vector);
                (key, (user, user_vector))
            });
        //
        // // TODO now we can do a reduce per bucket to compute the actual recommendations
        //
        bucketed_user_vectors
            .inspect(|x| println!("BUCKETING {:?}", x))
            .probe()


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