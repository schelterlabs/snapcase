extern crate timely;
extern crate differential_dataflow;
extern crate datasketch_minhash_lsh;

pub mod types;
pub mod aggregation;
pub mod dataflow;

use crate::tifuknn::types::Basket;

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;

use datasketch_minhash_lsh::Weights;

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

pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
)   -> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    worker.dataflow(|scope| {

        let baskets = baskets_input.to_collection(scope);

        let user_vectors = dataflow::user_vectors(&baskets);
        let recommendations = dataflow::lsh_recommendations(&user_vectors);

        recommendations
            //.inspect(|x| println!("RECO {:?}", x))
            .probe()
    })
}