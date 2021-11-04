extern crate timely;
extern crate differential_dataflow;
extern crate datasketch_minhash_lsh;

pub mod types;
pub mod aggregation;
pub mod dataflow;
pub mod hyperparams;

use crate::tifuknn::types::Basket;

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use crate::tifuknn::types::HyperParams;

pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
    query_users_input: &mut InputSession<T, u32, isize>,
    hyperparams: HyperParams,
)   -> ProbeHandle<T>
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {

    worker.dataflow(|scope| {

        let baskets = baskets_input.to_collection(scope);
        let query_users = query_users_input.to_collection(scope);

        let user_vectors = dataflow::user_vectors(&baskets, hyperparams);
        let recommendations = dataflow::lsh_recommendations(
            &user_vectors,
            &query_users,
            hyperparams
        );

        recommendations
            .probe()
    })
}