extern crate timely;
extern crate differential_dataflow;
extern crate datasketch_minhash_lsh;

pub mod types;
pub mod aggregation;
pub mod dataflow;
pub mod hyperparams;

use std::fmt::Display;
use crate::tifuknn::types::{Basket, DiscretisedItemVector};
use std::time::Instant;
use std::collections::HashSet;

use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;
use timely::dataflow::operators::Probe;
use timely::dataflow::operators::probe::Handle;

use self::timely::progress::frontier::AntichainRef;

use differential_dataflow::trace::{Cursor, TraceReader, BatchReader};
use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::operators::arrange::ArrangeByKey;
use crate::tifuknn::types::HyperParams;
// TODO this should be in higher-level module
use crate::vsknn::types::Trace;



pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, (u32, Basket), isize>,
    query_users_input: &mut InputSession<T, u32, isize>,
    hyperparams: HyperParams,
//)   -> ProbeHandle<T>
) -> (ProbeHandle<T>, Trace<u32, DiscretisedItemVector, T, isize>)
    where T: Timestamp + TotalOrder + Lattice + Refines<()> + Display {

    worker.dataflow(|scope| {

        let baskets = baskets_input.to_collection(scope);
        let query_users = query_users_input.to_collection(scope);

        let mut probe = Handle::new();

        let user_vectors = dataflow::user_vectors(&baskets, hyperparams);
        let recommendations = dataflow::lsh_recommendations(
            &user_vectors,
            &query_users,
            hyperparams
        );

        let arranged_users_with_recommendations = recommendations
            //.map(|(user, _)| (user, 1_usize))
            //.inspect(|((user, _), time, x)| eprintln!("{}: {} {}", *time, *user, x))
            //.probe()
            .arrange_by_key();

        arranged_users_with_recommendations.stream.probe_with(&mut probe);

        (probe, arranged_users_with_recommendations.trace)
    })
}

pub fn update_recommendations(
    time: usize,
    baskets_input: &mut InputSession<usize, (u32, Basket), isize>,
    query_users_input: &mut InputSession<usize, u32, isize>,
    worker: &mut Worker<Allocator>,
    probe: &Handle<usize>,
    trace: &mut Trace<u32, DiscretisedItemVector, usize, isize>,
    latency_in_micros: &mut u128
) -> usize {

    let start = Instant::now();

    baskets_input.advance_to(time);
    baskets_input.flush();
    query_users_input.advance_to(time);
    query_users_input.flush();

    worker.step_while(|| probe.less_than(baskets_input.time()) &&
        probe.less_than(query_users_input.time()));

    let duration = start.elapsed();
    *latency_in_micros = duration.as_micros();


    let mut changed_keys = HashSet::new();

    let time_of_interest = time - 1;

    // TODO refactor this to take a closure
    trace.map_batches(|batch| {
        if batch.lower().elements().iter().find(|t| *(*t) == time_of_interest) != None {

            let mut cursor = batch.cursor();

            while cursor.key_valid(&batch) {
                while cursor.val_valid(&batch) {

                    let key = cursor.key(&batch);

                    cursor.map_times(&batch, |time, diff| {
                        if *time == time_of_interest && *diff != 0 {
                            assert_eq!((*diff).abs(), 1);
                            changed_keys.insert(*key);
                        }
                    });

                    cursor.step_val(&batch);
                }
                cursor.step_key(&batch);
            }
        }
    });

    // TODO We might want to do this at more coarse grained intervals
    let frontier_time = [time];
    let frontier = AntichainRef::new(&frontier_time);
    trace.set_physical_compaction(frontier);
    trace.set_logical_compaction(frontier);

    changed_keys.len()
}