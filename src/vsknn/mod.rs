extern crate timely;
extern crate differential_dataflow;

pub mod types;
pub mod dataflow;

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use std::collections::hash_map::Entry;
use std::fmt::Display;

use timely::dataflow::operators::probe::Handle;
use timely::dataflow::operators::Probe;
use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::trace::{Cursor, TraceReader, BatchReader};
use differential_dataflow::operators::{Threshold, CountTotal};
use differential_dataflow::operators::arrange::ArrangeByKey;
use differential_dataflow::operators::reduce::ReduceCore;

use types::{SessionId, ItemId, ItemScore, Trace, OrderedSessionItem};
use self::timely::progress::frontier::AntichainRef;


pub fn update_recommendations(
    recommendations: &mut HashMap<SessionId, HashMap<ItemId, f64>>,
    time: usize,
    evolving_sessions_input: &mut InputSession<usize, (SessionId, ItemId), isize>,
    historical_sessions_input: &mut InputSession<usize, OrderedSessionItem, isize>,
    worker: &mut Worker<Allocator>,
    probe: &Handle<usize>,
    trace: &mut Trace<SessionId, ItemScore, usize, isize>,
    latency_in_micros: &mut u128
) -> usize {

    let start = Instant::now();

    evolving_sessions_input.advance_to(time);
    evolving_sessions_input.flush();
    historical_sessions_input.advance_to(time);
    historical_sessions_input.flush();

    worker.step_while(|| probe.less_than(evolving_sessions_input.time()) &&
        probe.less_than(historical_sessions_input.time()));

    let duration = start.elapsed();
    *latency_in_micros = duration.as_micros();

    // eprintln!(
    //     "\tWorker {} done with processing for time {} after {} micros.",
    //     worker.index(),
    //     time,
    //     duration.as_micros()
    // );

    let mut changed_keys = HashSet::new();

    let time_of_interest = time - 1;

    // TODO refactor this to take a closure
    trace.map_batches(|batch| {
        if batch.lower().elements().iter().find(|t| *(*t) == time_of_interest) != None {

            let mut cursor = batch.cursor();

            while cursor.key_valid(&batch) {
                while cursor.val_valid(&batch) {

                    let key = cursor.key(&batch);
                    let value = cursor.val(&batch);

                    cursor.map_times(&batch, |time, diff| {
                        if *time == time_of_interest && *diff < 0 {

                            assert_eq!((*diff).abs(), 1);
                            let (item, _score) = value;

                            if let Entry::Occupied(entry) = recommendations.entry(*key) {
                                changed_keys.insert(*key);
                                entry.into_mut().remove(&item);
                            }
                        }
                    });

                    cursor.step_val(&batch);
                }
                cursor.step_key(&batch);
            }
        }
    });

    trace.map_batches(|batch| {

        if batch.lower().elements().iter().find(|t| *(*t) == time_of_interest) != None {

            let mut cursor = batch.cursor();

            while cursor.key_valid(&batch) {
                while cursor.val_valid(&batch) {

                    let key = cursor.key(&batch);
                    let value = cursor.val(&batch);

                    cursor.map_times(&batch, |time, diff| {
                        if *time == time_of_interest && *diff > 0 {
                            assert_eq!((*diff).abs(), 1);

                            changed_keys.insert(*key);

                            let recommendations_for_session = recommendations.entry(*key)
                                .or_insert(HashMap::new());

                            let (item, score) = value;
                            recommendations_for_session.insert(*item, score.value);
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

pub fn vsknn<T>(
    worker: &mut Worker<Allocator>,
    historical_sessions_input: &mut InputSession<T, OrderedSessionItem, isize>,
    evolving_sessions_input: &mut InputSession<T, (SessionId, ItemId), isize>,
    k: usize,
    m: usize,
    num_total_sessions: usize,
) -> (ProbeHandle<T>, Trace<SessionId, ItemScore, T, isize>)
    where T: Timestamp + TotalOrder + Lattice + Refines<()> + Display {

    worker.dataflow(|scope| {

        let historical_sessions_with_duplicates = historical_sessions_input.to_collection(scope);
        let evolving_sessions_by_session = evolving_sessions_input.to_collection(scope);

        let mut probe = Handle::new();

        let (historical_session_max_order, historical_sessions_by_item, item_idfs) =
            dataflow::prepare(&historical_sessions_with_duplicates, m, num_total_sessions);

        let historical_sessions_arranged_by_session = historical_sessions_by_item
            .reduce_abelian("Reduce", |_item, sessions_with_multiplicities, output| {
                for (session, _) in sessions_with_multiplicities {
                    output.push((**session, 1))
                }
            });

        let evolving_sessions_by_item = evolving_sessions_by_session
            .map(|(session, item)| (item, session));

        let evolving_sessions_by_session_and_item = evolving_sessions_by_session
            .map(|(session, item)| ((session, item), ()));

        let evolving_session_lengths = evolving_sessions_by_session
            // We only want to count each interaction once
            .distinct()
            .map(|(session, _item)| (session))
            .count_total();


        let sampled_session_matches = dataflow::session_matches(
            &historical_sessions_by_item,
            &evolving_sessions_by_item,
            &historical_session_max_order,
            m
        );

        let similarities = dataflow::similarities(
            &sampled_session_matches,
            &historical_sessions_arranged_by_session,
            &evolving_sessions_by_session_and_item,
            &evolving_session_lengths,
            k
        );

        let weighted_item_scores = dataflow::item_scores(
            &similarities,
            &historical_sessions_arranged_by_session,
            &item_idfs
        );

        let arranged_item_scores = weighted_item_scores
            //.inspect(|((session, _), time, x)| eprintln!("{}: {} {}", *time, *session, x))
            .arrange_by_key();

        arranged_item_scores.stream.probe_with(&mut probe);

        (probe, arranged_item_scores.trace)
    })
}