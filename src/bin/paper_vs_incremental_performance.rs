extern crate differential_dataflow;
extern crate snapcase;
extern crate itertools;
extern crate fnv;

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::Rng;
#[allow(deprecated)] use rand::XorShiftRng;
use rand::prelude::*;

use itertools::Itertools;

use differential_dataflow::input::InputSession;

use snapcase::io;
use snapcase::vsknn::{vsknn, update_recommendations};
use snapcase::vsknn::types::{SessionId, ItemId, OrderedSessionItem, Order};


fn main() {

    let k: usize = 100;
    let m: usize = 500;
    let num_samples_to_add: usize = 10_000;

    for seed in [42] {//}, 767, 9000909] {
        for num_active_sessions in [100, 1000, 10_000] {
            for batch_size in [1, 10, 100] {
                run_experiment(
                    "ecom1m".to_owned(),
                    "./datasets/session/bolcom-clicks-1m_train.txt".to_owned(),
                    "./datasets/session/bolcom-clicks-1m_test.txt".to_owned(),
                    k,
                    m,
                    num_active_sessions,
                    num_samples_to_add,
                    batch_size,
                    seed,
                );

                run_experiment(
                    "rsc15".to_owned(),
                    "./datasets/session/rsc15-clicks_train_full.txt".to_owned(),
                    "./datasets/session/rsc15-clicks_test.txt".to_owned(),
                    k,
                    m,
                    num_active_sessions,
                    num_samples_to_add,
                    batch_size,
                    seed,
                );

                run_experiment(
                    "ecom60m".to_owned(),
                    "./datasets/session/bolcom-clicks-50m_train.txt".to_owned(),
                    "./datasets/session/bolcom-clicks-50m_test.txt".to_owned(),
                    k,
                    m,
                    num_active_sessions,
                    num_samples_to_add,
                    batch_size,
                    seed,
                );
            }
        }
    }
}

fn run_experiment(
    dataset_name: String,
    historical_sessions_file: String,
    evolving_sessions_file: String,
    k: usize,
    m: usize,
    num_active_sessions: usize,
    num_samples_to_add: usize,
    batch_size: usize,
    seed: u64,
) {
    timely::execute_from_args(std::env::args(), move |worker| {

        #[allow(deprecated)] let mut rng = XorShiftRng::seed_from_u64(seed);

        let mut historical_sessions = io::vsknn::read_historical_sessions_partitioned(
            &*historical_sessions_file,
            worker.index(),
            worker.peers(),
        );

        let evolving_sessions = io::vsknn::read_evolving_sessions_partitioned(
            &*evolving_sessions_file,
            worker.index(),
            worker.peers(),
        );

        let num_historical_sessions = historical_sessions.len();

        eprintln!("# Found {} interactions in historical sessions", historical_sessions.len());

        let mut historical_sessions_input: InputSession<_, OrderedSessionItem, _> =
            InputSession::new();
        let mut evolving_sessions_input: InputSession<_, (SessionId, ItemId), _> =
            InputSession::new();

        let (probe, mut trace) = vsknn(
            worker,
            &mut historical_sessions_input,
            &mut evolving_sessions_input,
            k,
            m,
            num_historical_sessions,
        );

        let (historical_sessions, clicks_to_add) = historical_sessions
            .split_at_mut(num_historical_sessions - num_samples_to_add);


        eprintln!("# Loading {} historical interactions", historical_sessions.len());

        for (session, item, order) in historical_sessions {
            historical_sessions_input.insert((*session, (*item, Order::new(*order))));
        }

        let mut recommmendations: HashMap<SessionId, HashMap<ItemId, f64>> = HashMap::new();
        let mut time = 0;

        let session_ids: Vec<_> = evolving_sessions.keys().sorted().collect();

        let num_active_sessions_per_worker = num_active_sessions / worker.peers();

        let random_session_ids =
            session_ids.choose_multiple(&mut rng, num_active_sessions_per_worker);

        for evolving_session_id in random_session_ids {

            let evolving_session_items = &evolving_sessions[evolving_session_id];

            let random_session_length = rng.gen_range(1, evolving_session_items.len());

            for session_length in 1..random_session_length {
                let current_item = &evolving_session_items[session_length - 1];

                evolving_sessions_input.update(
                    (**evolving_session_id, *current_item),
                    session_length as isize,
                );
            }
        }

        let mut latency_in_micros: u128 = 0;

        time += 1;
        update_recommendations(
            &mut recommmendations,
            time,
            &mut evolving_sessions_input,
            &mut historical_sessions_input,
            worker,
            &probe,
            &mut trace,
            &mut latency_in_micros,
        );

        eprintln!("# Indexed data for {} evolving sessions", num_active_sessions);

        let mut batch_counter = 0;

        for (session, item, order) in clicks_to_add {
            historical_sessions_input.insert((*session, (*item, Order::new(*order))));

            batch_counter += 1;

            if batch_counter == batch_size {
                let mut latency_in_micros: u128 = 0;

                time += 1;
                update_recommendations(
                    &mut recommmendations,
                    time,
                    &mut evolving_sessions_input,
                    &mut historical_sessions_input,
                    worker,
                    &probe,
                    &mut trace,
                    &mut latency_in_micros,
                );

                println!(
                    "vs,incremental_performance,{},{},{},{},{},{},{}",
                    dataset_name,
                    seed,
                    worker.index(),
                    worker.peers(),
                    batch_size,
                    num_active_sessions,
                    latency_in_micros
                );

                batch_counter = 0
            }
        }

    }).unwrap();
}
