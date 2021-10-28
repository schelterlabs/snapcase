extern crate differential_dataflow;
extern crate snapcase;
extern crate itertools;
extern crate fnv;

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::Rng;

use itertools::Itertools;

use differential_dataflow::input::InputSession;

use snapcase::io;
use snapcase::vsknn::{vsknn, update_recommendations};
use snapcase::vsknn::types::{SessionId, ItemId, OrderedSessionItem, Order};


fn main() {

    let k: usize = 100;
    let m: usize = 5000;
    let num_repetitions: usize = 1000;

    for num_active_sessions in [100, 1000, 10_000] {
        run_experiment(
            "ecom1m".to_owned(),
            "./datasets/session/bolcom-clicks-1m_train.txt".to_owned(),
            "./datasets/session/bolcom-clicks-1m_test.txt".to_owned(),
            k,
            m,
            num_active_sessions,
            num_repetitions
        );

        run_experiment(
            "rsc15".to_owned(),
            "./datasets/session/rsc15-clicks_train_full.txt".to_owned(),
            "./datasets/session/rsc15-clicks_test.txt".to_owned(),
            k,
            m,
            num_active_sessions,
            num_repetitions
        );

        run_experiment(
            "ecom60m".to_owned(),
            "./datasets/session/bolcom-clicks-50m_train.txt".to_owned(),
            "./datasets/session/bolcom-clicks-50m_test.txt".to_owned(),
            k,
            m,
            num_active_sessions,
            num_repetitions
        );
    }
}

fn run_experiment(
    dataset_name: String,
    historical_sessions_file: String,
    evolving_sessions_file: String,
    k: usize,
    m: usize,
    num_active_sessions: usize,
    num_repetitions: usize,
) {

    let historical_sessions = io::vsknn::read_historical_sessions(&*historical_sessions_file);
    let num_historical_sessions: usize = historical_sessions.len();

    timely::execute_from_args(std::env::args(), move |worker| {
        let historical_sessions = io::vsknn::read_historical_sessions_partitioned(
            &*historical_sessions_file,
            worker.index(),
            worker.peers(),
        );

        let evolving_sessions = io::vsknn::read_evolving_sessions_partitioned(
            &*evolving_sessions_file,
            worker.index(),
            worker.peers(),
        );

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

        eprintln!("# Loading {} historical interactions", historical_sessions.len());

        for (session, item, order) in &historical_sessions {
            historical_sessions_input.insert((*session, (*item, Order::new(*order))));
        }

        let mut recommmendations: HashMap<SessionId, HashMap<ItemId, f64>> = HashMap::new();
        let mut time = 0;

        let session_ids: Vec<_> = evolving_sessions.keys().sorted().collect();

        let random_session_ids =
            session_ids.choose_multiple(&mut rand::thread_rng(), num_active_sessions);

        for evolving_session_id in random_session_ids {
            let evolving_session_items = &evolving_sessions[evolving_session_id];

            let random_session_length =
                rand::thread_rng().gen_range(1, evolving_session_items.len());

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

        println!("# Indexed data for {} evolving sessions", num_active_sessions);

        let random_clicks =
            historical_sessions.choose_multiple(&mut rand::thread_rng(), num_repetitions);

        for (session, item, order) in random_clicks {
            historical_sessions_input.remove((*session, (*item, Order::new(*order))));

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
                "vs,deletion_performance,{},{},{}",
                dataset_name,
                num_active_sessions,
                latency_in_micros / 1000
            );
        }

    }).unwrap();
}
