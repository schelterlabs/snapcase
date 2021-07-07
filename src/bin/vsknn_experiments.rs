extern crate differential_dataflow;
extern crate snapcase;
extern crate itertools;
extern crate fnv;

use std::collections::HashMap;

use std::io::prelude::*;

use itertools::Itertools;

use differential_dataflow::input::InputSession;

use snapcase::io;
use snapcase::vsknn::{vsknn, update_recommendations};
use snapcase::vsknn::types::{SessionId, ItemId, OrderedSessionItem, Order};


fn main() {
    let historical_sessions_file = std::env::args().nth(2)
        .expect("Historical sessions file not specified!");

    let evolving_sessions_file = std::env::args().nth(3)
        .expect("Evolving sessions file not specified!");

    let latencies_output_file = std::env::args().nth(4)
        .expect("Latencies output file not specified!");

    let k: usize = std::env::args().nth(5)
        .expect("Number of neighbors not specified!").parse()
        .expect("Cannot parse number of neighbors");

    let m: usize = std::env::args().nth(6)
        .expect("Sample size not specified!").parse()
        .expect("Cannot parse sample size");

    let batch_size: usize = std::env::args().nth(7)
        .expect("Batch size not specified!").parse()
        .expect("Cannot parse batch size");

    let num_deletions: usize = std::env::args().nth(8)
        .expect("Deletions not specified!").parse()
        .expect("Cannot parse deletions");

    let historical_sessions = io::vsknn::read_historical_sessions(&*historical_sessions_file);
    let num_historical_sessions: usize = historical_sessions.len();

    timely::execute_from_args(std::env::args(), move |worker| {
        let historical_sessions = io::vsknn::read_historical_sessions_partitioned(
            &*historical_sessions_file,
            worker.index(),
            worker.peers(),
        );

        let mut historical_sessions_by_id: HashMap<SessionId, Vec<(ItemId, Order)>> = HashMap::new();

        for (session, item, order) in &historical_sessions {
            let bucket = historical_sessions_by_id.entry(*session).or_insert(Vec::new());
            bucket.push((item.clone(), Order::new(*order)));
        }
        let mut hkeys: Vec<SessionId> = historical_sessions_by_id.keys().map(|k| *k).collect();
        hkeys.sort();

        let evolving_sessions = io::vsknn::read_evolving_sessions_partitioned(
            &*evolving_sessions_file,
            worker.index(),
            worker.peers(),
        );

        eprintln!("Found {} interactions in historical sessions", historical_sessions.len());

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

        let mut latency_writer: Option<_> = None;

        if worker.index() == 0 {
            let mut writer = io::vsknn::create_linewriter_file(&*latencies_output_file).unwrap();
            writer.write(("batch,session_length,latency_in_micros\n").as_ref()).unwrap();
            latency_writer = Some(writer);
        }

        eprintln!("Loading {} historical interactions", historical_sessions.len());

        for (session, item, order) in &historical_sessions {
            historical_sessions_input.insert((*session, (*item, Order::new(*order))));
        }

        //historical_sessions_input.close();

        let mut recommmendations: HashMap<SessionId, HashMap<ItemId, f64>> = HashMap::new();
        let mut time = 0;

        let session_ids: Vec<_> = evolving_sessions.keys().sorted().collect();

        let mut batch_index = 0;

        let batch_size_per_worker = batch_size / worker.peers();

        eprintln!(
            "Processing {} evolving sessions with a batch size of {} in {} workers",
            session_ids.len(),
            batch_size_per_worker,
            worker.peers()
        );

        for session_ids_batch in session_ids.chunks(batch_size_per_worker) {

            let max_session_length: usize = 10;/*session_ids_batch.iter()
                .map(|id| evolving_sessions[id].len())
                .max().unwrap();*/

            for session_length in 1..max_session_length {
                for evolving_session_id in session_ids_batch {
                    let evolving_session_items = &evolving_sessions[evolving_session_id];

                    if evolving_session_items.len() >= session_length {
                        let current_item = &evolving_session_items[session_length - 1];

                        evolving_sessions_input.update(
                            (**evolving_session_id, *current_item),
                            session_length as isize,
                        );
                    }

                    if session_length == 3 || session_length == 7 {

                        let deletions_per_worker = num_deletions / worker.peers();

                        let history_to_forget = (batch_index * max_session_length + session_length) * deletions_per_worker;
                        for index in 0..deletions_per_worker {
                            let id: u32 = hkeys[history_to_forget + index];
                            for (item, order) in &historical_sessions_by_id[&id] {
                                historical_sessions_input.remove((id, (*item, *order)));
                            }
                        }
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

                if worker.index() == 0 {

                    let mut writer = latency_writer.unwrap();

                    writer.write(
                        format!("{},{},{}\n", batch_index, session_length, latency_in_micros).as_ref()
                    ).unwrap();

                    latency_writer = Some(writer);
                }
            }

            batch_index += 1;
            recommmendations.clear();
        }
        if worker.index() == 0 {
            latency_writer.unwrap().flush().unwrap();
        }
    }).unwrap();
}