extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, HyperParams};
use snapcase::tifuknn::hyperparams::{PARAMS_INSTACART, PARAMS_TAFANG, PARAMS_VALUEDSHOPPER};
use snapcase::tifuknn::{tifu_knn, update_recommendations};
use snapcase::io::tifuknn::{baskets_from_file, users_from_baskets};

use rand::seq::SliceRandom;
#[allow(deprecated)] use rand::XorShiftRng;
use rand::prelude::*;

fn main() {

    let num_baskets_to_add = 1000;

    for seed in [42, 767, 999] {
        for num_query_users in [10, 100, 1000] {
            for batch_size in [1, 10, 100] {

                run_experiment(
                    "valuedshopper".to_owned(),
                    "./datasets/nbr/VS_history_order.csv".to_owned(),
                    seed,
                    num_query_users,
                    PARAMS_VALUEDSHOPPER,
                    num_baskets_to_add,
                    batch_size
                );

                run_experiment(
                    "instacart".to_owned(),
                    "./datasets/nbr/Instacart_history.csv".to_owned(),
                    seed,
                    num_query_users,
                    PARAMS_INSTACART,
                    num_baskets_to_add,
                    batch_size
                );

                run_experiment(
                    "tafang".to_owned(),
                    "./datasets/nbr/TaFang_history_NB.csv".to_owned(),
                    seed,
                    num_query_users,
                    PARAMS_TAFANG,
                    num_baskets_to_add,
                    batch_size
                );
            }
        }
    }
}

fn run_experiment(
    dataset_name: String,
    dataset_path: String,
    seed: u64,
    num_query_users: usize,
    hyperparams: HyperParams,
    num_baskets_to_add: usize,
    batch_size: usize
)
{
    eprintln!(
        "tifu_incremental,dataset={},seed={},queries={}, batch_size={}",
        dataset_name,
        seed,
        num_query_users,
        batch_size
    );


    timely::execute_from_args(std::env::args(), move |worker| {

        #[allow(deprecated)] let mut rng = XorShiftRng::seed_from_u64(seed);

        let (mut baskets, _) = baskets_from_file(&dataset_path);

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();
        let mut query_users_input: InputSession<_, u32,_> = InputSession::new();

        let (probe, mut trace) = tifu_knn(worker, &mut baskets_input, &mut query_users_input, hyperparams);

        baskets_input.advance_to(1);
        query_users_input.advance_to(1);

        let all_users = users_from_baskets(&baskets);
        let query_users = all_users.choose_multiple(&mut rng, num_query_users);

        for query_user in query_users {
            query_users_input.insert(*query_user);
        }

        let num_baskets = baskets.len();
        let (baskets, baskets_to_add) = baskets.split_at_mut(num_baskets - num_baskets_to_add);

        for (user, basket, items) in baskets.iter() {
            if *user as usize % worker.peers() == worker.index() {
                baskets_input.update(
                    (*user, Basket::new(*basket, items.clone())),
                    *basket as isize
                );
            }
        }

        // baskets_input.advance_to(2);
        // query_users_input.advance_to(2);
        // baskets_input.flush();
        // query_users_input.flush();

        eprintln!(
            "# Training TIFU-kNN model for {} query users and batch size {}",
            num_query_users,
            batch_size
        );

        let mut latency_setup = 0_u128;
        let _ = update_recommendations(
            2,
            &mut baskets_input,
            &mut query_users_input,
            worker,
            &probe,
            &mut trace,
            &mut latency_setup
        );

        //
        // worker.step_while(|| probe.less_than(baskets_input.time()) &&
        //     probe.less_than(query_users_input.time()));

        let mut batch_counter = 0;

        for run in 0..num_baskets_to_add {
            let (user, basket, items) = baskets_to_add.get(run).unwrap();

            baskets_input.update((*user, Basket::new(*basket, items.clone())), *basket as isize);

            batch_counter += 1;

            if batch_counter == batch_size {

                // baskets_input.advance_to(3 + run);
                // query_users_input.advance_to(3 + run);
                // baskets_input.flush();
                // query_users_input.flush();
                //
                // let now = Instant::now();
                // worker.step_while(|| probe.less_than(baskets_input.time()) &&
                //     probe.less_than(query_users_input.time()));
                // let latency_in_micros = now.elapsed().as_micros();

                let mut latency_in_micros = 0_u128;

                let num_updates = update_recommendations(
                    3 + run,
                    &mut baskets_input,
                    &mut query_users_input,
                    worker,
                    &probe,
                    &mut trace,
                    &mut latency_in_micros
                );

                println!(
                    "tifu,incremental_performance,{},{},{},{},{}",
                    dataset_name,
                    num_query_users,
                    batch_size,
                    latency_in_micros,
                    num_updates
                );

                batch_counter = 0;
            }
        }
    }).unwrap();
}