extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, HyperParams};
use snapcase::tifuknn::hyperparams::{PARAMS_INSTACART, PARAMS_TAFANG, PARAMS_VALUEDSHOPPER};
use snapcase::tifuknn::tifu_knn;
use snapcase::io::tifuknn::baskets_from_file;

use rand::seq::SliceRandom;
use std::time::Instant;

fn main() {

    let num_repetitions = 50;

    run_experiment(
        "valuedshopper".to_owned(),
        "./datasets/nbr/VS_history_order.csv".to_owned(),
        PARAMS_VALUEDSHOPPER,
        num_repetitions
    );

    run_experiment(
        "instacart".to_owned(),
        "./datasets/nbr/Instacart_history.csv".to_owned(),
        PARAMS_INSTACART,
        num_repetitions
    );

    run_experiment(
        "tafang".to_owned(),
        "./datasets/nbr/TaFang_history_NB.csv".to_owned(),
        PARAMS_TAFANG,
        num_repetitions
    );
}

fn run_experiment(
    dataset_name: String,
    dataset_path: String,
    hyperparams: HyperParams,
    num_repetitions: usize)
{
    timely::execute_from_args(std::env::args(), move |worker| {

        let (baskets, _num_items) = baskets_from_file(&dataset_path);

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input, hyperparams);

        baskets_input.advance_to(1);

        for (user, basket, items) in baskets.iter() {
            if *user as usize % worker.peers() == worker.index() {
                baskets_input.update(
                    (*user, Basket::new(*basket, items.clone())),
                    *basket as isize
                );
            }
        }

        baskets_input.advance_to(2);
        baskets_input.flush();

        println!("# Training TIFU-kNN model");

        worker.step_while(|| probe.less_than(baskets_input.time()));


        for run in 0..num_repetitions {

            // TODO this only works in single threaded mode at the moment, for multi-threaded
            // TODO execution, we would have to fix the seed
            let (random_user, random_basket, items) =
                baskets.choose(&mut rand::thread_rng()).unwrap();

            baskets_input.update(
                (*random_user, Basket::new(*random_basket, items.clone())),
                -1 * *random_basket as isize
            );

            for (user, basket, items) in baskets.iter() {
                if *user == *random_user && *basket > *random_basket {
                    baskets_input.update(
                        (*user, Basket::new(*basket, items.clone())),
                        -1 * *basket as isize
                    );

                    baskets_input.update(
                        (*user, Basket::new(*basket - 1, items.clone())),
                        *basket as isize - 1
                    );
                }
            }

            //println!("# Deleting basket {} of user {}", random_basket, random_user);
            baskets_input.advance_to(3 + run);
            baskets_input.flush();

            let now = Instant::now();
            worker.step_while(|| probe.less_than(baskets_input.time()));
            println!("tifu,deletion_performance,{},{}", dataset_name, now.elapsed().as_millis());
        }
    }).unwrap();
}