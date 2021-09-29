extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, HyperParams};
use snapcase::tifuknn::tifu_knn;
use snapcase::io::tifuknn::baskets_from_file;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let (baskets, _num_items) = baskets_from_file("./datasets/nbr/TaFang_history_NB.csv");

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let params = HyperParams {
            group_size: 7,
            r_basket: 0.9,
            r_group: 0.7,
            random_seed: 42,
            k: 300,
            alpha: 0.7,
            num_permutation_functions: 1280,
            jaccard_threshold: 0.1,
        };

        let probe = tifu_knn(worker, &mut baskets_input, params);

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

        println!("Processing to t=2");

        worker.step_while(|| probe.less_than(baskets_input.time()));


    }).unwrap();
}