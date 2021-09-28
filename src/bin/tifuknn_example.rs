extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, HyperParams};
use snapcase::tifuknn::tifu_knn;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let params = HyperParams {
            group_size: 7,
            r_group: 0.7,
            r_user: 0.9,
            random_seed: 42,
            k: 300,
            alpha: 0.7,
            num_permutation_functions: 1280,
            jaccard_threshold: 0.1,
        };

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input, params);

        baskets_input.advance_to(1);

        baskets_input.update((0, Basket::new(1, vec![0, 1, 2])), 1);
        baskets_input.update((0, Basket::new(2, vec![0, 1, 3])), 2);
        baskets_input.update((0, Basket::new(3, vec![0, 1, 3])), 3);
        baskets_input.update((0, Basket::new(4, vec![2, 3])), 4);

        baskets_input.update((1, Basket::new(5, vec![2])), 1);
        baskets_input.update((1, Basket::new(6, vec![0, 1 , 3])), 2);

        baskets_input.advance_to(2);
        baskets_input.flush();

        println!("Processing to t=2");

        worker.step_while(|| probe.less_than(baskets_input.time()));

        baskets_input.update((0, Basket::new(3, vec![0, 1, 3])), -3);
        baskets_input.update((0, Basket::new(4, vec![2, 3])), -1);

        baskets_input.advance_to(2);
        baskets_input.flush();

        println!("Processing to t=3");

        worker.step_while(|| probe.less_than(baskets_input.time()));

    }).unwrap();
}