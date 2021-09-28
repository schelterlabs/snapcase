extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, HyperParams};
use snapcase::tifuknn::tifu_knn;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let params = HyperParams {
            group_size: 2,
            r_group: 0.9,
            r_user: 0.7,
            random_seed: 42,
            k: 300,
            alpha: 0.7,
            num_permutation_functions: 1280,
            jaccard_threshold: 0.1,
        };

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input, params);

        baskets_input.advance_to(1);

        // v_{b_1} = [1 1 1 0]
        baskets_input.update((0, Basket::new(1, vec![0, 1, 2])), 1);

        baskets_input.advance_to(2);
        baskets_input.flush();

        println!("Processing to t=2");
        // v_{g_1} should be [1,1,1,0]
        // v_{u} should be [1,1,1,0]

        worker.step_while(|| probe.less_than(baskets_input.time()));

        // v_{b_2} = [0 1 1 1]
        baskets_input.update((0, Basket::new(1, vec![1, 2, 3])), 2);

        baskets_input.advance_to(3);
        baskets_input.flush();

        println!("Processing to t=3");
        // v_{g_1} should be [0.45,0.95,0.95,0.5]
        // v_{u} should be [0.45,0.95,0.95,0.5]

        worker.step_while(|| probe.less_than(baskets_input.time()));

        // v_{b_4} = [1 1 0 1]
        baskets_input.update((0, Basket::new(1, vec![0, 1, 3])), 3);

        baskets_input.advance_to(4);
        baskets_input.flush();

        println!("Processing to t=4");
        // v_{g_1} should not change
        // v_{g_2} should be [1,1,0,1]
        // v_{u} should be [0.66, 0.83, 0.33, 0.68]

        worker.step_while(|| probe.less_than(baskets_input.time()));

        // v_{b_4} = [0 0 1 1]
        baskets_input.update((0, Basket::new(1, vec![2, 3])), 4);

        baskets_input.advance_to(5);
        baskets_input.flush();

        println!("Processing to t=5");
        // v_{g_1} should not change
        // v_{g_2} should be [0.45,0.45,0.5,0.95]
        // v_{u} should be [0.38, 0.56, 0.58, 0.65]

        worker.step_while(|| probe.less_than(baskets_input.time()));

        // v_{b_5} = [1 1 1 1]
        baskets_input.update((0, Basket::new(1, vec![0, 1, 2, 3])), 5);

        baskets_input.advance_to(6);
        baskets_input.flush();

        println!("Processing to t=6");
        // v_{g_1} should not change
        // v_{g_2} should not change
        // v_{g_3} should be [1,1,1,1]
        // v_{u} should be [0.51, 0.60, 0.61, 0.64]

        worker.step_while(|| probe.less_than(baskets_input.time()));

    }).unwrap();
}