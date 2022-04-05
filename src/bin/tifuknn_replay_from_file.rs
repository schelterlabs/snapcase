extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::Basket;
use snapcase::tifuknn::tifu_knn;
use snapcase::io::tifuknn::{baskets_from_file, users_from_baskets};
use snapcase::tifuknn::hyperparams;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let (baskets, _num_items) = baskets_from_file("./datasets/nbr/VS_history_order.csv");
        let all_users = users_from_baskets(&baskets);

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();
        let mut query_users_input: InputSession<_, u32,_> = InputSession::new();

        // We want recommendations for all users here
        for user in all_users {
            query_users_input.insert(user);
        }

        let (probe, _) = tifu_knn(
            worker,
            &mut baskets_input,
            &mut query_users_input,
            hyperparams::PARAMS_VALUEDSHOPPER
        );

        baskets_input.advance_to(1);
        query_users_input.advance_to(1);

        for (user, basket, items) in baskets.iter() {
            if *user as usize % worker.peers() == worker.index() {
                baskets_input.update(
                    (*user, Basket::new(*basket, items.clone())),
                    *basket as isize
                );
            }
        }

        baskets_input.advance_to(2);
        query_users_input.advance_to(2);
        baskets_input.flush();
        query_users_input.flush();

        println!("Processing to t=2");

        worker.step_while(|| probe.less_than(baskets_input.time()) &&
            probe.less_than(query_users_input.time()));

    }).unwrap();
}