extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket};
use snapcase::tifuknn::tifu_knn;
use snapcase::io::tifuknn::baskets_from_file;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let (baskets, _num_items) = baskets_from_file("./datasets/nbr/TaFang_history_NB.csv");

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input);

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