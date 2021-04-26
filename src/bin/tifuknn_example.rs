extern crate blas;
extern crate openblas_src;
extern crate timely;
extern crate differential_dataflow;
extern crate sprs;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket};
use snapcase::tifuknn::tifu_knn;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let num_items: usize = 4;

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input, num_items);

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