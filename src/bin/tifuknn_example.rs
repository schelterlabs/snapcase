extern crate blas;
extern crate openblas_src;
extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket, ProjectionMatrix};
use snapcase::tifuknn::tifu_knn;

fn main() {
    
    timely::execute_from_args(std::env::args(), move |worker| {

        // Eventually we need the user id here too
        let baskets = vec![
            Basket::new(0, (1, 1, 1, 0)),
            Basket::new(1, (1, 1, 0, 1)),
            Basket::new(2, (1, 1, 0, 1)),
            Basket::new(3, (0, 0, 1, 1))
        ];

        let mut baskets_input: InputSession<_, Basket,_> = InputSession::new();
        let mut projection_matrices_input: InputSession<_, ProjectionMatrix, _> =
            InputSession::new();

        let (probe, _trace) = tifu_knn(worker, &mut baskets_input, &mut projection_matrices_input);

        projection_matrices_input.close();

        for basket in baskets.iter() {
            baskets_input.insert(basket.clone());
        }

        baskets_input.advance_to(1);
        baskets_input.flush();


        worker.step_while(|| probe.less_than(baskets_input.time()));

    }).unwrap();
}