extern crate csv;
extern crate blas;
extern crate openblas_src;
extern crate timely;
extern crate differential_dataflow;
extern crate sprs;

use differential_dataflow::input::InputSession;

use snapcase::tifuknn::types::{Basket};
use snapcase::tifuknn::tifu_knn;

use std::cmp::max;

fn main() {

    timely::execute_from_args(std::env::args(), move |worker| {

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b',')
            .from_path("/home/ssc/Entwicklung/projects/serenesia/datasets/next-basket/TaFang_history_NB.csv")
            .unwrap();

        let mut current_user: Option<u32> = None;
        let mut current_basket: Option<usize> = None;
        let mut baskets: Vec<(u32, usize, Vec<usize>)> = Vec::new();
        let mut buffer: Vec<usize> = Vec::new();

        let mut num_items = 0;

        for result in reader.deserialize() {
            let raw: (u32, usize, usize) = result.unwrap();
            //println!("{:?}", raw);

            let user = raw.0;
            let basket = raw.1;
            let item = raw.2 - 1;

            num_items = max(num_items, raw.2);

            if current_user.is_none() {
                current_user = Some(user);
            }
            if current_basket.is_none() {
                current_basket = Some(basket);
            }

            if user == current_user.unwrap() {
                if basket == current_basket.unwrap() {
                    buffer.push(item);
                } else {
                    baskets.push(
                        (current_user.unwrap().clone(),
                         current_basket.unwrap().clone(),
                         buffer.clone())
                    );
                    buffer.clear();
                    current_basket = Some(basket);
                    buffer.push(item);
                }
            } else {
                baskets.push(
                    (current_user.unwrap().clone(),
                     current_basket.unwrap().clone(),
                     buffer.clone())
                );
                buffer.clear();
                current_user = Some(user);
                current_basket = Some(basket);
                buffer.push(item);
            }
        }

        if !buffer.is_empty() {
            baskets.push(
                (current_user.unwrap().clone(),
                 current_basket.unwrap().clone(),
                 buffer.clone())
            );
        }

        println!("Found {} baskets for {} items", baskets.len(), num_items);

        let mut baskets_input: InputSession<_, (u32, Basket),_> = InputSession::new();

        let probe = tifu_knn(worker, &mut baskets_input, num_items);

        baskets_input.advance_to(1);

        for (user, basket, items) in baskets.iter() { //baskets.iter().take(10_000) {

            let mut cloned_items = items.clone();
            cloned_items.sort();

            baskets_input.update((*user, Basket::new(*basket, cloned_items)), *basket as isize)
        }

        // baskets_input.update((0, Basket::new(1, vec![0, 1, 2])), 1);
        // baskets_input.update((0, Basket::new(2, vec![0, 1, 3])), 2);
        // baskets_input.update((0, Basket::new(3, vec![0, 1, 3])), 3);
        // baskets_input.update((0, Basket::new(4, vec![2, 3])), 4);
        //
        // baskets_input.update((1, Basket::new(5, vec![2])), 1);
        // baskets_input.update((1, Basket::new(6, vec![0, 1 , 3])), 2);

        baskets_input.advance_to(2);
        baskets_input.flush();

        println!("Processing to t=2");

        worker.step_while(|| probe.less_than(baskets_input.time()));

        // baskets_input.update((0, Basket::new(3, vec![0, 1, 3])), -3);
        // baskets_input.update((0, Basket::new(4, vec![2, 3])), -1);
        //
        // baskets_input.advance_to(2);
        // baskets_input.flush();
        //
        // println!("Processing to t=3");
        //
        // worker.step_while(|| probe.less_than(baskets_input.time()));

    }).unwrap();

    // reader.deserialize()
    //     .for_each(move |result| {
    //         let raw: (u32, u32, u32) = result.unwrap();
    //         println!("{:?}", raw);
    //     })
}