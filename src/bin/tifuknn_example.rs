#[macro_use]
extern crate abomonation_derive;

extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;
use differential_dataflow::operators::{Reduce, Consolidate};
use itertools::enumerate;
use std::cmp::Ordering;

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash)]
struct Basket {
    seq: usize,
    items: (u32, u32, u32, u32),
}

impl Basket {
    fn new(time: usize, items: (u32, u32, u32, u32)) -> Self {
        Basket { seq: time, items }
    }
}

impl Ord for Basket {
    fn cmp(&self, other: &Self) -> Ordering {
        self.seq.cmp(&other.seq)
    }
}

impl PartialOrd for Basket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


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

        // TODO inputs should be sorted by the seq
        let probe = worker.dataflow(|scope| {
            let baskets = baskets_input.to_collection(scope);

            let groups = baskets
                .map(|basket| {
                    let group = basket.seq / 2;
                    (group, basket)
                });

            let group_vectors = groups
                .consolidate()
                .reduce(|_group, baskets, out| {

                    let mut group_vector = (0, 0, 0, 0);

                    for (basket, _) in baskets {

                        // TODO we could have an assertion that each multiplicity is only one

                        let r_b: f64 = 0.9;
                        let k_minus_i: i32 = 2 - ((basket.seq as i32 % 2) + 1);
                        let r_b_k_minus_i = r_b.powi(k_minus_i);

                        //println!("{:?}", r_b_k_minus_i);

                        group_vector.0 += ((basket.items.0 as f64 * r_b_k_minus_i / 2.0) * 100.0) as u64;
                        group_vector.1 += ((basket.items.1 as f64 * r_b_k_minus_i / 2.0) * 100.0) as u64;
                        group_vector.2 += ((basket.items.2 as f64 * r_b_k_minus_i / 2.0) * 100.0) as u64;
                        group_vector.3 += ((basket.items.3 as f64 * r_b_k_minus_i / 2.0) * 100.0) as u64;

                        //println!("{:?} {:?} {:?} {:?}", group, seq, k_minus_i, basket);
                    }

                    out.push((group_vector, 1));
                })
                .inspect(|x| println!("{:?}", x));

            let user_vectors = group_vectors
                .map(|(group, group_vector)| ((), (group, group_vector)))
                .reduce(|_, group_vectors, out| {

                    let mut user_vector = (0, 0, 0, 0);

                    let r_g: f64 = 0.5;

                    let num_groups = group_vectors.len();

                    // TODO ignores deletions at the moment
                    for (index, ((_, group_vector), _)) in enumerate(group_vectors) {

                        let m_minus_i = (num_groups - index + 1) as i32;
                        let r_g_m_minus_i = r_g.powi(m_minus_i);

                        //println!("{:?} {:?} {:?} {:?}", r_g_m_minus_i, m_minus_i, index, num_groups);

                        user_vector.0 += (group_vector.0 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                        user_vector.1 += (group_vector.1 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                        user_vector.2 += (group_vector.2 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                        user_vector.3 += (group_vector.3 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                    }

                    out.push((user_vector, 1));
                });

            user_vectors
                .inspect(|x| println!("{:?}", x))
                .probe()
        });

        for basket in baskets.iter() {
            baskets_input.insert(basket.clone());
        }

        baskets_input.advance_to(1);
        baskets_input.flush();

        worker.step_while(|| probe.less_than(baskets_input.time()));

    }).unwrap();
}