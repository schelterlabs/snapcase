extern crate timely;
extern crate differential_dataflow;

pub mod types;

use crate::tifuknn::types::{Basket, Trace};

use timely::dataflow::operators::probe::Handle;
use timely::dataflow::operators::Probe;
use timely::worker::Worker;
use timely::communication::Allocator;
use timely::dataflow::ProbeHandle;
use timely::progress::Timestamp;
use timely::progress::timestamp::Refines;
use timely::order::TotalOrder;

use differential_dataflow::input::InputSession;
use differential_dataflow::lattice::Lattice;
use differential_dataflow::operators::arrange::ArrangeByKey;

use crate::tifuknn::types::ProjectionMatrix;

use differential_dataflow::operators::{Reduce, Consolidate, Join};
use itertools::enumerate;

use blas::*;

use rand::distributions::Normal;
use rand::{thread_rng, Rng};

const NUM_HASH_DIMENSIONS: usize = 5;
const NUM_FEATURES: usize = 4;

pub fn tifu_knn<T>(
    worker: &mut Worker<Allocator>,
    baskets_input: &mut InputSession<T, Basket, isize>,
    projection_matrices_input: &mut InputSession<T, ProjectionMatrix, isize>,
) -> (ProbeHandle<T>, Trace<u32, (u64, u64, u64, u64), T, isize>)
    where T: Timestamp + TotalOrder + Lattice + Refines<()> {



    let mut rng = thread_rng();
    let distribution = Normal::new(0.0, 1.0 / NUM_HASH_DIMENSIONS as f64);

    let mut random_matrix: Vec<f64> =
        Vec::with_capacity(NUM_HASH_DIMENSIONS * NUM_FEATURES);

    for _ in 0..(NUM_HASH_DIMENSIONS * NUM_FEATURES) {
        let sampled: f64 = rng.sample(distribution);
        random_matrix.push(sampled);
    }
    let projection_matrix = ProjectionMatrix::new(1, random_matrix);

    projection_matrices_input.insert(projection_matrix);

    // TODO inputs should be sorted by the seq
    worker.dataflow(|scope| {
        let baskets = baskets_input.to_collection(scope);
        let projection_matrices = projection_matrices_input.to_collection(scope)
            .map(|matrix| ((), matrix));

        let groups = baskets
            .map(|basket| {
                let group = basket.seq / 2;
                (group, basket)
            });

        let group_vectors = groups
            .consolidate()
            .reduce(|_group, baskets, out| {

                let mut group_vector = (0, 0, 0, 0);

                for (basket, multiplicity) in baskets {
                    assert_eq!(*multiplicity, 1);

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

                for (index, ((_, group_vector), _)) in enumerate(group_vectors) {

                    let m_minus_i = (num_groups - index + 1) as i32;
                    let r_g_m_minus_i = r_g.powi(m_minus_i);

                    //println!("{:?} {:?} {:?} {:?}", r_g_m_minus_i, m_minus_i, index, num_groups);

                    user_vector.0 += (group_vector.0 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                    user_vector.1 += (group_vector.1 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                    user_vector.2 += (group_vector.2 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                    user_vector.3 += (group_vector.3 as f64 * r_g_m_minus_i / num_groups as f64) as u64;
                }

                out.push((user_vector, 1))
            })
            .map(|(_, user_vector)| user_vector);

        let bucketed_user_vectors = user_vectors
            .map(|user_vector| ((), user_vector))
            .join_map(&projection_matrices, |_, user_vector, projection_matrix| {

                let mut projection: Vec<f64> = vec![0.0; NUM_HASH_DIMENSIONS];

                let vector = [
                    user_vector.0 as f64,
                    user_vector.1 as f64,
                    user_vector.2 as f64,
                    user_vector.3 as f64,
                ];

                // Random projection of user vector

                unsafe {
                    dgemv(
                        b'T',
                        NUM_FEATURES as i32,
                        NUM_HASH_DIMENSIONS as i32,
                        1.0,
                        &projection_matrix.weights,
                        NUM_FEATURES as i32,
                        &vector,
                        1,
                        0.0,
                        &mut projection,
                        1);
                }

                // Signs of the result of the random projection give us the bucket key
                let mut key = 0u32;
                for (dimension, value) in projection.iter().enumerate() {
                    if *value > 0.0 {
                        key |= 1u32 << dimension as u32;
                    }
                }

                (key, *user_vector)
            });

        let mut probe = Handle::new();

        let arranged_bucketed_user_vectors = bucketed_user_vectors
            .inspect(|x| println!("{:?}", x))
            .arrange_by_key();

        arranged_bucketed_user_vectors.stream.probe_with(&mut probe);

        (probe, arranged_bucketed_user_vectors.trace)
    })
}