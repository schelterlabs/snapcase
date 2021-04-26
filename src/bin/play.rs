use rand::thread_rng;
use rand::seq::SliceRandom;

fn minhash(permutations: &[Vec<usize>], set: &[usize], num_items: usize) -> Vec<usize> {

    let mut hashes = Vec::with_capacity(num_items);

    for permutation in permutations {
        'search: for index in permutation {
            if set.contains(index) {
                hashes.push(*index);
                break 'search;
            }
        }
    }
    assert_eq!(hashes.len(), permutations.len());
    hashes
}

fn main () {

    let num_projections = 2;
    let num_items = 7;

    let permutations: Vec<Vec<usize>> = (0..num_projections)
        .map(|_| {
            let mut permutation: Vec<usize> = (0..num_items).collect();
            permutation.shuffle(&mut thread_rng());
            permutation
        })
        .collect();

    //let mut pi: Vec<usize> = (0..num_items).collect();
    //pi.shuffle(&mut thread_rng());
    println!("{:?}", permutations);

    let vec1 = vec![1, 3, 5, 6];//0, 1, 0, 1, 0, 1, 1];
    let vec2 = vec![1, 5, 6];//[0, 1, 0, 0, 0, 1, 1];
    let vec3 = vec![0, 4];//1, 0, 0, 0, 1, 0, 0];

    println!("{:?} -> {:?}", vec1, minhash(&permutations, &vec1, num_items));
    println!("{:?} -> {:?}", vec2, minhash(&permutations, &vec2, num_items));
    println!("{:?} -> {:?}", vec3, minhash(&permutations, &vec3, num_items));
}
// fn main() {
//     timely::execute_from_args(std::env::args(), move |worker| {
//
//         let mut thingies_input: InputSession<_, (u32, u32),_> = InputSession::new();
//
//         let probe = worker.dataflow(|scope| {
//
//             let thingies = thingies_input.to_collection(scope);
//
//             let groups = thingies.reduce(|_, values, out| {
//
//                 for (item, multiplicity) in values {
//                     let group = (*multiplicity - 1) / 2;
//                     println!("{} {}", *item, group);
//                     out.push(((group, *item.clone()), 1));
//                 }
//             });
//
//             groups
//                 .map(|(basket, (group, item))| (group, (basket, item)))
//                 .inspect(|x| println!("INSPECT {:?}", x))
//                 .probe()
//         });
//
//         thingies_input.advance_to(1);
//
//         thingies_input.update((1, 1), 1);
//         thingies_input.update((1, 2), 2);
//         thingies_input.update((1, 3), 3);
//         thingies_input.update((1, 4), 4);
//
//         thingies_input.advance_to(2);
//         thingies_input.flush();
//
//         worker.step_while(|| probe.less_than(thingies_input.time()));
//
//         thingies_input.update((1, 3), -3);
//         thingies_input.update((1, 4), -1);
//
//         thingies_input.advance_to(3);
//         thingies_input.flush();
//
//         worker.step_while(|| probe.less_than(thingies_input.time()));
//
//
//     }).unwrap();
// }

// fn main() {
//     timely::execute_from_args(std::env::args(), move |worker| {
//
//         let mut thingies_input: InputSession<_, (u32, u32),_> = InputSession::new();
//
//         let probe = worker.dataflow(|scope| {
//
//             let thingies = thingies_input.to_collection(scope);
//
//             let key_counts = thingies.reduce(|key, values, out| {
//
//                 println!("Processing key {}", key);
//                 let mut count: isize = 0;
//                 for (value, mult) in values {
//                     println!("\tUpdate {} {}", value, mult);
//                     count += mult;
//                 }
//
//                 out.push(((), count));
//             });
//
//             key_counts
//                 .map(|(key, _)| key)
//                 .inspect(|x| println!("INSPECT {:?}", x))
//                 .probe()
//         });
//
//         let stuff = vec![(1, 5), (1, 7),  (2, 3)];
//
//         for (key, value) in stuff.iter() {
//             thingies_input.insert((*key, *value));
//         }
//
//         thingies_input.advance_to(1);
//         thingies_input.flush();
//
//         //worker.step_while(|| probe.less_than(thingies_input.time()));
//         //println!("Next round");
//
//         let stuff2 = vec![(1, 3),  (2, 6)];
//
//         for (key, value) in stuff2.iter() {
//             thingies_input.insert((*key, *value));
//         }
//
//         let stuff2_remove = vec![(1, 5, 2)];
//
//         for (key, value, mult) in stuff2_remove.iter() {
//             thingies_input.update((*key, *value), -1 * mult);
//         }
//
//         thingies_input.advance_to(2);
//         thingies_input.flush();
//
//         worker.step_while(|| probe.less_than(thingies_input.time()));
//
//     }).unwrap();
// }