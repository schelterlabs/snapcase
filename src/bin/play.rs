extern crate timely;
extern crate differential_dataflow;

use differential_dataflow::input::InputSession;
use differential_dataflow::operators::{Reduce};


fn main() {
    timely::execute_from_args(std::env::args(), move |worker| {

        let mut thingies_input: InputSession<_, (u32, u32),_> = InputSession::new();

        let probe = worker.dataflow(|scope| {

            let thingies = thingies_input.to_collection(scope);

            let groups = thingies.reduce(|_, values, out| {

                for (item, multiplicity) in values {
                    let group = (*multiplicity - 1) / 2;
                    println!("{} {}", *item, group);
                    out.push(((group, *item.clone()), 1));
                }
            });

            groups
                .map(|(basket, (group, item))| (group, (basket, item)))
                .inspect(|x| println!("INSPECT {:?}", x))
                .probe()
        });

        thingies_input.advance_to(1);

        thingies_input.update((1, 1), 1);
        thingies_input.update((1, 2), 2);
        thingies_input.update((1, 3), 3);
        thingies_input.update((1, 4), 4);

        thingies_input.advance_to(2);
        thingies_input.flush();

        worker.step_while(|| probe.less_than(thingies_input.time()));

        thingies_input.update((1, 3), -3);
        thingies_input.update((1, 4), -1);

        thingies_input.advance_to(3);
        thingies_input.flush();

        worker.step_while(|| probe.less_than(thingies_input.time()));


    }).unwrap();
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