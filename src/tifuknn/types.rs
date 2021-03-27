extern crate timely;
extern crate differential_dataflow;

use std::cmp::Ordering;

use differential_dataflow::Hashable;
use std::hash::Hasher;
use std::rc::Rc;
use differential_dataflow::trace::implementations::ord::OrdValBatch;
use differential_dataflow::trace::implementations::spine_fueled::Spine;
use differential_dataflow::operators::arrange::TraceAgent;

pub type Trace<K, V, T, R> = TraceAgent<Spine<K, V, T, R, Rc<OrdValBatch<K, V, T, R>>>>;

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash)]
pub struct Basket {
    pub seq: usize,
    pub items: (u32, u32, u32, u32),
}

impl Basket {
    pub fn new(time: usize, items: (u32, u32, u32, u32)) -> Self {
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

#[derive(Abomonation, Debug, Clone)]
pub struct ProjectionMatrix {
    pub table_index: usize,
    pub weights: Vec<f64>,
}

impl ProjectionMatrix {
    pub fn new(table_index: usize, weights: Vec<f64>) -> ProjectionMatrix {
        ProjectionMatrix { table_index, weights }
    }
}

impl PartialEq for ProjectionMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.table_index == other.table_index
    }
}

impl Eq for ProjectionMatrix {}


impl PartialOrd for ProjectionMatrix {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.table_index.cmp(&other.table_index))
    }
}

impl Ord for ProjectionMatrix {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Hashable for ProjectionMatrix {
    type Output = u64;
    fn hashed(&self) -> u64 {
        let mut h: ::fnv::FnvHasher = Default::default();
        h.write_usize(self.table_index);
        h.finish()
    }
}