extern crate timely;
extern crate differential_dataflow;

use std::rc::Rc;
use std::cmp::Ord;
use differential_dataflow::trace::implementations::ord::OrdValBatch;
use differential_dataflow::trace::implementations::spine_fueled::Spine;
use differential_dataflow::operators::arrange::TraceAgent;
use std::hash::Hash;
use std::collections::HashMap;

pub type Trace<K, V, T, R> = TraceAgent<Spine<K, V, T, R, Rc<OrdValBatch<K, V, T, R>>>>;


#[derive(Debug, Clone, Copy)]
pub struct HyperParams {
    pub group_size: isize,
    pub r_group: f64,
    pub r_user: f64,
    pub random_seed: u64,
    pub k: usize,
    pub alpha: f64,
    pub num_permutation_functions: usize,
    pub jaccard_threshold: f64,
}

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash,Ord,PartialOrd)]
pub struct Basket {
    pub id: usize,
    pub items: Vec<usize>,
}

impl Basket {
    pub fn new(id: usize, items: Vec<usize>) -> Self {
        Basket { id, items }
    }
}

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash,Ord,PartialOrd)]
pub struct DiscretisedItemVector {
    pub id: usize,
    pub indices: Vec<usize>,
    pub data: Vec<u64>,
}

pub struct SparseItemVector {
    entries: HashMap<usize, f64>,
}

impl SparseItemVector {

    pub fn new() -> Self {
        SparseItemVector { entries: HashMap::new() }
    }

    pub fn mult(&mut self, mult: f64) {
        for (_, val) in self.entries.iter_mut() {
            *val *= mult;
        }
    }

    #[inline]
    pub fn plus_at(&mut self, index: usize, value: f64) {
        let entry = self.entries.entry(index).or_insert(0.0);
        *entry += value;
    }

    pub fn plus(&mut self, other: &DiscretisedItemVector) {
        // The compiler should remove the multiplication for us
        self.plus_mult(1.0, other);
    }

    pub fn plus_mult(&mut self, mult: f64, other: &DiscretisedItemVector) {
        for (index, other_val) in other.indices.iter().zip(other.data.iter()) {
            let to_add = (*other_val as f64 / DISCRETISATION_FACTOR) * mult;
            self.plus_at(*index, to_add)
        }
    }
}

const DISCRETISATION_FACTOR: f64 = 1_000_000_000.0;

impl DiscretisedItemVector {

    pub fn new(id: usize, vector: SparseItemVector) -> Self {
        // TODO optimise later
        let mut indices = Vec::with_capacity(vector.entries.len());
        let mut data = Vec::with_capacity(vector.entries.len());
        for (index, value) in vector.entries {
            indices.push(index);
            data.push((value * DISCRETISATION_FACTOR) as u64);
        }

        DiscretisedItemVector { id, indices, data }
    }

    pub fn print(&self) -> String {
        // TODO sort them
        self.indices.iter().zip(self.data.iter())
            .map(|(index, value)| format!("{}:{}", index, (*value as f64 / DISCRETISATION_FACTOR)))
            .collect::<Vec<_>>()
            .join(";")
    }
}

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash,Ord,PartialOrd)]
pub struct BucketKey {
    pub index: usize,
    pub hashes: Vec<u64>,
}

impl BucketKey {
    pub(crate) fn new(index: usize, hashes: Vec<u64>) -> Self {
        Self { index, hashes }
    }
}