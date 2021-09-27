extern crate timely;
extern crate differential_dataflow;

use std::rc::Rc;
use std::cmp::Ord;
use differential_dataflow::trace::implementations::ord::OrdValBatch;
use differential_dataflow::trace::implementations::spine_fueled::Spine;
use differential_dataflow::operators::arrange::TraceAgent;
use super::sprs::CsVec;
use std::hash::Hash;
use std::collections::HashMap;

pub type Trace<K, V, T, R> = TraceAgent<Spine<K, V, T, R, Rc<OrdValBatch<K, V, T, R>>>>;

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
pub struct Embedding {
    pub id: usize,
    pub indices: Vec<usize>,
    pub data: Vec<u64>,
}

pub struct SparseVector {
    entries: HashMap<usize, f64>,
}

impl SparseVector {
    fn mult(&mut self, mult: f64) {
        for (_, val) in self.entries.iter_mut() {
            *val *= mult;
        }
    }

    fn plus_mult(&mut self, mult: f64, other: &Embedding) {
        for (index, other_val) in other.indices.iter().zip(other.data.iter()) {

        }
    }
}




const DISCRETISATION_FACTOR: f64 = 1_000_000_000.0;

impl Embedding {
    pub fn new(id: usize, vector: CsVec<f64>) -> Self {
        let (indices, data) = vector.into_raw_storage();
        // TODO we need to work with limited precision here to be able to compute equalities
        let discretised_data: Vec<u64> = data.iter()
            .map(|value| (value * DISCRETISATION_FACTOR) as u64)
            .collect();

        Embedding { id, indices, data: discretised_data }
    }

    pub fn into_sparse_vector(self, num_items: usize) -> CsVec<f64> {

        let undiscretised_data: Vec<f64> = self.data.iter()
            .map(|value| *value as f64 / DISCRETISATION_FACTOR)
            .collect();

        CsVec::new(num_items, self.indices, undiscretised_data)
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