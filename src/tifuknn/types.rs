extern crate timely;
extern crate differential_dataflow;

use std::rc::Rc;
use std::cmp::{Ord, Ordering};
use differential_dataflow::trace::implementations::ord::OrdValBatch;
use differential_dataflow::trace::implementations::spine_fueled::Spine;
use differential_dataflow::operators::arrange::TraceAgent;
use super::sprs::CsVec;
use std::hash::{Hash, Hasher};
use itertools::zip;

pub type Trace<K, V, T, R> = TraceAgent<Spine<K, V, T, R, Rc<OrdValBatch<K, V, T, R>>>>;

#[derive(Eq,PartialEq,Debug,Abomonation,Clone,Hash,Ord, PartialOrd)]
pub struct Basket {
    pub id: usize,
    //pub items: (u32, u32, u32, u32),
    pub items: Vec<usize>,
}

impl Basket {
    pub fn new(id: usize, items: Vec<usize>) -> Self {
        Basket { id, items }
    }
}

#[derive(Debug,Abomonation,Clone)]
pub struct Embedding {
    pub id: usize,
    // Abomonation cannot handle the CsVecBase, so we work with the raw sparse vector here
    pub dim: usize,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl Embedding {
    pub fn new(id: usize, vector: CsVec<f64>) -> Self {
        let dim = vector.dim();
        let (indices, data) = vector.into_raw_storage();
        Embedding { id, dim, indices, data }
    }

    pub fn into_sparse_vector(self) -> CsVec<f64> {
        CsVec::new(self.dim, self.indices, self.data)
    }

    pub fn clone_into_dense_vector(&self) -> Vec<f64> {
        let mut dense_vector = vec![0.0; self.dim];
        for (index, value) in zip(&self.indices, &self.data) {
            dense_vector[*index] = *value;
        }
        dense_vector
    }
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Embedding {}

impl Ord for Embedding {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Embedding {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for Embedding {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id)
    }
}