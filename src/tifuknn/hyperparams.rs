use crate::tifuknn::types::HyperParams;

pub const PARAMS_VALUEDSHOPPER: HyperParams = HyperParams {
    group_size: 7,
    r_basket: 1.0,
    r_group: 0.6,
    random_seed: 42,
    k: 300,
    alpha: 0.7,
    num_permutation_functions: 1280,
    jaccard_threshold: 0.2,
};

pub const PARAMS_INSTACART: HyperParams = HyperParams {
    group_size: 3,
    r_basket: 0.9,
    r_group: 0.7,
    random_seed: 42,
    k: 900,
    alpha: 0.9,
    num_permutation_functions: 1280,
    jaccard_threshold: 0.1,
};


pub const PARAMS_TAFANG: HyperParams = HyperParams {
    group_size: 7,
    r_basket: 0.9,
    r_group: 0.7,
    random_seed: 42,
    k: 300,
    alpha: 0.7,
    num_permutation_functions: 1280,
    jaccard_threshold: 0.1,
};