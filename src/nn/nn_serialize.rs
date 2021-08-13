use std::{collections::HashMap};

use crate::{r1cs::{ConstraintSystem, TensorAddress}, serialize::{MySerialize, SimplySerialize}};
use super::{AccuracyParams, NeuralNetwork};

impl SimplySerialize for AccuracyParams {}
impl MySerialize for HashMap<String, TensorAddress> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.len().my_serialize(w);
        for (k, v) in self {
            k.my_serialize(w);
            v.my_serialize(w);
        }
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let mut map: HashMap<String, TensorAddress> = HashMap::new();
        let len = usize::my_deserialize(r);
        for _ in 0..len {
            map.insert(String::my_deserialize(r), u32::my_deserialize(r));
        }
        map
    }
}
impl MySerialize for NeuralNetwork {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.cons.my_serialize(w);
        self.weight_map.my_serialize(w);
        self.input.my_serialize(w);
        self.output.my_serialize(w);
        self.commit_hash.my_serialize(w);
        self.commit_open.my_serialize(w);
        self.acc.my_serialize(w);
        self.scaling.my_serialize(w);
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        Self {
            cons: ConstraintSystem::my_deserialize(r),
            weight_map: HashMap::my_deserialize(r),
            input: TensorAddress::my_deserialize(r),
            output: TensorAddress::my_deserialize(r),
            commit_hash: TensorAddress::my_deserialize(r),
            commit_open: TensorAddress::my_deserialize(r),
            acc: Option::my_deserialize(r),
            scaling: f64::my_deserialize(r),
        }
    }
}