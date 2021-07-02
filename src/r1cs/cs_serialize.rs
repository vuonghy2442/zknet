use curve25519_dalek::scalar::Scalar;

use crate::{serialize::{MySerialize, SimplySerialize}, tensor::VariableTensor};

use super::{ConstraintSystem, Functions, MemoryManager};

impl SimplySerialize for Functions {}

impl MySerialize for MemoryManager {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.n_var.my_serialize(w);
        self.one_var.my_serialize(w);
        self.mem_dict.my_serialize(w);
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        MemoryManager {
            n_var : u32::my_deserialize(r),
            one_var : u32::my_deserialize(r),
            mem_dict: Vec::<VariableTensor>::my_deserialize(r)
        }
    }
}

impl MySerialize for Vec<(Box<[u32]>, Functions)> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let len = self.len();
        len.my_serialize(w);
        for i in self {
            i.0.my_serialize(w);
            i.1.my_serialize(w);
        }
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut data: Vec<(Box<[u32]>, Functions)> = Vec:: with_capacity(len);
        for _ in 0..len {
            data.push((Box::<[u32]>::my_deserialize(r), Functions::my_deserialize(r)));
        }
        data
    }
}

impl SimplySerialize for Scalar {}

impl MySerialize for Vec<(u32, u32, Scalar)> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let len = self.len();
        len.my_serialize(w);
        for i in self {
            i.0.my_serialize(w);
            i.1.my_serialize(w);
            i.2.my_serialize(w);
        }
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut data: Vec<(u32, u32, Scalar)> = Vec:: with_capacity(len);
        for _ in 0..len {
            data.push((u32::my_deserialize(r), u32::my_deserialize(r), Scalar::my_deserialize(r)));
        }
        data
    }
}

impl MySerialize for ConstraintSystem {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.a.my_serialize(w);
        self.b.my_serialize(w);
        self.c.my_serialize(w);
        self.n_cons.my_serialize(w);
        self.mem.my_serialize(w);
        self.compute.my_serialize(w);
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        ConstraintSystem {
            a: Vec::<(u32, u32, Scalar)>::my_deserialize(r),
            b: Vec::<(u32, u32, Scalar)>::my_deserialize(r),
            c: Vec::<(u32, u32, Scalar)>::my_deserialize(r),
            n_cons: u32::my_deserialize(r),
            mem: MemoryManager::my_deserialize(r),
            compute: Vec::<(Box<[u32]>, Functions)>::my_deserialize(r),
        }
    }
}