use crate::serialize::{MySerialize};
use super::VariableTensor;

impl MySerialize for VariableTensor {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        self.start.my_serialize(w);
        self.dim.my_serialize(w);
        self.step.my_serialize(w);
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        VariableTensor {
            start: u32::my_deserialize(r),
            dim: Box::<[u32]>::my_deserialize(r),
            step: Box::<[i32]>::my_deserialize(r),
        }
    }
}

impl MySerialize for Vec<VariableTensor> {
    fn my_serialize<W: std::io::Write>(&self, w : &mut W) {
        let len = self.len();
        len.my_serialize(w);
        for i in self {
            i.my_serialize(w);
        }
    }

    fn my_deserialize<R: std::io::Read>(r : &mut R) -> Self {
        let len = usize::my_deserialize(r);
        let mut data: Vec<VariableTensor> = Vec:: with_capacity(len);
        for _ in 0..len {
            data.push(VariableTensor::my_deserialize(r));
        }
        data
    }
}
