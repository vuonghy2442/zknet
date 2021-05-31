use std::iter::Iterator;
use std::ops::{Range, RangeFrom, RangeTo};

pub enum TensorIndex {
    Range(Range<u32>),
    RangeFrom(RangeFrom<u32>),
    RangeTo(RangeTo<u32>),
    RangeFull(),
    Id(u32)
}

#[derive(Clone)]
pub struct VariableTensor {
    start: u32,
    pub dim: Box<[u32]>,
    step: Box<[u32]>
}

pub struct VariableTensorIter {
    tensor: VariableTensor,
    pub idx: Vec<u32>,
    val_next : u32
}

impl Iterator for VariableTensorIter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx.len() == 0 {
            self.idx.resize(self.tensor.dim.len(), 0);
            self.val_next = self.tensor.start;
        } else {
            for i in (0..self.tensor.dim.len()).rev() {
                if self.idx[i] < self.tensor.dim[i] - 1 {
                    self.idx[i] += 1;
                    self.val_next += self.tensor.step[i];
                    break;
                } else {
                    if i == 0 {
                        return None;
                    }
                    self.idx[i] = 0;
                    self.val_next -= self.tensor.step[i] * (self.tensor.dim[i] - 1);
                }
            };
        }
        return Some(self.val_next);
    }
}

impl VariableTensor {
    pub fn new(start: u32, shape: &[u32]) -> VariableTensor {
        let mut step: Vec<u32> = Vec::new();
        step.resize(shape.len(), 0);
        let mut s = 1u32;
        for i in (0..shape.len()).rev() {
            step[i] = s;
            s *= shape[i];
        };
        VariableTensor {
            start,
            dim: shape.iter().cloned().collect(),
            step: step.into_boxed_slice()
        }
    }

    pub fn size(&self) -> u32 {
        let mut s = 1u32;
        for d in self.dim.iter() {
            s *= d;
        }
        s
    }

    pub fn at_idx(&self, idx: &[u32]) -> u32 {
        assert_eq!(self.dim.len(), idx.len());
        let mut res = self.start;
        for i in 0..idx.len() {
            assert!((0..self.dim[i]).contains(&idx[i]));
            res += idx[i] * self.step[i];
        }
        return res;
    }

    pub fn at_(&self, idx: &[u32]) -> VariableTensor {
        assert!(idx.len() < self.dim.len());

        let mut s = self.start;
        for i in 0..idx.len() {
            let pos = idx[i];
            assert!(pos < self.dim[i]);
            s += self.step[i] * pos;
        }

        VariableTensor {
            start: s,
            dim: self.dim[idx.len()..].to_vec().into_boxed_slice(),
            step: self.step[idx.len()..].to_vec().into_boxed_slice()
        }
    }

    pub fn at(&self, idx: &[TensorIndex]) -> VariableTensor {
        let mut s = self.start;
        let mut dim: Vec<u32> = Vec::new();
        let mut step: Vec<u32> = Vec::new();
        dim.reserve(self.dim.len());
        step.reserve(self.dim.len());
        for i in 0..idx.len() {
            match idx[i] {
                TensorIndex::Id(pos) => {
                    assert!(pos < self.dim[i]);
                    s += self.step[i] * pos;
                },
                TensorIndex::Range(Range {start, end}) => {
                    assert!(start < end && end <= self.dim[i]);
                    s += self.step[i] * start;
                    dim.push(end - start);
                    step.push(self.step[i]);
                }
                TensorIndex::RangeFrom(RangeFrom {start}) => {
                    assert!(start < self.dim[i]);
                    s += self.step[i] * start;
                    dim.push(self.dim[i] - start);
                    step.push(self.step[i]);
                }
                TensorIndex::RangeTo(RangeTo {end}) => {
                    assert!((1..=self.dim[i]).contains(&end));
                    dim.push(end);
                    step.push(self.step[i]);
                }
                TensorIndex::RangeFull() => {
                    dim.push(self.dim[i]);
                    step.push(self.step[i]);
                }
            };
        }

        for i in idx.len()..self.dim.len() {
            dim.push(self.dim[i]);
            step.push(self.step[i]);
        }

        assert!(dim.len() > 0);
        VariableTensor {
            start: s,
            dim: dim.into_boxed_slice(),
            step: step.into_boxed_slice()
        }
    }

    pub fn resize(&self, dim: &[u32]) -> VariableTensor {
        assert_eq!(self.dim.len(), dim.len());
        VariableTensor {
            start: self.start,
            dim: dim.to_vec().into_boxed_slice(),
            step: self.step.clone()
        }
    }

    pub fn squeeze(&self) -> VariableTensor {
        let start = self.start;
        let mut dim: Vec<u32> = Vec::new();
        let mut step: Vec<u32> = Vec::new();
        dim.reserve(self.dim.len());
        step.reserve(self.dim.len());
        for i in 0..self.dim.len() {
            if self.dim[i] > 1 {
                dim.push(self.dim[i]);
                step.push(self.step[i]);
            }
        };
        VariableTensor {
            start,
            dim: dim.into_boxed_slice(),
            step: step.into_boxed_slice()
        }
    }

    pub fn first(&self) -> u32 {
        self.start
    }

    pub fn iter(&self) -> VariableTensorIter {
        VariableTensorIter {
            tensor: self.squeeze(),
            idx: Vec::new(),
            val_next: self.start
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_tensor() {
        let x = VariableTensor::new(15, &[2,3,4]);
        assert_eq!(x.at_idx(&[0,0,0]), 15);
        assert_eq!(x.at_idx(&[0,0,3]), 18);
        assert_eq!(x.at_idx(&[1,0,3]), 30);

        for (x, y) in Iterator::zip(x.iter(), 15..(15+24)) {
            assert_eq!(x, y);
        }

        let x = x.at(&[TensorIndex::Id(1), TensorIndex::Range(1..2), TensorIndex::RangeFull()]);
        assert_eq!(x.dim.to_vec(), [1,4]);
        for (x, y) in Iterator::zip(x.iter(), 31..35) {
            assert_eq!(x, y);
        }
    }
}