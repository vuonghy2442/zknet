use super::{ConstraintSystem, Scalar, MemoryManager, Memory, TensorAddress, SCALAR_SIZE, BigScalar, RangeFull, Range, RangeFrom, RangeTo, Id, min, Functions};
use crate::scalar::power_of_two;

impl ConstraintSystem {
    pub fn run_conv2d_compact<T: Scalar>(mem: &MemoryManager, param: &[u32], var_dict: &mut Memory<T>) {
        let (mul_result, k_col, packed_size,bit_length,extracted) = (param[0], param[1], param[2], param[3], param[4]);
        let (fout, row_out, col_packed) = (mem[mul_result].dim[0],mem[mul_result].dim[1],mem[mul_result].dim[2]);

        let offset = power_of_two::<T>(bit_length - 1);
        let mut big_offset = T::zero();
        for _ in 0..packed_size + k_col - 1 {
            big_offset = (big_offset * T::from_i32(2) + T::one()) * offset;
        }
        let n_packed = packed_size + k_col - 1;
        for layer_out in 0..fout {
            //matching result
            for r in 0..row_out {
                for c in 0..col_packed {
                    let val = (var_dict[mem[mul_result].at_idx(&[layer_out, r, c]) as usize] + big_offset).to_bytes();
                    let mut ext = Vec::new();
                    ext.resize((packed_size + k_col - 1) as usize, T::zero());
                    for k in 0..(packed_size + k_col - 1) * bit_length {
                        ext[(k / bit_length) as usize] += T::from_i32(((val[(k/8) as usize] >> (k % 8)) & 1) as i32) * power_of_two(k % bit_length);
                    }
                    for k in 0..packed_size + k_col - 1 {
                        var_dict[mem[extracted].at_idx(&[layer_out,r,c * n_packed + k]) as usize] = ext[k as usize] - offset;
                    }
                }
            }
        }
    }

    pub fn conv2d_compact(&mut self, input: TensorAddress, output: TensorAddress, weight_rev: TensorAddress, bias: Option<(TensorAddress, u32)>, bit_length: u8) {
        // packing weight
        let dim = &self.mem[weight_rev].dim;
        let (fout, fin, k_row, k_col) = (dim[0], dim[1], dim[2], dim[3]);

        let packed_weight = self.mem.alloc(&[fout, fin, k_row]);
        assert!(k_col * (bit_length as u32) <= SCALAR_SIZE);
        self.packing_tensor(weight_rev, packed_weight, bit_length, k_col as u8,1, BigScalar::one(), true);

        let (row, col) = (self.mem[input].dim[1], self.mem[input].dim[2]);
        let packed_size = min((SCALAR_SIZE / (bit_length as u32)).checked_sub(k_col).unwrap(),col);
        let col_packed = (col-1)/packed_size + 1;
        let packed_layer = self.mem.alloc(&[fin, row, col_packed]);

        //packing row of inputs
        for layer in 0..fin {
            for r in 0..row {
                let input_row = self.mem.save(self.mem[input].at_(&[layer, r]));
                let packed_row = self.mem.save(self.mem[packed_layer].at_(&[layer, r]));
                self.packing_tensor(input_row, packed_row, bit_length, packed_size as u8,1, BigScalar::one(), true);
            }
        }

        // splicing output by row
        let mut mul_input = Vec::new();
        for r in 0..row - k_row + 1 {
            let mut mul_input_row = Vec::new();
            for c in 0..col_packed {
                mul_input_row.push(self.mem.save(self.mem[packed_layer].at(&[RangeFull(), Range(r..r + k_row), Id(c)])));
            }
            mul_input.push(mul_input_row);
        }

        //packing bias
        let mut packed_bias: Vec<Vec<TensorAddress>> = Vec::with_capacity(fout as usize);
        let mut bias_dim = 0;
        let mut bias_scale = 0;
        if let Some((b, scale)) = bias {
            bias_dim = (col - k_col)/packed_size + 1;
            bias_scale = scale;
            for layer_out in 0..fout {
                let mut packed_bias_row: Vec<TensorAddress> = Vec::with_capacity(((row - k_row)/scale + 1) as usize);
                for r in 0..(row - k_row)/scale + 1 {
                    let packed_bias = self.mem.alloc(&[bias_dim]);
                    let bias_row = self.mem.save(self.mem[b].at_(&[layer_out, r]));
                    self.packing_tensor(bias_row, packed_bias, bit_length, packed_size as u8, scale,power_of_two(bit_length as u32 * (k_col - 1)), true);
                    packed_bias_row.push(packed_bias);
                }
                packed_bias.push(packed_bias_row);
            }
        }

        let mul_result = self.mem.alloc(&[fout, row - k_row + 1, col_packed]);
        for layer_out in 0..fout {
            let packed_weight = self.mem.save(self.mem[packed_weight].at_(&[layer_out]));
            for r in 0..row - k_row + 1 {
                for c in 0..col_packed {
                    let cur_bias = if c < bias_dim {Some(self.mem[packed_bias[layer_out as usize][(r/bias_scale) as usize]].at_idx(&[c]))} else {None};
                    self.dot(mul_input[r as usize][c as usize], packed_weight, self.mem[mul_result].at_idx(&[layer_out, r, c]), cur_bias);
                }
            }
        }

        // sign extraction
        let n_packed = packed_size + k_col - 1;
        let extracted_length = (col_packed - 1) * n_packed + ((col-1) % packed_size) + k_col;
        let extracted = self.mem.alloc(&[fout, row - k_row + 1, extracted_length]);

        self.packing_tensor(extracted, mul_result, bit_length, n_packed as u8,1,BigScalar::one(), false);

        let params = vec![mul_result, k_col, packed_size, bit_length as u32, extracted];
        self.compute.push((params.into_boxed_slice(), Functions::ConvCompact));

        fn split_tensor<const N:usize>(mem: &mut MemoryManager,tensor: TensorAddress, length: u32, pos: [u32; N]) -> [(Option<TensorAddress>, Option<TensorAddress>); N] {
            let fully_packed = mem[tensor].dim[2]/length;
            let remainder = mem[tensor].dim[2] % length;

            // should not save this
            let tmp=mem[tensor].partition(2, length);

            let mut res: [(Option<TensorAddress>, Option<TensorAddress>); N] = [(None, None); N];
            for i in 0..N - 1 {
                let n= fully_packed + if remainder >= pos[i+1] {1} else {0};
                let full = if n > 0 {
                    Some(mem.save(tmp.at(&[RangeFull(), RangeFull(), RangeTo(..n), Range(pos[i]..pos[i+1])])))
                } else {
                    None
                };
                let rem = if pos[i] < remainder && remainder < pos[i+1] {
                    Some(mem.save(tmp.at(&[RangeFull(), RangeFull(), Id(n), Range(pos[i]..remainder)])))
                } else {
                    None
                };
                res[i] = (full, rem);
            }
            res
        }

        fn extract_sign_part(c: &mut ConstraintSystem, extracted: TensorAddress, bit_length: u8) {
            let output = c.mem.alloc(&c.mem[extracted].dim.to_owned());
            c.sign(extracted, output, bit_length - 1);
        }

        let reduced_extract = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), RangeTo(..extracted_length - k_col  + 1)]));
        let rem_extract = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), RangeFrom(extracted_length - k_col  + 1..)]));
        extract_sign_part(self, rem_extract, bit_length);

        let [(output_full, output_full_rem), (output_part, output_part_rem), (_,_)]= split_tensor(&mut self.mem, output, packed_size, [0, packed_size-(k_col-1), packed_size]);
        let [(ext_left, ext_left_rem), (ext_full, ext_full_rem), (ext_right,ext_right_rem), (_,_)]= split_tensor(&mut self.mem, reduced_extract, n_packed, [0, k_col-1, packed_size, n_packed]);

        // extract the fully correct part
        if let Some(e) = ext_full {
            self.sign(e, output_full.unwrap(), bit_length - 1);
        }

        if let Some(e) = ext_full_rem {
            self.sign(e, output_full_rem.unwrap(), bit_length - 1);
        }

        //extract left and right sign part
        if let Some(e) = ext_left {
            extract_sign_part(self,e, bit_length);
        }

        if let Some(e) = ext_left_rem {
            extract_sign_part(self,e, bit_length);
        }

        if let Some(e) = ext_right {
            extract_sign_part(self,e, bit_length);
        }

        assert_eq!(ext_right_rem, None);
        if let Some(left_rem) = ext_left_rem {
            if let Some(right) = ext_right {
                let sum_res = self.mem.alloc(&[fout, row - k_row + 1, self.mem[right].dim[2] - 1, k_col - 1]);
                let left = self.mem.save(self.mem[ext_left.unwrap()].at(&[RangeFull(), RangeFrom(1..)]));
                self.sum_two(right, left, sum_res);
                self.sign(sum_res, output_part.unwrap(), bit_length - 1);

                let sum_res = self.mem.alloc(&[fout, row - k_row + 1, self.mem[left_rem].dim[2]]);
                let right_rem = self.mem.save(self.mem[right].at(&[RangeFull(), Id(self.mem[right].dim[2] - 1), RangeTo(..self.mem[left_rem].dim[2])]));
                self.sum_two(right_rem, left_rem, sum_res);
                self.sign(sum_res, output_part_rem.unwrap(), bit_length - 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::slice_to_scalar;

    #[test]
    fn conv2d_compact_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,3,3]);
        let bias = x.mem.alloc(&[2,3,3]);


        let weight_rev = x.mem.save(x.mem[weight].reverse(3));

        x.conv2d_compact(input, output, weight_rev, Some((bias, 1)), 7);

        let mut mem: Vec<BigScalar> = slice_to_scalar(&[1,0,1,-1,0,0,0,-2,4,-1,-4,0,3,-4,0,0,0,1,-1,1,-4,2,3,-1,0,-4,2,2,-3,-1,-1,1,2,-1,1,4,4,2,3,-3,0,3,-2,3,0,2,3,3,-2,2,4,3,3,-4,-4,-1,3,1,4,-2,-2,0,-2,4,-3,0,0,0,-2,0,0,0,0,3,4,-3,-4,-1,-1,-4,3,1,-2,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,0,-3,0,1,-4,-1,2,0,0,-4,2,1,3,2,-3,4,-3]);
        mem.resize(x.mem.n_var as usize, Scalar::zero());

        x.compute(&mut mem);
        assert_eq!(mem[87..87+18], slice_to_scalar(&[1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }

    #[test]
    fn conv2d_compact_test_small() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[1,4,3]);
        let weight = x.mem.alloc(&[1,1,3,3]);
        let output = x.mem.alloc(&[1,2,1]);


        let weight_rev = x.mem.save(x.mem[weight].reverse(3));

        x.conv2d_compact(input, output, weight_rev, None, 5);
        let mut mem = x.mem.new_memory::<BigScalar>();
        x.load_memory(input, &mut mem, &slice_to_scalar(&[1,1,2, 1,2,1, 1,1,1, 1,2,1]));
        x.load_memory(weight, &mut mem, &slice_to_scalar(&[1,1,-1, 1,-1,1, 1,1,1]));

        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], slice_to_scalar(&[1,1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}