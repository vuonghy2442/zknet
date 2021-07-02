use super::{ConstraintSystem, MemoryManager, TensorAddress, SCALAR_SIZE, BigScalar, RangeFull, Range, RangeFrom, RangeTo, Id, min, Functions};
use crate::{r1cs::ActivationFunction, scalar::power_of_two};

impl ConstraintSystem {
    pub fn conv2d_padded_compact(&mut self, input: TensorAddress, output: TensorAddress, weight_rev: TensorAddress, bias: Option<(TensorAddress, u32)>, bit_length: u8, act: ActivationFunction) {
        // packing weight
        let dim = &self.mem[weight_rev].dim;
        let (fout, fin, k_row, k_col) = (dim[0], dim[1], dim[2], dim[3]);
        assert!(k_row % 2 == 1 && k_col % 2 == 1);
        let pad_row = k_row/2;
        let pad_col = k_col/2;

        let packed_weight = self.mem.alloc(&[fout, fin, k_row]);
        assert!(k_col * (bit_length as u32) <= SCALAR_SIZE);
        self.packing_tensor(weight_rev, packed_weight, bit_length, k_col as u8,1, BigScalar::one(), true);

        let (row, col) = (self.mem[input].dim[1], self.mem[input].dim[2]);
        let packed_size = min((SCALAR_SIZE / (bit_length as u32)).checked_sub(k_col).unwrap(),col);
        let col_packed = (col-1)/packed_size + 1;
        let packed_layer = self.mem.alloc(&[fin, row, col_packed]);

        //packing row of inputs
        self.packing_tensor_by_dim(input,&[-1], packed_layer, bit_length, packed_size as u8,1,BigScalar::one(), true);

        // splicing output by row
        let mut mul_input = Vec::new();
        for r in 0..row {
            let mut mul_input_row = Vec::new();
            for c in 0..col_packed {
                mul_input_row.push(self.mem.save(self.mem[packed_layer].at(&[RangeFull(), Range(r.saturating_sub(pad_row)..min(row,r + pad_row + 1)), Id(c)])));
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
                let mut packed_bias_row: Vec<TensorAddress> = Vec::with_capacity((row/scale + 1) as usize);
                for r in 0..(row-1)/scale + 1 {
                    let packed_bias = self.mem.alloc(&[bias_dim]);
                    let bias_row = self.mem.save(self.mem[b].at_(&[layer_out, r]));
                    self.packing_tensor(bias_row, packed_bias, bit_length, packed_size as u8, scale,power_of_two(bit_length as u32 * (pad_col)), true);
                    packed_bias_row.push(packed_bias);
                }
                packed_bias.push(packed_bias_row);
            }
        }

        let mul_result = self.mem.alloc(&[fout, row, col_packed]);
        for layer_out in 0..fout {
            let mut cur_weight: Vec<TensorAddress> = Vec::with_capacity(k_row as usize);
            for i in (0..k_row).rev() {
                cur_weight.push(self.mem.save(self.mem[packed_weight].at(&[Id(layer_out), RangeFull(), Range(i.saturating_sub(pad_row)..min(k_row,i+pad_row+1))])));
            }
            for r in 0..row {
                let u = if r < pad_row {r} else if r >= row - pad_row {r - (row - 2*pad_row - 1)} else {pad_row} as usize;
                for c in 0..col_packed {
                    let cur_bias = if c < bias_dim {Some(self.mem[packed_bias[layer_out as usize][(r/bias_scale) as usize]].at_idx(&[c]))} else {None};
                    self.dot(mul_input[r as usize][c as usize], cur_weight[u], self.mem[mul_result].at_idx(&[layer_out, r, c]), cur_bias);
                }
            }
        }

        // sign extraction
        let n_packed = packed_size + k_col - 1;
        let extracted_length = (col_packed - 1) * n_packed + ((col-1) % packed_size) + k_col;
        let extracted = self.mem.alloc(&[fout, row, extracted_length]);

        self.packing_tensor_by_dim(extracted, &[-1], mul_result, bit_length, n_packed as u8,1,BigScalar::one(), false);

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

        let output_left = self.mem.save(self.mem[output].at(&[RangeFull(), RangeFull(), RangeTo(..pad_col)]));
        let output_mid = self.mem.save(self.mem[output].at(&[RangeFull(), RangeFull(), Range(pad_col..col-pad_col)]));
        let output_right = self.mem.save(self.mem[output].at(&[RangeFull(), RangeFull(), RangeFrom(col-pad_col..)]));

        let reduced_extract = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), RangeTo(..extracted_length - k_col  + 1)]));
        if k_col != 1 {
            let rem_extract = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), RangeFrom(extracted_length - pad_col..)]));
            extract_sign_part(self, rem_extract, bit_length);
        }

        let extract_first = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), Range(pad_col..k_col-1)]));
        let extract_last = self.mem.save(self.mem[extracted].at(&[RangeFull(), RangeFull(), Range(extracted_length - k_col + 1..extracted_length - pad_col)]));

        self.activation(extract_first, output_left, bit_length - 1, act);
        self.activation(extract_last, output_right, bit_length - 1, act);

        let [(output_full, output_full_rem), (output_part, output_part_rem), (_,_)]= split_tensor(&mut self.mem, output_mid, packed_size, [0, packed_size-(k_col-1), packed_size]);
        let [(ext_left, ext_left_rem), (ext_full, ext_full_rem), (ext_right,ext_right_rem), (_,_)]= split_tensor(&mut self.mem, reduced_extract, n_packed, [0, k_col-1, packed_size, n_packed]);

        // extract the fully correct part
        if let Some(e) = ext_full {
            self.activation(e, output_full.unwrap(), bit_length - 1, act);
        }

        if let Some(e) = ext_full_rem {
            self.activation(e, output_full_rem.unwrap(), bit_length - 1, act);
        }

        //extract left and right sign part
        // waste a bit of constraints here for the sign :(
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
                self.activation(sum_res, output_part.unwrap(), bit_length - 1, act);

                let sum_res = self.mem.alloc(&[fout, row - k_row + 1, self.mem[left_rem].dim[2]]);
                let right_rem = self.mem.save(self.mem[right].at(&[RangeFull(), Id(self.mem[right].dim[2] - 1), RangeTo(..self.mem[left_rem].dim[2])]));
                self.sum_two(right_rem, left_rem, sum_res);
                self.activation(sum_res, output_part_rem.unwrap(), bit_length - 1, act);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::slice_to_scalar;
    #[test]
    fn conv2d_padded_compact_test() {
        let mut x = ConstraintSystem::new();
        let input = x.mem.alloc(&[2,5,5]);
        let weight = x.mem.alloc(&[2,2,3,3]);
        let output = x.mem.alloc(&[2,5,5]);
        let bias = x.mem.alloc(&[2,5,5]);

        let weight_rev = x.mem.save(x.mem[weight].reverse(3));

        x.conv2d_padded_compact(input, output, weight_rev, Some((bias, 1)), 5, ActivationFunction::Sign);
        let mut mem = x.mem.new_memory();
        x.load_memory(input, &mut mem, &slice_to_scalar(&[-1,0,2,0,2,0,1,-1,0,2,1,0,2,1,1,-1,0,1,0,1,-1,0,-1,0,0,1,0,0,0,1,0,2,0,0,1,0,0,2,0,0,0,1,1,0,2,-1,0,-1,2,0]));
        x.load_memory(weight, &mut mem, &slice_to_scalar(&[0,-1,1,1,1,2,2,2,1,1,0,0,1,2,-1,0,0,-1,-1,0,-1,1,1,2,0,0,2,1,-1,-1,-1,0,0,1,0,1]));
        x.load_memory(bias, &mut mem, &slice_to_scalar(&[1,2,-1,2,0,-1,2,-1,1,2,2,-1,0,-1,2,0,2,2,-1,0,-1,0,1,0,-1,2,0,2,1,0,0,1,2,2,1,-1,-1,-1,0,1,1,-1,0,-1,0,1,0,-1,0,0]));


        x.compute(&mut mem);
        assert_eq!(mem[x.mem[output].begin() as usize..x.mem[output].end() as usize], slice_to_scalar(&[1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1]));
        x.sort_cons();
        assert!(x.verify(&mem));
    }
}