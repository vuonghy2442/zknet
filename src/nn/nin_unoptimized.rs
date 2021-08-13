use std::collections::HashMap;
use super::{NeuralNetwork, ConstraintSystem, sum_pool, padded_convolution_layer, convolution_layer, convolution_layer_act_compact, sign_activation, max_pool, AccuracyParams, TensorAddress};
use crate::{nn::relu_activation, r1cs::{ActivationFunction, elliptic_curve::{get_a, get_d}}};


macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

impl NeuralNetwork {
    pub fn new_nin_unoptimized(accuracy: bool) -> NeuralNetwork {
        let mut c = ConstraintSystem::new();
        let input = c.mem.alloc(&[3,32,32]);
        let (conv1_out, conv1_weight, conv1_bias) = padded_convolution_layer(&mut c, input, [5,5], 192, 0);
        c.log_cons("conv1");
        let conv1_out_sign = sign_activation(&mut c, conv1_out, 16);
        c.log_cons("conv1_sign");
        let (conv1_a_out, conv1_a_weight, conv1_a_bias) = convolution_layer(&mut c, conv1_out_sign, [1,1], 160, 0);
        c.log_cons("conv1_a");
        let conv1_a_out_sign = sign_activation(&mut c, conv1_a_out, 9);
        c.log_cons("conv1_a_sign");
        let (conv1_b_out, conv1_b_weight, conv1_b_bias) = convolution_layer(&mut c, conv1_a_out_sign, [1,1], 96, 0);
        c.log_cons("conv1_b");
        let conv1_b_out_sign = sign_activation(&mut c, conv1_b_out, 9);
        c.log_cons("conv1_b_sign");

        let pool1 = max_pool(&mut c, conv1_b_out_sign); // 16 x 16
        c.log_cons("pool1");

        let (conv2_out, conv2_weight, conv2_bias) = padded_convolution_layer(&mut c, pool1, [5,5], 192, 0);
        c.log_cons("conv2");
        let conv2_out_sign = sign_activation(&mut c, conv2_out, 11);
        c.log_cons("conv2_sign");
        let (conv2_a_out, conv2_a_weight, conv2_a_bias) = convolution_layer(&mut c, conv2_out_sign, [1,1], 192, 0);
        c.log_cons("conv2_a");
        let conv2_a_out_sign = sign_activation(&mut c, conv2_a_out, 9);
        c.log_cons("conv2_a_sign");
        let (conv2_b_out, conv2_b_weight, conv2_b_bias) = convolution_layer(&mut c, conv2_a_out_sign, [1,1], 192, 0);
        c.log_cons("conv2_b");
        let conv2_b_out_sign = sign_activation(&mut c, conv2_b_out, 9);
        c.log_cons("conv2_b_sign");

        let pool2 = max_pool(&mut c, conv2_b_out_sign); // 8 x 8 //average pool instead but whatever lol
        c.log_cons("pool2");

        let (conv3_out, conv3_weight, conv3_bias) = padded_convolution_layer(&mut c, pool2, [3, 3], 192, 0);
        c.log_cons("conv3");
        let conv3_out_sign= sign_activation(&mut c, conv3_out, 12);
        c.log_cons("conv3_sign");
        let (conv3_a_out, conv3_a_weight, conv3_a_bias) = convolution_layer(&mut c, conv3_out_sign, [1,1], 192, 0);
        c.log_cons("conv3_a");
        let conv3_a_out_sign= relu_activation(&mut c, conv3_a_out, 9);
        c.log_cons("conv3_a_relu");
        let (conv3_b_out, conv3_b_weight, conv3_b_bias) = convolution_layer(&mut c, conv3_a_out_sign, [1,1], 10, 0);
        c.log_cons("conv3_b");
        let conv3_b_out_sign= relu_activation(&mut c, conv3_b_out, 20);

        c.log_cons("conv3_b_relu");

        let pool3 = sum_pool(&mut c, conv3_b_out_sign, [8,8]);
        c.log_cons("sum_pool");

        let nn_output = c.mem.save(c.mem[pool3].flatten());

        let conv1_weight_packed = c.packing_and_check_range(conv1_weight, 6);
        let conv1_bias_packed = c.packing_and_check_range(conv1_bias, 14);
        let conv1_a_weight_packed = c.packing_and_check_range(conv1_a_weight, 1);
        let conv1_a_bias_packed = c.packing_and_check_range(conv1_a_bias, 6);
        let conv1_b_weight_packed = c.packing_and_check_range(conv1_b_weight, 1);
        let conv1_b_bias_packed = c.packing_and_check_range(conv1_b_bias, 7);
        c.log_cons("conv1x_packing");

        let conv2_weight_packed = c.packing_and_check_range(conv2_weight, 1);
        let conv2_bias_packed = c.packing_and_check_range(conv2_bias, 9);
        let conv2_a_weight_packed = c.packing_and_check_range(conv2_a_weight, 1);
        let conv2_a_bias_packed = c.packing_and_check_range(conv2_a_bias, 8);
        let conv2_b_weight_packed = c.packing_and_check_range(conv2_b_weight, 1);
        let conv2_b_bias_packed = c.packing_and_check_range(conv2_b_bias, 7);
        c.log_cons("conv2x_packing");

        let conv3_weight_packed = c.packing_and_check_range(conv3_weight, 1);
        let conv3_bias_packed = c.packing_and_check_range(conv3_bias, 10);
        let conv3_a_weight_packed = c.packing_and_check_range(conv3_a_weight, 1);
        let conv3_a_bias_packed = c.packing_and_check_range(conv3_a_bias, 3);
        let conv3_b_weight_packed = c.packing_and_check_range(conv3_b_weight, 13);
        let conv3_b_bias_packed = c.packing_and_check_range(conv3_b_bias, 14);
        c.log_cons("conv3x_packing");

        let commit_open = c.mem.alloc(&[1]);

        let hash_output = c.poseidon_hash(&[commit_open,
            conv1_weight_packed, conv1_bias_packed, conv1_a_weight_packed, conv1_a_bias_packed, conv1_b_weight_packed, conv1_b_bias_packed,
            conv2_weight_packed, conv2_bias_packed, conv2_a_weight_packed, conv2_a_bias_packed, conv2_b_weight_packed, conv2_b_bias_packed,
            conv3_weight_packed, conv3_bias_packed, conv3_a_weight_packed, conv3_a_bias_packed, conv3_b_weight_packed, conv3_b_bias_packed]);
        c.log_cons("hashing");

        let (output, acc) = if accuracy {
            let p = c.mem.alloc(&[2]);
            let q = c.mem.alloc(&[2]);
            let result_open = c.mem.alloc(&[1]);
            let ground_truth = c.mem.alloc(&[1]);
            let value = c.mem.alloc(&[1]);
            let result = c.mem.alloc(&[1]);
            let mul_p = c.mem.alloc(&[2]);
            let commited_result = c.mem.alloc(&[2]);


            c.multiplexer(nn_output, c.mem[ground_truth].begin(), c.mem[value].begin());
            c.is_max(nn_output, c.mem[value].begin(), c.mem[result].begin(), 20);
            c.elliptic_mul(&c.mem[p].to_vec(), c.mem[result_open].begin(), &c.mem[mul_p].to_vec(), get_a(), get_d());
            c.elliptic_add_cond(&c.mem[mul_p].to_vec(), &c.mem[q].to_vec(), &c.mem[commited_result].to_vec(), c.mem[result].begin(), get_a(), get_d());
            c.log_cons("accuracy");


            c.reorder_for_spartan(&[input, ground_truth, commited_result, hash_output, p, q]);
            (result, Some(AccuracyParams{
                ground_truth,
                result_open,
                p,
                q,
                output: commited_result
            }))
        } else {
            c.reorder_for_spartan(&[input, nn_output, hash_output]);
            (nn_output, None)
        };
        c.sort_cons();

        info!("Constraints {}", c.cons_size());

        let weight_map: HashMap<String, TensorAddress> = hashmap!{
            String::from("conv1_weight") => conv1_weight,
            String::from("conv1_bias") => conv1_bias,
            String::from("conv1_a_weight") => conv1_a_weight,
            String::from("conv1_a_bias") => conv1_a_bias,
            String::from("conv1_b_weight") => conv1_b_weight,
            String::from("conv1_b_bias") => conv1_b_bias,

            String::from("conv2_weight") => conv2_weight,
            String::from("conv2_bias") => conv2_bias,
            String::from("conv2_a_weight") => conv2_a_weight,
            String::from("conv2_a_bias") => conv2_a_bias,
            String::from("conv2_b_weight") => conv2_b_weight,
            String::from("conv2_b_bias") => conv2_b_bias,

            String::from("conv3_weight") => conv3_weight,
            String::from("conv3_bias") => conv3_bias,
            String::from("conv3_a_weight") => conv3_a_weight,
            String::from("conv3_a_bias") => conv3_a_bias,
            String::from("conv3_b_weight") => conv3_b_weight,
            String::from("conv3_b_bias") => conv3_b_bias
        };

        NeuralNetwork {
            cons: c,
            weight_map,
            input,
            output,
            commit_hash: hash_output,
            commit_open,
            acc,
            scaling: 65536.0
        }
    }
}