#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <limits>

/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


template<typename Scalar, int output_size, typename ActFun>
inline void Dense_model_lo(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
{
    for(int i = 0; i < output_size; ++i){
        Scalar sum = 0;
        
        for(int j = 0; j < input_size; ++j){
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


template <typename Scalar = double>
inline auto model_lo(const std::array<Scalar, 4>& initial_input) {

    // Dense layer 1
    constexpr std::array<Scalar, 4> weights_1 = {6.95699572563171387e-01, 7.53253877162933350e-01, 2.88867563009262085e-01, 9.44031417369842529e-01};
    constexpr std::array<Scalar, 1> biases_1 = {1.32193624973297119e+00};

    // Dense layer 2
    constexpr std::array<Scalar, 1> weights_2 = {8.65521848201751709e-01};
    constexpr std::array<Scalar, 1> biases_2 = {-1.24879634380340576e+00};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    constexpr std::array<Scalar, 4> input_scale = {1.80300346143819179e-01, 1.98745151564207773e-01, 2.44016081735489471e-01, 7.46339341083486829e+02};

    constexpr std::array<Scalar, 4> input_shift = {2.47209850753038224e-01, 3.55041737107038069e-01, 3.97748403223400437e-01, 1.33617565750000517e+03};

    constexpr std::array<Scalar, 1> output_scale = {1.74706578985354754e-05};

    constexpr std::array<Scalar, 1> output_shift = {3.62897560987500003e-05};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    auto relu = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input > 0 ? input : 0;
    };

    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input;
    };


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 

    
    // model input and flattened
    constexpr int flat_size = 4; 
    std::array<Scalar, flat_size> model_input;

    // normalize input
    for (int i = 0; i < 4; i++) { model_input[i] = (initial_input[i] - input_shift[i]) / (input_scale[i]); } 

    if (model_input.size() != 4) { throw std::invalid_argument("Invalid input size. Expected size: 4"); }

    // Dense, layer 1
    static std::array<Scalar, 1> layer_1_output;
    Dense_model_lo<Scalar, 1>(
        layer_1_output.data(), model_input.data(),
        weights_1.data(), biases_1.data(),
        4, relu, 0.0);

    // Dense, layer 2
    static std::array<Scalar, 1> layer_2_output;
    Dense_model_lo<Scalar, 1>(
        layer_2_output.data(), layer_1_output.data(),
        weights_2.data(), biases_2.data(),
        1, linear, 0.0);


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    static std::array<Scalar, 1> model_output;

    for (int i = 0; i < 1; i++) { model_output[i] = (layer_2_output[i] * output_scale[i]) + output_shift[i]; }

    return model_output;

}