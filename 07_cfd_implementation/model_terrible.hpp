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
inline void Dense_model_terrible(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
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
inline auto model_terrible(const std::array<Scalar, 4>& initial_input) {

    // Dense layer 1
    constexpr std::array<Scalar, 4> weights_1 = {2.53108978271484375e-01, 2.89949923753738403e-01, 6.29331350326538086e-01, -7.78608560562133789e-01};
    constexpr std::array<Scalar, 1> biases_1 = {1.97122955322265625e+00};

    // Dense layer 2
    constexpr std::array<Scalar, 1> weights_2 = {-1.14139521121978760e+00};
    constexpr std::array<Scalar, 1> biases_2 = {2.24960708618164062e+00};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    constexpr std::array<Scalar, 4> input_scale = {1.81374641230114986e-01, 1.80960398755279528e-01, 1.79769208247424539e-01, 8.05995710489369458e+02};

    constexpr std::array<Scalar, 4> input_shift = {3.35766814614249176e-01, 3.33784303816374750e-01, 3.30448885607413478e-01, 1.60139987412500068e+03};

    constexpr std::array<Scalar, 1> output_scale = {1.73947136499238812e-05};

    constexpr std::array<Scalar, 1> output_shift = {4.20059727762499960e-05};


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
    Dense_model_terrible<Scalar, 1>(
        layer_1_output.data(), model_input.data(),
        weights_1.data(), biases_1.data(),
        4, relu, 0.0);

    // Dense, layer 2
    static std::array<Scalar, 1> layer_2_output;
    Dense_model_terrible<Scalar, 1>(
        layer_2_output.data(), layer_1_output.data(),
        weights_2.data(), biases_2.data(),
        1, linear, 0.0);


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    static std::array<Scalar, 1> model_output;

    for (int i = 0; i < 1; i++) { model_output[i] = (layer_2_output[i] * output_scale[i]) + output_shift[i]; }

    return model_output;

}