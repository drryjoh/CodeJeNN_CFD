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
inline void Dense_model_1x4(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
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
inline auto model_1x4(const std::array<Scalar, 4>& initial_input) {

    // Dense layer 1
    constexpr std::array<Scalar, 16> weights_1 = {-1.43603622913360596e-01, -1.76170259714126587e-01, 5.04172623157501221e-01, -4.38760340213775635e-01, -1.59164413809776306e-01, -1.67366757988929749e-01, 4.52981829643249512e-01, -4.66007888317108154e-01, -1.98590666055679321e-01, 3.68415601551532745e-02, -1.11087448894977570e-02, -6.85584008693695068e-01, 6.23282432556152344e-01, -4.39566016197204590e-01, 3.90319615602493286e-01, -9.44917559623718262e-01};
    constexpr std::array<Scalar, 4> biases_1 = {1.09426426887512207e+00, -2.19099894165992737e-02, -1.37993466854095459e+00, -2.19248557090759277e+00};

    // Dense layer 2
    constexpr std::array<Scalar, 4> weights_2 = {8.28457355499267578e-01, -9.62084054946899414e-01, 1.88441014289855957e+00, -7.62892186641693115e-01};
    constexpr std::array<Scalar, 1> biases_2 = {2.31910094618797302e-01};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    constexpr std::array<Scalar, 4> input_scale = {1.81374641230114986e-01, 1.80960398755279528e-01, 1.79769208247424539e-01, 8.05995710489369458e+02};

    constexpr std::array<Scalar, 4> input_shift = {3.35766814614249176e-01, 3.33784303816374750e-01, 3.30448885607413478e-01, 1.60139987412500068e+03};

    constexpr std::array<Scalar, 1> output_scale = {1.73947136499238812e-05};

    constexpr std::array<Scalar, 1> output_shift = {4.20059727762499960e-05};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = std::tanh(input);
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
    static std::array<Scalar, 4> layer_1_output;
    Dense_model_1x4<Scalar, 4>(
        layer_1_output.data(), model_input.data(),
        weights_1.data(), biases_1.data(),
        4, tanhCustom, 0.0);

    // Dense, layer 2
    static std::array<Scalar, 1> layer_2_output;
    Dense_model_1x4<Scalar, 1>(
        layer_2_output.data(), layer_1_output.data(),
        weights_2.data(), biases_2.data(),
        4, linear, 0.0);


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    return model_output;

}