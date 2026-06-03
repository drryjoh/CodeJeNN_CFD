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
inline void Dense_model(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
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
inline auto model(const std::array<Scalar, 4>& initial_input) {

    // Dense layer 1
    constexpr std::array<Scalar, 16> weights_1 = {-1.46614968776702881e-01, -1.75664350390434265e-01, 5.09866058826446533e-01, -4.27038043737411499e-01, -1.61846116185188293e-01, -1.65114298462867737e-01, 4.57602679729461670e-01, -4.56399410963058472e-01, -2.14557126164436340e-01, 4.87668961286544800e-02, -1.69384367763996124e-02, -6.96137785911560059e-01, 6.48693621158599854e-01, -4.42994028329849243e-01, 3.68681699037551880e-01, -1.06722140312194824e+00};
    constexpr std::array<Scalar, 4> biases_1 = {1.14283370971679688e+00, 9.16160084307193756e-03, -1.39432680606842041e+00, -2.44630408287048340e+00};

    // Dense layer 2
    constexpr std::array<Scalar, 4> weights_2 = {8.53352725505828857e-01, -9.54788148403167725e-01, 1.92001605033874512e+00, -8.64136099815368652e-01};
    constexpr std::array<Scalar, 1> biases_2 = {1.61365360021591187e-01};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    constexpr std::array<Scalar, 4> input_scale = {1.80501588927165668e-01, 1.79634541210539650e-01, 1.80674009123781287e-01, 8.08037593829024786e+02};

    constexpr std::array<Scalar, 4> input_shift = {3.31779923560859058e-01, 3.32114278896807891e-01, 3.36105797542332385e-01, 1.60029700000000003e+03};

    constexpr std::array<Scalar, 1> output_scale = {1.66417622824412930e-05};

    constexpr std::array<Scalar, 1> output_shift = {4.10687488422331300e-05};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    auto linear = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = input;
    };

    auto tanhCustom = +[](Scalar& output, Scalar input, Scalar alpha) noexcept 
    {
        output = std::tanh(input);
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
    Dense_model<Scalar, 4>(
        layer_1_output.data(), model_input.data(),
        weights_1.data(), biases_1.data(),
        4, tanhCustom, 0.0);

    // Dense, layer 2
    static std::array<Scalar, 1> layer_2_output;
    Dense_model<Scalar, 1>(
        layer_2_output.data(), layer_1_output.data(),
        weights_2.data(), biases_2.data(),
        4, linear, 0.0);


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    return model_output;

}