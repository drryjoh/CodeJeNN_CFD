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
inline void Dense_model_2x8(Scalar* __restrict outputs, const Scalar* __restrict inputs, const Scalar * __restrict weights, const Scalar * __restrict biases, int input_size, ActFun activation_function, Scalar alpha) noexcept 
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
inline auto model_2x8(const std::array<Scalar, 4>& initial_input) {

    // Dense layer 1
    constexpr std::array<Scalar, 32> weights_1 = {7.37565802410244942e-04, 1.18858320638537407e-02, 4.91082996129989624e-01, -2.67788410186767578e-01, -2.38145694136619568e-01, -3.58065485954284668e-01, 5.32215368002653122e-03, -4.89667803049087524e-01, 1.19397556409239769e-03, -7.55202770233154297e-02, 4.24715936183929443e-01, -2.30824083089828491e-01, -1.83604821562767029e-01, -3.28536212444305420e-01, 1.06242872774600983e-01, -4.91717338562011719e-01, -2.54236727952957153e-01, 1.39215409755706787e-01, 4.41020995378494263e-01, -5.62347590923309326e-01, -3.70274722576141357e-01, -1.22642442584037781e-01, 1.64944142103195190e-01, 1.88787311315536499e-01, 3.90171408653259277e-01, -5.74611425399780273e-01, 3.69258463382720947e-01, 3.70695069432258606e-02, -4.77109879255294800e-01, -4.66121107339859009e-01, -3.24386745691299438e-01, 3.86154018342494965e-02};
    constexpr std::array<Scalar, 8> biases_1 = {-7.20570623874664307e-01, -8.81947338581085205e-01, 1.03151954710483551e-01, -4.76289480924606323e-01, -4.47180420160293579e-01, -4.28879797458648682e-01, 7.13931381702423096e-01, 7.03413963317871094e-01};

    // Dense layer 2
    constexpr std::array<Scalar, 64> weights_2 = {-3.80368679761886597e-01, -3.63871037960052490e-01, 8.12810808420181274e-02, 1.13245591521263123e-01, -4.30759280920028687e-01, 4.33662012219429016e-02, -5.28673827648162842e-01, 7.90233969688415527e-01, -2.35617026686668396e-01, 1.35112509131431580e-01, 2.14896380901336670e-01, 8.91292452812194824e-01, 1.28967627882957458e-01, 2.76292473077774048e-01, 1.87098592519760132e-01, 2.56152123212814331e-01, -3.62404398620128632e-02, 1.72506481409072876e-01, -2.15283006429672241e-01, -4.50649857521057129e-02, 1.06984652578830719e-01, -4.66635644435882568e-01, 2.68598496913909912e-01, 2.61948592960834503e-02, -2.30127438902854919e-01, -4.13812875747680664e-01, -3.85654449462890625e-01, 8.34670662879943848e-01, -5.44297695159912109e-01, -3.67920547723770142e-01, 3.40671360492706299e-01, 3.03423196077346802e-01, 2.36880034208297729e-01, -5.75116090476512909e-02, 6.82183265686035156e-01, 7.28582799434661865e-01, 6.20711207389831543e-01, -6.05743825435638428e-01, -7.81284421682357788e-02, -3.70831966400146484e-01, 4.33788567781448364e-01, 4.08854752779006958e-01, 6.29206001758575439e-02, 8.62132728099822998e-01, 4.26707029342651367e-01, 5.26331782341003418e-01, 9.37417089939117432e-01, 2.13454842567443848e-01, 7.04197943210601807e-01, 1.84457451105117798e-01, -3.65412831306457520e-01, 3.38555946946144104e-02, 3.87269973754882812e-01, 2.89936959743499756e-01, 3.85775953531265259e-01, -9.50956284999847412e-01, 4.96164709329605103e-01, 9.99164581298828125e-02, 1.60591632127761841e-01, -4.00030940771102905e-01, 2.02971801161766052e-01, 3.81802737712860107e-01, -3.01558881998062134e-01, -6.00050926208496094e-01};
    constexpr std::array<Scalar, 8> biases_2 = {6.33446812629699707e-01, -3.00369113683700562e-01, -2.31334697455167770e-02, -5.20503997802734375e-01, 2.43532598018646240e-01, -2.82524470239877701e-02, -2.56466299295425415e-01, -4.27821666002273560e-01};

    // Dense layer 3
    constexpr std::array<Scalar, 8> weights_3 = {-4.12692248821258545e-01, -3.20863127708435059e-01, -6.99312448501586914e-01, -9.61724519729614258e-01, -7.47792243957519531e-01, -1.20259083807468414e-01, -3.54182332754135132e-01, 7.63015985488891602e-01};
    constexpr std::array<Scalar, 1> biases_3 = {1.30823880434036255e-01};


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    constexpr std::array<Scalar, 4> input_scale = {1.81374641230114986e-01, 1.80960398755279528e-01, 1.79769208247424539e-01, 8.05995710489369458e+02};

    constexpr std::array<Scalar, 4> input_shift = {3.35766814614249176e-01, 3.33784303816374750e-01, 3.30448885607413478e-01, 1.60139987412500068e+03};

    constexpr std::array<Scalar, 1> output_scale = {1.73947136499238812e-05};

    constexpr std::array<Scalar, 1> output_shift = {4.20059727762499960e-05};


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
    static std::array<Scalar, 8> layer_1_output;
    Dense_model_2x8<Scalar, 8>(
        layer_1_output.data(), model_input.data(),
        weights_1.data(), biases_1.data(),
        4, tanhCustom, 0.0);

    // Dense, layer 2
    static std::array<Scalar, 8> layer_2_output;
    Dense_model_2x8<Scalar, 8>(
        layer_2_output.data(), layer_1_output.data(),
        weights_2.data(), biases_2.data(),
        8, tanhCustom, 0.0);

    // Dense, layer 3
    static std::array<Scalar, 1> layer_3_output;
    Dense_model_2x8<Scalar, 1>(
        layer_3_output.data(), layer_2_output.data(),
        weights_3.data(), biases_3.data(),
        8, linear, 0.0);


/*\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//\//*/ 


    return layer_3_output[0] * output_scale[0] + output_shift[0];

}