#pragma once
#include <array>
#include <cmath>
#include <string>

// ============================================================
// codeJeNN_mu.H
//
// CodeJeNN-generated mixture viscosity NN for a H2/O2/N2 system.
//
// Architecture  : Dense(4→4, tanh) → Dense(4→1, linear)
// NN input order: [O2, N2, H2, T]   (reordering handled internally)
// Output        : mixture dynamic viscosity [Pa·s]
// ============================================================

// Number of species the NN was trained on.
const int n_species_mu = 3;

// Species name table in solver order (H2=0, O2=1, N2=2).
// predict_mu input follows this order: [T, Y_H2, Y_O2, Y_N2].
static constexpr std::array<const char*, n_species_mu> species_names_mu_gen()
{
    return {"H2", "O2", "N2"};
}

static std::string species_name_mu_gen(int index)
{
    constexpr auto names = species_names_mu_gen();
    return names[index];
}

// ----------------------------------------------------------
// Dense layer forward pass (matches CodeJeNN template)
// ----------------------------------------------------------
template<typename Scalar, int output_size, typename ActFun>
inline void Dense_model_mu(
    Scalar* __restrict__ outputs,
    const Scalar* __restrict__ inputs,
    const Scalar* __restrict__ weights,
    const Scalar* __restrict__ biases,
    int input_size,
    ActFun activation_function,
    Scalar alpha) noexcept
{
    for (int i = 0; i < output_size; ++i)
    {
        Scalar sum = 0;
        for (int j = 0; j < input_size; ++j)
            sum += inputs[j] * weights[j * output_size + i];
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}

// ----------------------------------------------------------
// predict_mu
//
// Input layout (size = 4):
//   [0] T     (K)
//   [1] Y_H2  (mass fraction)
//   [2] Y_O2  (mass fraction)
//   [3] Y_N2  (mass fraction)
//
// Output: mixture dynamic viscosity [Pa·s]
//
// The NN was trained with input order [O2, N2, H2, T];
// this function reorders internally before the forward pass.
// ----------------------------------------------------------
inline double predict_mu(const std::array<double, 4>& input)
{
    const double T    = input[0];
    const double Y_H2 = input[1];
    const double Y_O2 = input[2];
    const double Y_N2 = input[3];

    // ---- Weights and biases (Dense 1: 4→4) ----
    constexpr std::array<double, 16> weights_1 =
    {
        -1.46614968776702881e-01, -1.75664350390434265e-01,
         5.09866058826446533e-01, -4.27038043737411499e-01,
        -1.61846116185188293e-01, -1.65114298462867737e-01,
         4.57602679729461670e-01, -4.56399410963058472e-01,
        -2.14557126164436340e-01,  4.87668961286544800e-02,
        -1.69384367763996124e-02, -6.96137785911560059e-01,
         6.48693621158599854e-01, -4.42994028329849243e-01,
         3.68681699037551880e-01, -1.06722140312194824e+00
    };
    constexpr std::array<double, 4> biases_1 =
    {
         1.14283370971679688e+00,  9.16160084307193756e-03,
        -1.39432680606842041e+00, -2.44630408287048340e+00
    };

    // ---- Weights and biases (Dense 2: 4→1) ----
    constexpr std::array<double, 4> weights_2 =
    {
         8.53352725505828857e-01, -9.54788148403167725e-01,
         1.92001605033874512e+00, -8.64136099815368652e-01
    };
    constexpr std::array<double, 1> biases_2 =
    {
         1.61365360021591187e-01
    };

    // ---- Input normalisation (order: O2, N2, H2, T) ----
    constexpr std::array<double, 4> input_scale =
    {
        1.80501588927165668e-01, 1.79634541210539650e-01,
        1.80674009123781287e-01, 8.08037593829024786e+02
    };
    constexpr std::array<double, 4> input_shift =
    {
        3.31779923560859058e-01, 3.32114278896807891e-01,
        3.36105797542332385e-01, 1.60029700000000003e+03
    };

    // ---- Output denormalisation ----
    constexpr double output_scale = 1.66417622824412930e-05;
    constexpr double output_shift = 4.10687488422331300e-05;

    // ---- Activation functions ----
    auto tanhAct   = +[](double& out, double in, double) noexcept
                     { out = std::tanh(in); };
    auto linearAct = +[](double& out, double in, double) noexcept
                     { out = in; };

    // ---- Build normalised input in NN order: [O2, N2, H2, T] ----
    std::array<double, 4> nn_in;
    nn_in[0] = (Y_O2 - input_shift[0]) / input_scale[0];
    nn_in[1] = (Y_N2 - input_shift[1]) / input_scale[1];
    nn_in[2] = (Y_H2 - input_shift[2]) / input_scale[2];
    nn_in[3] = (T    - input_shift[3]) / input_scale[3];

    // ---- Forward pass ----
    std::array<double, 4> layer1_out;
    Dense_model_mu<double, 4>(
        layer1_out.data(), nn_in.data(),
        weights_1.data(), biases_1.data(),
        4, tanhAct, double(0));

    std::array<double, 1> layer2_out;
    Dense_model_mu<double, 1>(
        layer2_out.data(), layer1_out.data(),
        weights_2.data(), biases_2.data(),
        4, linearAct, double(0));

    // ---- Denormalise and return mixture viscosity [Pa·s] ----
    return layer2_out[0] * output_scale + output_shift;
}
