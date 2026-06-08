/*---------------------------------------------------------------------------*\
  checkNNPrediction.C

  Validates three CodeJeNN viscosity NNs (1x4, 2x8, 3x12) against the
  polynomial Wilke model.

  For each of nSamples random states (T, Y):
    1. Draw Y_i ~ Uniform(0,1), normalise so sum(Y_i) = 1.
    2. Draw T    ~ Uniform(200, 3000) K.
    3. Compute polynomial per-species Mu_i, apply Wilke rule → mu_wilke.
    4. Call each NN model → mu_NN1, mu_NN2, mu_NN3.
    5. Relative error = |mu_wilke - mu_NNk| / mu_wilke.
    6. Write CSV: j, Y_H2, Y_O2, Y_N2, T, mu_wilke, mu_NN1, mu_NN2, mu_NN3,
                  e1, e2, e3

  NN input order: [Y_H2, Y_O2, Y_N2, T]  (matches training normalization)

  Reads from:  constant/generateDataProperties
  Writes to:   outputFile (default: nn_check_results.csv)

  Usage:
      checkNNPrediction
\*---------------------------------------------------------------------------*/

// NN model headers — included at file scope; template declarations are not
// permitted inside a local scope such as main().
#include "codeJeNN_mu.H"       // NN0: predict_mu([T, Y_H2, Y_O2, Y_N2])
#include "model_1x4.hpp"       // NN1: model_1x4([Y_O2, Y_N2, Y_H2, T])
#include "model_2x8.hpp"       // NN2: model_2x8([Y_O2, Y_N2, Y_H2, T])
#include "model_3x12.hpp"      // NN3: model_3x12([Y_O2, Y_N2, Y_H2, T])
#include "model_terrible.hpp"  // NNT: model_terrible([Y_O2, Y_N2, Y_H2, T])

#include "argList.H"
#include "Time.H"
#include "IOdictionary.H"

#include <fstream>
#include <random>
#include <cmath>
#include <limits>

using namespace Foam;

// Wilke mixture viscosity rule.
static scalar wilkeMix
(
    const List<scalar>& Mu,
    const List<scalar>& X,
    const List<scalar>& W
)
{
    static const scalar inv_sqrt8 = 1.0 / std::sqrt(8.0);
    const int n = Mu.size();
    scalar mix = 0;
    for (int i = 0; i < n; ++i)
    {
        scalar denom = 0;
        for (int z = 0; z < n; ++z)
        {
            denom += X[z] * inv_sqrt8
                   * std::pow(1.0 + std::sqrt(Mu[i]/Mu[z])
                                  * std::pow(W[z]/W[i], 0.25), 2.0)
                   / std::sqrt(1.0 + W[i]/W[z]);
        }
        mix += X[i] * Mu[i] / denom;
    }
    return mix;
}


int main(int argc, char *argv[])
{
    argList::noParallel();

    #include "setRootCaseLists.H"
    #include "createTime.H"

    // ----------------------------------------------------------------
    // Read configuration
    // ----------------------------------------------------------------
    IOdictionary props
    (
        IOobject
        (
            "generateDataProperties",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const label nSamples =
        readLabel(props.lookup("nSamples"));

    const word outputFile =
        props.lookupOrDefault<word>("outputFile", "nn_check_results.csv");

    // ----------------------------------------------------------------
    // Species data
    // ----------------------------------------------------------------
    const wordList speciesNames(props.lookup("species"));
    const int nSp = speciesNames.size();

    List<scalar> W(nSp);
    List<scalar> mu1(nSp), mu2(nSp), mu3(nSp), mu4(nSp);

    forAll(speciesNames, i)
    {
        const dictionary& sd = props.subDict(speciesNames[i]);
        W[i]   = readScalar(sd.lookup("W"));
        mu1[i] = readScalar(sd.lookup("Mu1"));
        mu2[i] = readScalar(sd.lookup("Mu2"));
        mu3[i] = readScalar(sd.lookup("Mu3"));
        mu4[i] = readScalar(sd.lookup("Mu4"));
    }

    Info << "Checking NN predictions for " << nSamples << " samples"
         << " | species: " << speciesNames << nl << endl;

    // ----------------------------------------------------------------
    // Random sampling
    // ----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<scalar> uni(0.0, 1.0);

    std::ofstream out(outputFile);

    out << "j,Y_H2,Y_O2,Y_N2,T,mu_wilke,mu_NN0,mu_NN1,mu_NN2,mu_NN3,mu_NNT,e0,e1,e2,e3,eT\n";

    scalar sumErr0 = 0, sumErr1 = 0, sumErr2 = 0, sumErr3 = 0, sumErrT = 0;
    scalar maxErr0 = 0, maxErr1 = 0, maxErr2 = 0, maxErr3 = 0, maxErrT = 0;

    for (label j = 0; j < nSamples; ++j)
    {
        // -- Mass fractions ----------------------------------------
        List<scalar> Y(nSp);
        scalar Ysum = 0;
        for (int i = 0; i < nSp; ++i) { Y[i] = uni(rng); Ysum += Y[i]; }
        for (int i = 0; i < nSp; ++i) Y[i] /= Ysum;

        // -- Temperature -------------------------------------------
        const scalar T    = 200.0 + uni(rng) * 2800.0;
        const scalar logT = std::log(T);

        // -- Mole fractions ----------------------------------------
        scalar invWmix = 0;
        for (int i = 0; i < nSp; ++i) invWmix += Y[i] / W[i];
        const scalar Wmix = 1.0 / invWmix;

        List<scalar> X(nSp);
        for (int i = 0; i < nSp; ++i) X[i] = Y[i] * Wmix / W[i];

        // -- Polynomial per-species Mu [Pa·s] ----------------------
        List<scalar> MuPoly(nSp);
        for (int i = 0; i < nSp; ++i)
            MuPoly[i] = 0.1 * std::exp(
                mu1[i] + logT*(mu2[i] + logT*(mu3[i] + logT*mu4[i])));

        const scalar muWilke = wilkeMix(MuPoly, X, W);

        // -- NN predictions [Pa·s] ---------------------------------
        //    NN0 input order: [T, Y_H2, Y_O2, Y_N2]  (predict_mu wrapper)
        //    NN1/2/3 input order: [Y_O2, Y_N2, Y_H2, T]  (train.py: df[['O2','N2','H2','T']])
        //    Species list order (generateDataProperties): H2=Y[0], O2=Y[1], N2=Y[2]
        std::array<scalar, 4> nn0_input = { T,    Y[0], Y[1], Y[2] };
        std::array<scalar, 4> nn_input  = { Y[1], Y[2], Y[0], T   };

        const scalar muNN0 = std::max(scalar(predict_mu(nn0_input)), scalar(1e-30));
        const scalar muNN1 = std::max(model_1x4(nn_input)[0],         scalar(1e-30));
        const scalar muNN2 = std::max(model_2x8(nn_input)[0],         scalar(1e-30));
        const scalar muNN3 = std::max(model_3x12(nn_input)[0],        scalar(1e-30));
        const scalar muNNT = std::max(model_terrible(nn_input)[0],    scalar(1e-30));

        // -- Relative errors ---------------------------------------
        const scalar e0 = std::abs(muWilke - muNN0) / muWilke;
        const scalar e1 = std::abs(muWilke - muNN1) / muWilke;
        const scalar e2 = std::abs(muWilke - muNN2) / muWilke;
        const scalar e3 = std::abs(muWilke - muNN3) / muWilke;
        const scalar eT = std::abs(muWilke - muNNT) / muWilke;

        sumErr0 += e0;  maxErr0 = std::max(maxErr0, e0);
        sumErr1 += e1;  maxErr1 = std::max(maxErr1, e1);
        sumErr2 += e2;  maxErr2 = std::max(maxErr2, e2);
        sumErr3 += e3;  maxErr3 = std::max(maxErr3, e3);
        sumErrT += eT;  maxErrT = std::max(maxErrT, eT);

        // -- Write -------------------------------------------------
        out << j
            << "," << Y[0] << "," << Y[1] << "," << Y[2]
            << "," << T
            << "," << muWilke
            << "," << muNN0 << "," << muNN1 << "," << muNN2 << "," << muNN3 << "," << muNNT
            << "," << e0 << "," << e1 << "," << e2 << "," << e3 << "," << eT
            << "\n";
    }

    out.close();

    Info << "Results written to '" << outputFile << "'" << nl
         << "  NN0 (old)  | mean: " << sumErr0/nSamples*100 << " %"
                      << "  max: " << maxErr0*100 << " %" << nl
         << "  NN1 (1x4)  | mean: " << sumErr1/nSamples*100 << " %"
                      << "  max: " << maxErr1*100 << " %" << nl
         << "  NN2 (2x8)  | mean: " << sumErr2/nSamples*100 << " %"
                      << "  max: " << maxErr2*100 << " %" << nl
         << "  NN3 (3x12) | mean: " << sumErr3/nSamples*100 << " %"
                      << "  max: " << maxErr3*100 << " %" << nl
         << "  NNT (terr) | mean: " << sumErrT/nSamples*100 << " %"
                      << "  max: " << maxErrT*100 << " %" << endl;

    return 0;
}
