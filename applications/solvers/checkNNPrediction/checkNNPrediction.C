/*---------------------------------------------------------------------------*\
  checkNNPrediction.C

  Validates the CodeJeNN viscosity NN against the polynomial Wilke model.

  For each of nSamples random states (T, Y):
    1. Draw Y_i ~ Uniform(0,1), normalise so sum(Y_i) = 1.
    2. Draw T    ~ Uniform(200, 3000) K.
    3. Compute polynomial per-species Mu_i, apply Wilke rule → mu_poly.
    4. Call predict_mu() → NN per-species MuNN_i, apply same Wilke rule → mu_nn.
    5. Relative error = |mu_poly - mu_nn| / mu_poly.
    6. Write one CSV line: i, Y_0,...,Y_{n-1}, T, mu, muNN, rel_error

  Relative error is chosen because viscosity spans ~3x over the temperature
  range; an absolute metric would be biased toward high-T samples.

  A summary (mean and max relative error) is printed to stdout on completion.

  Reads from:  constant/generateDataProperties  (same format as generateData)
  Writes to:   outputFile (default: nn_check_results.csv)

  Usage:
      checkNNPrediction
\*---------------------------------------------------------------------------*/

// NN must be included at file scope — template declarations are not
// permitted inside a local class (i.e. inside main()).
#include "codeJeNN_mu.H"

#include "argList.H"
#include "Time.H"
#include "IOdictionary.H"

#include <fstream>
#include <random>
#include <cmath>
#include <limits>

using namespace Foam;

// Helper: apply the Wilke mixture rule to per-species Mu and mole fractions X.
// Matches the formula in updateTransProperties.H exactly.
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
    // Read configuration (same dict format as generateData)
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

    // Validate species count matches the NN at compile time
    if (nSp != n_species_mu)
    {
        FatalErrorInFunction
            << "Species count mismatch: properties file has " << nSp
            << " species, but codeJeNN_mu.H defines n_species_mu = "
            << n_species_mu << ".\n"
            << "Update codeJeNN_mu.H or the species list to match."
            << exit(FatalError);
    }

    List<scalar> W(nSp);
    List<scalar> mu1(nSp), mu2(nSp), mu3(nSp), mu4(nSp);

    forAll(speciesNames, i)
    {
        const dictionary& sd = props.subDict(speciesNames[i]);
        W[i]    = readScalar(sd.lookup("W"));
        mu1[i]  = readScalar(sd.lookup("Mu1"));
        mu2[i]  = readScalar(sd.lookup("Mu2"));
        mu3[i]  = readScalar(sd.lookup("Mu3"));
        mu4[i]  = readScalar(sd.lookup("Mu4"));
    }

    Info << "Checking NN predictions for " << nSamples << " samples"
         << " | species: " << speciesNames << nl << endl;

    // ----------------------------------------------------------------
    // Random sampling
    // ----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<scalar> uni(0.0, 1.0);

    std::ofstream out(outputFile);

    // Header
    out << "i";
    forAll(speciesNames, i)
        out << "," << speciesNames[i];
    out << ",T,mu,muNN,rel_error\n";

    // Running statistics
    scalar sumRelErr = 0;
    scalar maxRelErr = 0;

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

        const scalar muPoly = wilkeMix(MuPoly, X, W);

        // -- NN per-species Mu [Pa·s] ------------------------------
        //    predict_mu input: [T, Y_0, Y_1, ..., Y_{n-1}]
        std::array<scalar, n_species_mu + 1> nn_input;
        nn_input[0] = T;
        for (int i = 0; i < n_species_mu; ++i)
            nn_input[1 + i] = Y[i];

        const auto nn_out = predict_mu(nn_input);

        List<scalar> MuNN(nSp);
        for (int i = 0; i < nSp; ++i)
            MuNN[i] = std::max(nn_out[i], scalar(1e-30));

        const scalar muNN = wilkeMix(MuNN, X, W);

        // -- Relative error ----------------------------------------
        const scalar relErr = std::abs(muPoly - muNN) / muPoly;

        sumRelErr += relErr;
        maxRelErr  = std::max(maxRelErr, relErr);

        // -- Write -------------------------------------------------
        out << j;
        for (int i = 0; i < nSp; ++i)
            out << "," << Y[i];
        out << "," << T
            << "," << muPoly
            << "," << muNN
            << "," << relErr
            << "\n";
    }

    out.close();

    const scalar meanRelErr = sumRelErr / nSamples;

    Info << "Results written to '" << outputFile << "'" << nl
         << "  Mean relative error : " << meanRelErr * 100 << " %" << nl
         << "  Max  relative error : " << maxRelErr  * 100 << " %" << endl;

    return 0;
}
