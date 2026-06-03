/*---------------------------------------------------------------------------*\
  generateData.C

  Standalone data-generation utility for training viscosity NNs.

  For each of nSamples random states (T, Y):
    1. Draw Y_i ~ Uniform(0,1), then normalise so sum(Y_i) = 1.
    2. Draw T    ~ Uniform(200, 3000) K.
    3. Compute per-species viscosity from log-polynomial fits.
    4. Apply the Wilke mixture rule (identical to updateTransProperties.H).
    5. Write one CSV line:  j, Y_0, ..., Y_{n-1}, T, mu_mix

  Run from a case directory that contains:
      constant/generateDataProperties

  Usage:
      generateData
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "IOdictionary.H"

#include <fstream>
#include <random>
#include <cmath>

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::noParallel();
    argList::noFunctionObjects();

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
        props.lookupOrDefault<word>("outputFile", "mu_training_data.csv");

    // ----------------------------------------------------------------
    // Species data
    // ----------------------------------------------------------------
    const wordList speciesNames(props.lookup("species"));
    const int nSp = speciesNames.size();

    List<scalar> W(nSp);                                  // mol weight [g/mol]
    List<scalar> mu1(nSp), mu2(nSp), mu3(nSp), mu4(nSp); // log-polynomial coeffs

    forAll(speciesNames, i)
    {
        const dictionary& sd = props.subDict(speciesNames[i]);
        W[i]    = readScalar(sd.lookup("W"));
        mu1[i]  = readScalar(sd.lookup("Mu1"));
        mu2[i]  = readScalar(sd.lookup("Mu2"));
        mu3[i]  = readScalar(sd.lookup("Mu3"));
        mu4[i]  = readScalar(sd.lookup("Mu4"));
    }

    Info << "Generating " << nSamples << " samples"
         << " | species: " << speciesNames << nl << endl;

    // ----------------------------------------------------------------
    // Random sampling
    // ----------------------------------------------------------------
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<scalar> uni(0.0, 1.0);

    std::ofstream out(outputFile);

    // Header
    out << "j";
    forAll(speciesNames, i)
        out << "," << speciesNames[i];
    out << ",T,mu\n";

    for (label j = 0; j < nSamples; ++j)
    {
        // -- Mass fractions: draw and normalise to sum = 1 ----------
        List<scalar> Y(nSp);
        scalar Ysum = 0;
        for (int i = 0; i < nSp; ++i) { Y[i] = uni(rng); Ysum += Y[i]; }
        for (int i = 0; i < nSp; ++i) Y[i] /= Ysum;

        // -- Temperature -------------------------------------------
        const scalar T    = 200.0 + uni(rng) * 2800.0;
        const scalar logT = std::log(T);

        // -- Per-species viscosity [Pa·s] --------------------------
        //    Mu_i = 0.1 * exp(Mu1 + logT*(Mu2 + logT*(Mu3 + logT*Mu4)))
        List<scalar> Mu(nSp);
        for (int i = 0; i < nSp; ++i)
            Mu[i] = 0.1 * std::exp(
                mu1[i] + logT*(mu2[i] + logT*(mu3[i] + logT*mu4[i])));

        // -- Mole fractions ----------------------------------------
        scalar invWmix = 0;
        for (int i = 0; i < nSp; ++i) invWmix += Y[i] / W[i];
        const scalar Wmix = 1.0 / invWmix;

        List<scalar> X(nSp);
        for (int i = 0; i < nSp; ++i) X[i] = Y[i] * Wmix / W[i];

        // -- Wilke mixture rule (matches updateTransProperties.H) --
        //    phi_iz = (1/sqrt(8)) * (1 + sqrt(Mu_i/Mu_z)*(Wz/Wi)^0.25)^2
        //                         / sqrt(1 + Wi/Wz)
        //    mixMu  = sum_i [ Xi * Mu_i / sum_z(Xz * phi_iz) ]
        static const scalar inv_sqrt8 = 1.0 / std::sqrt(8.0);

        scalar mixMu = 0;
        for (int i = 0; i < nSp; ++i)
        {
            scalar denom = 0;
            for (int z = 0; z < nSp; ++z)
            {
                const scalar phi =
                    inv_sqrt8
                    * std::pow(1.0 + std::sqrt(Mu[i]/Mu[z])
                                   * std::pow(W[z]/W[i], 0.25), 2.0)
                    / std::sqrt(1.0 + W[i]/W[z]);
                denom += X[z] * phi;
            }
            mixMu += X[i] * Mu[i] / denom;
        }

        // -- Write one line ----------------------------------------
        out << j;
        for (int i = 0; i < nSp; ++i)
            out << "," << Y[i];
        out << "," << T << "," << mixMu << "\n";
    }

    out.close();

    Info << "Done.  " << nSamples << " samples written to '"
         << outputFile << "'" << endl;

    return 0;
}
