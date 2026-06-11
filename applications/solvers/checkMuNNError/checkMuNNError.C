/*---------------------------------------------------------------------------*\
  checkMuNNError.C

  Post-processing utility for detonationFoam cases.

  Reads T and species mass-fraction fields from the latest time directory,
  computes cell-by-cell mixture viscosity from the Wilke rule (using the
  log-polynomial coefficients in constant/speciesMu), then evaluates NNHI
  and NNLO and writes the relative error for each model:

      mu_err_hi   |mu_wilke - mu_NNHI| / mu_wilke
      mu_err_lo   |mu_wilke - mu_NNLO| / mu_wilke

  Transport coefficients are read from:
      constant/speciesMu    (Mu1-4 log-polynomial coefficients, same as solver)

  Species field names expected in the time directory: H2, O2, N2
  Species order assumed: Y[0]=H2, Y[1]=O2, Y[2]=N2  (matches nn_mu_interface.H)

  Molecular weights [g/mol] are hardcoded for H2/O2/N2:
      H2  2.016    O2  31.998    N2  28.014

  Usage (run from the case directory):
      checkMuNNError
\*---------------------------------------------------------------------------*/

#include "model_hi.hpp"   // NNHI: model_hi([Y_O2, Y_N2, Y_H2, T])
#include "model_lo.hpp"   // NNLO: model_lo([Y_O2, Y_N2, Y_H2, T])

#include "fvCFD.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::noParallel();

    #include "setRootCaseLists.H"
    #include "createTime.H"

    // Set runTime to the latest available time directory
    instantList times = runTime.times();
    if (times.empty())
        FatalErrorInFunction
            << "No time directories found in " << runTime.path()
            << exit(FatalError);

    runTime.setTime(times.last(), times.size() - 1);
    Info << "Processing time = " << runTime.timeName() << nl << endl;

    #include "createMesh.H"

    // ----------------------------------------------------------------
    // Read log-polynomial mu coefficients from constant/speciesMu
    // Format: mu_i [Pa·s] = 0.1 * exp(Mu1 + ln(T)*(Mu2 + ln(T)*(Mu3 + ln(T)*Mu4)))
    // ----------------------------------------------------------------
    IOdictionary speciesMuDict
    (
        IOobject
        (
            "speciesMu",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    // Fixed species list: H2=0, O2=1, N2=2
    const int nSp = 3;
    const char* spNames[nSp] = {"H2", "O2", "N2"};
    // Molecular weights [g/mol]
    const scalar W[nSp] = { 2.016, 31.998, 28.014 };

    scalar mu1[nSp], mu2[nSp], mu3[nSp], mu4[nSp];
    for (int i = 0; i < nSp; ++i)
    {
        const dictionary& sd = speciesMuDict.subDict(spNames[i]);
        mu1[i] = readScalar(sd.lookup("Mu1"));
        mu2[i] = readScalar(sd.lookup("Mu2"));
        mu3[i] = readScalar(sd.lookup("Mu3"));
        mu4[i] = readScalar(sd.lookup("Mu4"));
    }

    // ----------------------------------------------------------------
    // Read species and temperature fields from the latest time
    // ----------------------------------------------------------------
    volScalarField Y_H2
    (
        IOobject("H2", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField Y_O2
    (
        IOobject("O2", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField Y_N2
    (
        IOobject("N2", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField T
    (
        IOobject("T",  runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    // ----------------------------------------------------------------
    // Output error fields
    // ----------------------------------------------------------------
    volScalarField mu_err_hi
    (
        IOobject
        (
            "mu_err_hi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, scalar(0))
    );

    volScalarField mu_err_lo
    (
        IOobject
        (
            "mu_err_lo",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, scalar(0))
    );

    // ----------------------------------------------------------------
    // Cell-by-cell computation
    // ----------------------------------------------------------------
    static const scalar inv_sqrt8 = 1.0 / std::sqrt(8.0);

    scalar sumErrHI = 0, sumErrLO = 0;
    scalar maxErrHI = 0, maxErrLO = 0;

    forAll(T, celli)
    {
        const scalar Tc  = T[celli];
        const scalar lnT = std::log(Tc);

        const scalar Yc[nSp] = { Y_H2[celli], Y_O2[celli], Y_N2[celli] };

        // Per-species viscosity [Pa·s]
        scalar Mu[nSp];
        for (int i = 0; i < nSp; ++i)
            Mu[i] = 0.1 * std::exp(mu1[i] + lnT*(mu2[i] + lnT*(mu3[i] + lnT*mu4[i])));

        // Mole fractions
        scalar invWmix = 0;
        for (int i = 0; i < nSp; ++i) invWmix += Yc[i] / W[i];
        const scalar Wmix = 1.0 / invWmix;

        scalar X[nSp];
        for (int i = 0; i < nSp; ++i) X[i] = Yc[i] * Wmix / W[i];

        // Wilke mixture rule
        scalar muWilke = 0;
        for (int i = 0; i < nSp; ++i)
        {
            scalar denom = 0;
            for (int z = 0; z < nSp; ++z)
            {
                denom += X[z] * inv_sqrt8
                       * std::pow(1.0 + std::sqrt(Mu[i]/Mu[z])
                                      * std::pow(W[z]/W[i], 0.25), 2.0)
                       / std::sqrt(1.0 + W[i]/W[z]);
            }
            muWilke += X[i] * Mu[i] / denom;
        }

        // NN predictions — input order: [O2, N2, H2, T]
        std::array<scalar, 4> nn_in = { Y_O2[celli], Y_N2[celli], Y_H2[celli], Tc };

        const scalar muHI = std::max(model_hi(nn_in)[0], scalar(1e-30));
        const scalar muLO = std::max(model_lo(nn_in)[0], scalar(1e-30));

        const scalar eHI = std::abs(muWilke - muHI) / muWilke;
        const scalar eLO = std::abs(muWilke - muLO) / muWilke;

        mu_err_hi[celli] = eHI;
        mu_err_lo[celli] = eLO;

        sumErrHI += eHI;  maxErrHI = std::max(maxErrHI, eHI);
        sumErrLO += eLO;  maxErrLO = std::max(maxErrLO, eLO);
    }

    mu_err_hi.write();
    mu_err_lo.write();

    const label nCells = mesh.nCells();
    Info << "Written to " << runTime.timeName() << nl
         << "  NNHI | mean: " << sumErrHI/nCells*100 << " %"
                   << "  max: " << maxErrHI*100 << " %" << nl
         << "  NNLO | mean: " << sumErrLO/nCells*100 << " %"
                   << "  max: " << maxErrLO*100 << " %" << endl;

    return 0;
}
