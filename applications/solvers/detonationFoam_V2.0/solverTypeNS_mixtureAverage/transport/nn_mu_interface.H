#include "codeJeNN_mu.H"      // NN0 — also provides n_species_mu, species_name_mu_gen
#include "model_1x4.hpp"      // NN1
#include "model_2x8.hpp"      // NN2
#include "model_3x12.hpp"     // NN3
#include "model_terrible.hpp" // NNT

// ============================================================
// nn_mu_interface.H
//
// Interface between OpenFOAM and the CodeJeNN viscosity NNs.
// Five models are selectable via muNNmodel in chemistryProperties:
//
//   NN0 — codeJeNN_mu.H     (1x4,  input [T, Y_H2, Y_O2, Y_N2])
//   NN1 — model_1x4.hpp     (1x4,  input [Y_O2, Y_N2, Y_H2, T])
//   NN2 — model_2x8.hpp     (2x8,  input [Y_O2, Y_N2, Y_H2, T])
//   NN3 — model_3x12.hpp    (3x12, input [Y_O2, Y_N2, Y_H2, T])
//   NNT — model_terrible.hpp(1x1,  input [Y_O2, Y_N2, Y_H2, T])
//
// NN1/2/3 input order matches train.py: df[['O2','N2','H2','T']].
//
// Species requirement: n_species_mu = 3, solver order H2=0, O2=1, N2=2.
// The Wilke mixture rule is skipped in updateTransProperties when useMuNN.
// ============================================================

struct NNMuInterface
{
    const fvMesh& mesh_;
    label modelIdx_;

    NNMuInterface(const fvMesh& mesh, const word& modelName)
    :
        mesh_(mesh),
        modelIdx_(nameToIdx(modelName))
    {
        Info << "NNMuInterface: active model = " << modelName << endl;
    }

    static label nameToIdx(const word& name)
    {
        if (name == "NN0") return 0;
        if (name == "NN1") return 1;
        if (name == "NN2") return 2;
        if (name == "NN3") return 3;
        if (name == "NNT") return 4;
        FatalErrorInFunction
            << "Unknown muNNmodel '" << name
            << "'. Valid options: NN0, NN1, NN2, NN3, NNT."
            << exit(FatalError);
        return 0;
    }

    static Foam::stringList species_names()
    {
        Foam::stringList names(n_species_mu);
        forAll(names, i)
            names[i] = species_name_mu_gen(i);
        return names;
    }

    void check_species_list(const PtrList<volScalarField>& Y) const
    {
        if (Y.size() != n_species_mu)
        {
            FatalErrorInFunction
                << "Species count mismatch: solver has " << Y.size()
                << " species, Mu NN expects " << n_species_mu << "."
                << exit(FatalError);
        }

        Info << "NNMuInterface: checking species order" << endl;
        Foam::stringList names = species_names();
        forAll(names, i)
        {
            if (Y[i].name() != names[i])
            {
                FatalErrorInFunction
                    << "Species mismatch at index " << i
                    << ": solver has '" << Y[i].name()
                    << "', Mu NN expects '" << names[i] << "'."
                    << exit(FatalError);
            }
        }
        Info << "NNMuInterface: species check passed" << endl;
    }

    // Raw NN output [Pa·s] for a single cell — no clamping.
    // Returns a negative value when the NN extrapolates out of range.
    inline scalar predictCellRaw
    (
        scalar T_c,
        scalar Y_H2,
        scalar Y_O2,
        scalar Y_N2
    ) const
    {
        std::array<scalar, 4> in;
        switch (modelIdx_)
        {
            case 0: in = {T_c, Y_H2, Y_O2, Y_N2}; return predict_mu(in);
            case 1: in = {Y_O2, Y_N2, Y_H2, T_c}; return model_1x4(in)[0];
            case 2: in = {Y_O2, Y_N2, Y_H2, T_c}; return model_2x8(in)[0];
            case 3: in = {Y_O2, Y_N2, Y_H2, T_c}; return model_3x12(in)[0];
            case 4: in = {Y_O2, Y_N2, Y_H2, T_c}; return model_terrible(in)[0];
            default: return scalar(0);
        }
    }

    // Clamped prediction — returns SMALL when raw output is non-positive.
    inline scalar predictCell
    (
        scalar T_c,
        scalar Y_H2,
        scalar Y_O2,
        scalar Y_N2
    ) const
    {
        return Foam::max(predictCellRaw(T_c, Y_H2, Y_O2, Y_N2), scalar(SMALL));
    }

    // Fill mixMu [Pa·s] for every cell using the active NN.
    void compute
    (
        volScalarField& mixMu,
        const PtrList<volScalarField>& Y,
        const volScalarField& T
    ) const
    {
        forAll(T, celli)
        {
            mixMu[celli] = predictCell
            (
                T[celli],
                Y[0][celli],   // H2
                Y[1][celli],   // O2
                Y[2][celli]    // N2
            );
        }
    }

    // Fill mixMu and increment errorCount for cells where the raw NN output
    // was non-positive (i.e., physically invalid — clamped to SMALL).
    void compute
    (
        volScalarField& mixMu,
        const PtrList<volScalarField>& Y,
        const volScalarField& T,
        volScalarField& errorCount
    ) const
    {
        forAll(T, celli)
        {
            const scalar rawMu = predictCellRaw
            (
                T[celli],
                Y[0][celli],   // H2
                Y[1][celli],   // O2
                Y[2][celli]    // N2
            );

            if (rawMu < scalar(SMALL))
            {
                mixMu[celli] = scalar(SMALL);
                errorCount[celli] += scalar(1);
            }
            else
            {
                mixMu[celli] = rawMu;
            }
        }
    }
};
