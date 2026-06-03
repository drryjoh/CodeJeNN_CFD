#include "codeJeNN_mu.H"

// ============================================================
// nn_mu_interface.H
//
// Interface between OpenFOAM and the CodeJeNN-generated
// mixture viscosity NN (codeJeNN_mu.H).
//
// The NN outputs mu_mix directly (scalar per cell), so
// compute() fills mixMu rather than a per-species PtrList.
// The Wilke mixture rule in updateTransProperties is skipped
// when useMuNN is true (guarded by !useMuNN there).
//
// Species requirement: n_species_mu = 3, order H2, O2, N2.
// ============================================================

struct NNMuInterface
{
    const fvMesh& mesh_;

    NNMuInterface(const fvMesh& mesh)
    :
        mesh_(mesh)
    {}

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

    // Fill mixMu (Pa·s) for every cell using the NN.
    // Input order to predict_mu: [T, Y_H2, Y_O2, Y_N2]
    // (species Y[0]=H2, Y[1]=O2, Y[2]=N2, matching codeJeNN_mu.H)
    void compute(
        volScalarField& mixMu,
        const PtrList<volScalarField>& Y,
        const volScalarField& T) const
    {
        std::array<scalar, 4> nn_input;

        forAll(T, celli)
        {
            nn_input[0] = T[celli];
            for (label i = 0; i < n_species_mu; ++i)
                nn_input[1 + i] = Y[i][celli];

            mixMu[celli] = Foam::max(predict_mu(nn_input), scalar(SMALL));
        }
    }
};
