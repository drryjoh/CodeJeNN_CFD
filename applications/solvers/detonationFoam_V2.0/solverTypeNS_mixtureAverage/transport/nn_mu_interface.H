#include "codeJeNN_mu.H"

// ============================================================
// nn_mu_interface.H
//
// Interface between OpenFOAM and the CodeJeNN-generated
// per-species viscosity NN (codeJeNN_mu.H).
//
// Mirrors the NNDiffInterface pattern: constructor takes only
// the mesh; call check_species_list(Y) separately when
// useMuNN is true.  The mixture rule in updateTransProperties
// is applied to activeMu (which aliases either Mu or MuNN),
// so no changes to that logic are needed.
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

    static Foam::string species_name(const Foam::label specieI)
    {
        return species_name_mu_gen(specieI);
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

    // Populate MuNN[i] (Pa·s) for every cell and species
    // using the NN given the current T and mass fractions Y.
    void compute(
        PtrList<volScalarField>& MuNN,
        const PtrList<volScalarField>& Y,
        const volScalarField& T) const
    {
        std::array<scalar, n_species_mu + 1> nn_input;

        forAll(T, celli)
        {
            nn_input[0] = T[celli];
            for (label i = 0; i < n_species_mu; ++i)
                nn_input[1 + i] = Y[i][celli];

            const auto output = predict_mu(nn_input);

            for (label i = 0; i < n_species_mu; ++i)
                MuNN[i][celli] = Foam::max(output[i], scalar(SMALL));
        }
    }
};
