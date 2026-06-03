#include "readChemistryProperties.H"
#include "createDiffFields.H"
#include "createTDiffFields.H"
#include "createLambdaFields.H"
#include "createMuFields.H"

#include "readTranData.H"
#include "readBinaryDiff.H"
#include "readThermalDiff.H"
#include "readSpeciesLambda.H"
#include "readSpeciesMu.H"

NNDiffInterface nnDiffInterface(mesh);
if (useDiffNN)
    nnDiffInterface.check_species_list(Y);

// Reference to whichever diffusion list rhoYEqn should consume.
// Points to DiffNN when the NN is active, Diff otherwise.
// No data is ever copied — rhoYEqn always reads through this alias.
PtrList<volScalarField>& activeDiff = useDiffNN ? DiffNN : Diff;

NNMuInterface nnMuInterface(mesh);
if (useMuNN)
    nnMuInterface.check_species_list(Y);

// Reference to whichever viscosity list the mixture rule should consume.
// Points to MuNN when the NN is active, Mu otherwise.
PtrList<volScalarField>& activeMu = useMuNN ? MuNN : Mu;

while (runTime.run())
{
#include "updateTransProperties.H"
#include "detoCellular.H"

runTime++;
mesh.update();

#include "calculateDt.H"

Info<< "Time = " << runTime.timeName() << nl << endl;
//fluxSchemeFields->update(rho,U,e,p,c,phi,rhoPhi,rhoUPhi,rhoEPhi);
fluxSchemeFields->update(rho,rhoU,e,rPsi,c,phi,rhoPhi,rhoUPhi,rhoEPhi);

#include "rhoEqn_NS_mixtureAverage.H"
#include "rhoUEqn_NS_mixtureAverage.H"
#include "rhoYEqn_NS_mixtureAverage.H"
#include "rhoEEqn_NS_mixtureAverage.H"

turbulence->correct();
thermophysicalTransport->correct();

runTime.write();

Info<< "ExecutionTime = "<< runTime.elapsedCpuTime() << " s" << nl << endl;
Info<< "ClockTime = " << runTime.elapsedClockTime() << " s"<< nl << endl;
}

Info<< "End\n" << endl;

return 0;
