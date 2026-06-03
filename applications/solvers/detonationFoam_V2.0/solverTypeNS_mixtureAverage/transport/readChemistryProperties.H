Info<< "Reading chemistry properties\n" << endl;

IOdictionary chemistryProperties
(
    IOobject
    (
        "chemistryProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false
    )
);

Switch differentialDiffusion(chemistryProperties.lookupOrDefault("differentialDiffusion",true));
Switch thermalDiffusion(chemistryProperties.lookupOrDefault("thermalDiffusion",false));
Switch useDiffNN(chemistryProperties.lookupOrDefault("useDiffNN",false));
Switch useMuNN(chemistryProperties.lookupOrDefault("useMuNN",false));
dimensionedScalar Prt(chemistryProperties.lookup("Prt"));
dimensionedScalar Sct(chemistryProperties.lookup("Sct"));