def run_openmm_uniax_NPT1(parm_name, ambermin_rst, system_name, fix_bb=False, fix_oth=False):

    # COMPUTING-PLATFORM--------------------------------------------------------------

    # Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}

    ##If multiple GPUs
    # platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}

    # CPUs
    # platform = Platform.getPlatformByName('CPU')
    # platformProperties = {'CpuThreads': 'default'}

    # SIMULATION-INPUTS---------------------------------------------------------------

    ##variables related to barostat, thermostat, and force calculations

    nonbondedMethod = PME  # Uses Particle-Mesh-Ewald method for electrostatic forces
    nonbondedCutoff = 1.0 * nanometers  # (for CHARMM ff, 1.0 nm for Amberff)
    ewaldErrorTolerance = 0.0005  # Cutoff for PME forces
    constraints = HBonds  # Puts constraints on
    # rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001  # Tolerance for constraints
    temperature = 300 * kelvin  # Temperature in Kelvin
    friction = 2.8284 / picosecond  # Collision frequency for Langevin thermostat
    pz = 3000.0 * atmospheres

    parm = LoadParm(parm_name, ambermin_rst)
    print('Creating system')

    topology = parm.topology  # Could be psf.topology if done with CHARMM
    positions = parm.positions

    print('Building system...')
    system = parm.createSystem(nonbondedMethod=nonbondedMethod,
                               nonbondedCutoff=nonbondedCutoff,
                               constraints=constraints,
                               ewaldErrorTolerance=ewaldErrorTolerance)

    system.addForce(MonteCarloAnisotropicBarostat((0, 0, pz), temperature, False, False, True, 25))
    ##Generate initil xml file
    f = open(system_name + 'uniax_pres.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()

    print('Setting simulation length...')

    #300 picosecs of uniaxial compression:
    dt = 0.002 * picoseconds

    n_steps = 150000
   

    # INTEGRATOR----------------------------------------------------------------------

    print('Creating integrator...')
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    indices_int = []
    for j in range(300,301):
        get_atoms = str(parm.residues[j].atoms)
        indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
        for item in indices_str:
             for subitem in item.split('['):
                   if(subitem.isdigit()):
                         print('the subitem:')
                         print(subitem)
                         indices_int.append(int(subitem))


    #DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1.0*kilocalories_per_mole/angstroms**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    #add restraint to the particles
    for k in indices_int:
        restraint.addParticle(k, parm.positions[k].value_in_unit(u.nanometers))
    system.addForce(restraint)

    g = open(system_name + 'uniaxial_NPT1.xml', 'w+')
    g.write(XmlSerializer.serialize(integrator))
    g.close()



    # what gets saved from simulation runs

    dcdReporter = DCDReporter(system_name + 'uniax_NPT1.dcd',
                              500)  ##how often in number of steps should the output file be written
    dataReporter = StateDataReporter(system_name + 'uniax_NPT1.log',  ## other information about the state
                                     500,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator='	')
    checkpointReporter = CheckpointReporter(system_name + 'uniax_NPT1.chk', 5000)
    restartReporter = RestartReporter(system_name + 'uniax_NPT1.rst7', reportInterval=5000, netcdf=True)

    print('Creating Simulation object...')

    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S						   o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    simulation.saveState(system_name + 'uniax_NPT1_begin.xml')

    # pressure_______coupling________________________________________________________

    # system.addForce(MonteCarloAnisotropicBarostat((0,0,pz),temperature,True,True,True,25)) ### the scalez is set to False so the repeating unit ia allowed to change in all directions exprect the z
    simulation.step(n_steps)
    print('Done uniaxial NPT1 at', pz, 'atm')


    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True,
                                              getParameterDerivatives=True, getParameters=True)
    outfile = open(system_name + 'uniax_pres_done.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()



def run_openmm_800K_NVT2 (parm_name, system_name, i, fix_bb=False, fix_oth=False):
    # COMPUTING-PLATFORM--------------------------------------------------------------

    # how this simulation will be run

    # Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}

    ##If multiple GPUs
    # platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}

    # CPUs
    #platform = Platform.getPlatformByName('CPU')
    #platformProperties = {'CpuThreads': 'default'}

    # SIMULATION-INPUTS---------------------------------------------------------------

    ##variables related to barostat, thermostat, and force calculations

    nonbondedMethod = PME  # Uses Particle-Mesh-Ewald method for electrostatic forces
    nonbondedCutoff = 1.0 * nanometers  # (for CHARMM ff, 1.0 nm for Amberff)
    ewaldErrorTolerance = 0.0005  # Cutoff for PME forces
    constraints = HBonds  # Puts constraints on
    # rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001  # Tolerance for constraints
    temperature = 800 * kelvin  # Temperature in Kelvin
    friction = 2.8284 / picosecond  # Collision frequency for Langevin thermostat

    # DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    if fix_bb:
        force_bb = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force_bb.addGlobalParameter("k", 5.0 * kilocalories_per_mole / angstroms ** 2)
        force_bb.addPerParticleParameter("x0")
        force_bb.addPerParticleParameter("y0")
        force_bb.addPerParticleParameter("z0")

    if fix_oth:
        force_oth = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force_oth.addGlobalParameter("k", 5.0 * kilocalories_per_mole / angstroms ** 2)
        force_oth.addPerParticleParameter("x0")
        force_oth.addPerParticleParameter("y0")
        force_oth.addPerParticleParameter("z0")

######----------For the heating cooling compression loops for the first iteration load the output from NPT1 step or else load the output of the NPT4 step

    if i == 1 :
    	parm = LoadParm(parm_name, system_name + 'uniax_NPT1.rst7')
    else :
    	parm = LoadParm(parm_name, system_name + 'uniax_NPT4.rst7')
    
    print('Creating system')

    topology = parm.topology  # Could be psf.topology if done with CHARMM
    positions = parm.positions

    print('Building system...')
    system = parm.createSystem(nonbondedMethod=nonbondedMethod,
                               nonbondedCutoff=nonbondedCutoff,
                               constraints=constraints,
                               ewaldErrorTolerance=ewaldErrorTolerance)

    ##Apply Harmonic Restraints
    if fix_bb:
        for i, atom_crd in enumerate(parm.positions):
            if parm.atoms[i].name in ('C', 'CA', 'N'):
                force_bb.addParticle(i, atom_crd.value_in_unit(u.nanometers))
        system.addForce(force_bb)

    ####This part is for fixing entire residues
    if fix_oth:
        for j in range(300,302):
              get_atoms = str(parm.residues[j].atoms)
              print(get_atoms)
              indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
              indices_int = []
              for item in indices_str:
                    for subitem in item.split():
                          if(subitem.isdigit()):
                                indices_int.append(int(subitem))
              for k in indices_int:
                    force_oth.addParticle(j, parm.positions[k].value_in_unit(u.nanometers))
        system.addForce(force_oth)


    ##Generate initil xml file
    f = open(system_name + 'step2_800K_NVT_initialize.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()

    print('Setting simulation length...')

    # Determine simulation length
    dt = 0.002 * picoseconds

    n_steps = 50000

    # INTEGRATOR----------------------------------------------------------------------

    print('Creating integrator...')
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    indices_int = []
    for j in range(300,301):
        get_atoms = str(parm.residues[j].atoms)
        indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
        for item in indices_str:
             for subitem in item.split('['):
                   if(subitem.isdigit()):
                         print('the subitem:')
                         print(subitem)
                         indices_int.append(int(subitem))
                         

    #DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1.0*kilocalories_per_mole/angstroms**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    g = open(system_name + 'step2_800K_NVT2.xml', 'w+')
    g.write(XmlSerializer.serialize(integrator))
    g.close()

    # what gets saved from simulation runs

    dcdReporter = DCDReporter(system_name + 'step2_800K_NVT2.dcd',
                              500)  ##how often in number of steps should the output file be written
    dataReporter = StateDataReporter(system_name + 'step2_800K_NVT2.log',  ## other information about the state
                                     500,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator='	')
    checkpointReporter = CheckpointReporter(system_name + 'step2_800K_NVT2.chk', 5000)
    restartReporter = RestartReporter(system_name + 'step2_800K_NVT2.rst7', reportInterval=5000, netcdf=True)

    print('Creating Simulation object...')

    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions

    # ---------------------------------------------------------------------------------
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S						   o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    simulation.saveState(system_name + 'Step2_800K_NVT2.xml')

    # Performing NVT -----------------------------------------------------------------

    print('Performing NVT2')

    integrator.setTemperature(temperature)
    simulation.step(n_steps)

    print('Done with NVT2')

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True,
                                              getParameterDerivatives=True, getParameters=True)
    outfile = open(system_name + 'step2_800K_NVT2_DONE.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()


def run_openmm_300K_NVT3 (parm_name, system_name, fix_bb=False, fix_oth=False):
    # COMPUTING-PLATFORM--------------------------------------------------------------

    # how this simulation will be run

    # Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}

    ##If multiple GPUs
    # platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}

    # CPUs
    #platform = Platform.getPlatformByName('CPU')
    #platformProperties = {'CpuThreads': 'default'}

    # SIMULATION-INPUTS---------------------------------------------------------------

    ##variables related to barostat, thermostat, and force calculations

    nonbondedMethod = PME  # Uses Particle-Mesh-Ewald method for electrostatic forces
    nonbondedCutoff = 1.0 * nanometers  # (for CHARMM ff, 1.0 nm for Amberff)
    ewaldErrorTolerance = 0.0005  # Cutoff for PME forces
    constraints = HBonds  # Puts constraints on
    # rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001  # Tolerance for constraints
    temperature = 300 * kelvin  # Temperature in Kelvin
    friction = 2.8284 / picosecond  # Collision frequency for Langevin thermostat

    # DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    if fix_bb:
        force_bb = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force_bb.addGlobalParameter("k", 5.0 * kilocalories_per_mole / angstroms ** 2)
        force_bb.addPerParticleParameter("x0")
        force_bb.addPerParticleParameter("y0")
        force_bb.addPerParticleParameter("z0")

    if fix_oth:
        force_oth = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force_oth.addGlobalParameter("k", 5.0 * kilocalories_per_mole / angstroms ** 2)
        force_oth.addPerParticleParameter("x0")
        force_oth.addPerParticleParameter("y0")
        force_oth.addPerParticleParameter("z0")

    parm = LoadParm(parm_name, system_name + 'step2_800K_NVT2.rst7')
    print('Creating system')

    topology = parm.topology  # Could be psf.topology if done with CHARMM
    positions = parm.positions

    print('Building system...')
    system = parm.createSystem(nonbondedMethod=nonbondedMethod,
                               nonbondedCutoff=nonbondedCutoff,
                               constraints=constraints,
                               ewaldErrorTolerance=ewaldErrorTolerance)

    ##Apply Harmonic Restraints
    if fix_bb:
        for i, atom_crd in enumerate(parm.positions):
            if parm.atoms[i].name in ('C', 'CA', 'N'):
                force_bb.addParticle(i, atom_crd.value_in_unit(u.nanometers))
        system.addForce(force_bb)

    ####This part is for fixing entire residues
    if fix_oth:
        for j in range(190,192):
              get_atoms = str(parm.residues[j].atoms)
              print(get_atoms)
              indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
              indices_int = []
              for item in indices_str:
                    for subitem in item.split():
                          if(subitem.isdigit()):
                                indices_int.append(int(subitem))
              for k in indices_int:
                    force_oth.addParticle(j, parm.positions[k].value_in_unit(u.nanometers))
        system.addForce(force_oth)


    ##Generate initil xml file
    f = open(system_name + 'step3_300K_NVT_initialize.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()

    print('Setting simulation length...')

    # Determine simulation length
    dt = 0.002 * picoseconds

    n_steps = 50000

    # INTEGRATOR----------------------------------------------------------------------

    print('Creating integrator...')
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)


    indices_int = []
    for j in range(300,301):
        get_atoms = str(parm.residues[j].atoms)
        indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
        for item in indices_str:
             for subitem in item.split('['):
                   if(subitem.isdigit()):
                         print('the subitem:')
                         print(subitem)
                         indices_int.append(int(subitem))


    #DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1.0*kilocalories_per_mole/angstroms**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    g = open(system_name + 'step3_300K_NVT3.xml', 'w+')
    g.write(XmlSerializer.serialize(integrator))
    g.close()

    # what gets saved from simulation runs

    dcdReporter = DCDReporter(system_name + 'step3_300K_NVT3.dcd',
                              500)  ##how often in number of steps should the output file be written
    dataReporter = StateDataReporter(system_name + 'step3_300K_NVT3.log',  ## other information about the state
                                     500,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator='	')
    checkpointReporter = CheckpointReporter(system_name + 'step3_300K_NVT3.chk', 5000)
    restartReporter = RestartReporter(system_name + 'step3_300K_NVT3.rst7', reportInterval=5000, netcdf=True)

    print('Creating Simulation object...')

    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions


    # ---------------------------------------------------------------------------------
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S						   o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    simulation.saveState(system_name + 'Step3_300K_NVT3.xml')

    # Performing NVT -----------------------------------------------------------------

    print('Performing NVT3')

    integrator.setTemperature(temperature)
    simulation.step(n_steps)

    print('Done with NVT3')
    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True,
                                              getParameterDerivatives=True, getParameters=True)
    outfile = open(system_name + 'step3_300K_NVT3_DONE.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()

def run_openmm_uniax_NPT4(parm_name, system_name, fix_bb=False, fix_oth=False):

    # COMPUTING-PLATFORM--------------------------------------------------------------

    # Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}

    ##If multiple GPUs
    # platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}

    # CPUs
    # platform = Platform.getPlatformByName('CPU')
    # platformProperties = {'CpuThreads': 'default'}

    # SIMULATION-INPUTS---------------------------------------------------------------

    ##variables related to barostat, thermostat, and force calculations

    nonbondedMethod = PME  # Uses Particle-Mesh-Ewald method for electrostatic forces
    nonbondedCutoff = 1.0 * nanometers  # (for CHARMM ff, 1.0 nm for Amberff)
    ewaldErrorTolerance = 0.0005  # Cutoff for PME forces
    constraints = HBonds  # Puts constraints on
    # rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001  # Tolerance for constraints
    temperature = 300 * kelvin  # Temperature in Kelvin
    friction = 2.8284 / picosecond  # Collision frequency for Langevin thermostat
    pz = 1000.0 * atmospheres

    parm = LoadParm(parm_name, system_name + 'step3_300K_NVT3.rst7')
    print('Creating system')

    topology = parm.topology  # Could be psf.topology if done with CHARMM
    positions = parm.positions

    print('Building system...')
    system = parm.createSystem(nonbondedMethod=nonbondedMethod,
                               nonbondedCutoff=nonbondedCutoff,
                               constraints=constraints,
                               ewaldErrorTolerance=ewaldErrorTolerance)

    system.addForce(MonteCarloAnisotropicBarostat((0, 0, pz), temperature, False, False, True, 25))
    ##Generate initil xml file
    f = open(system_name + 'uniax_pres4.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()

    print('Setting simulation length...')

    #300 picosecs of uniaxial compression:
    dt = 0.002 * picoseconds

    n_steps = 150000

    # INTEGRATOR----------------------------------------------------------------------

    print('Creating integrator...')
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)


    indices_int = []
    for j in range(300,301):
        get_atoms = str(parm.residues[j].atoms)
        indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
        for item in indices_str:
             for subitem in item.split('['):
                   if(subitem.isdigit()):
                         print('the subitem:')
                         print(subitem)
                         indices_int.append(int(subitem))


    #DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1.0*kilocalories_per_mole/angstroms**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    #add restraint to the particles
    for k in indices_int:
        restraint.addParticle(k, parm.positions[k].value_in_unit(u.nanometers))
    system.addForce(restraint)


    g = open(system_name + 'uniaxial_NPT4.xml', 'w+')
    g.write(XmlSerializer.serialize(integrator))
    g.close()

    # what gets saved from simulation runs

    dcdReporter = DCDReporter(system_name + 'uniax_NPT4.dcd',
                              500)  ##how often in number of steps should the output file be written
    dataReporter = StateDataReporter(system_name + 'uniax_NPT4.log',  ## other information about the state
                                     500,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator='	')
    checkpointReporter = CheckpointReporter(system_name + 'uniax_NPT4.chk', 5000)
    restartReporter = RestartReporter(system_name + 'uniax_NPT4.rst7', reportInterval=5000, netcdf=True)

    print('Creating Simulation object...')

    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S						   o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    simulation.saveState(system_name + 'uniax_NPT4_begin.xml')
    simulation.step(n_steps)
    print('Done uniaxial NPT4 at', pz, 'atm')


    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True,
                                              getParameterDerivatives=True, getParameters=True)
    outfile = open(system_name + 'uniax_pres4_done.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()


def run_openmm_final_NPT(parm_name, system_name, fix_bb=False, fix_oth=False):

    # COMPUTING-PLATFORM--------------------------------------------------------------

    # Single GPU
    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'CudaPrecision': 'single'}

    ##If multiple GPUs
    # platformProperties = {'CudaPrecision': 'mixed', 'CUDADeviceIndex': '0,1,2,3'}

    # CPUs
    # platform = Platform.getPlatformByName('CPU')
    # platformProperties = {'CpuThreads': 'default'}

    # SIMULATION-INPUTS---------------------------------------------------------------

    ##variables related to barostat, thermostat, and force calculations

    nonbondedMethod = PME  # Uses Particle-Mesh-Ewald method for electrostatic forces
    nonbondedCutoff = 1.0 * nanometers  # (for CHARMM ff, 1.0 nm for Amberff)
    ewaldErrorTolerance = 0.0005  # Cutoff for PME forces
    constraints = HBonds  # Puts constraints on
    # rigidWater = True                                      #Makes water molecules rigid
    constraintTolerance = 0.000001  # Tolerance for constraints
    temperature = 300 * kelvin  # Temperature in Kelvin
    friction = 2.8284 / picosecond  # Collision frequency for Langevin thermostat
    pz = 1.0 * atmospheres


    parm = LoadParm(parm_name, system_name + 'uniax_NPT4.rst7')
    print('Creating system')

    topology = parm.topology  # Could be psf.topology if done with CHARMM
    positions = parm.positions

    print('Building system...')
    system = parm.createSystem(nonbondedMethod=nonbondedMethod,
                               nonbondedCutoff=nonbondedCutoff,
                               constraints=constraints,
                               ewaldErrorTolerance=ewaldErrorTolerance)

    system.addForce(MonteCarloAnisotropicBarostat((0, 0, pz), temperature, False, False, True, 25))
    ##Generate initil xml file
    f = open(system_name + 'uniax_pres4.xml', 'w+')
    f.write(XmlSerializer.serialize(system))
    f.close()

    print('Setting simulation length...')

    #300 picosecs of uniaxial compression:
    dt = 0.002 * picoseconds

    n_steps = 5000000

    # INTEGRATOR----------------------------------------------------------------------

    print('Creating integrator...')
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    indices_int = []
    for j in range(300,301):
        get_atoms = str(parm.residues[j].atoms)
        indices_str = re.findall(r'\[([^\]]*)\]', get_atoms)
        for item in indices_str:
             for subitem in item.split('['):
                   if(subitem.isdigit()):
                         print('the subitem:')
                         print(subitem)
                         indices_int.append(int(subitem))

    #DEFINE-HARMONIC-RESTRAINT-------------------------------------------------------
    restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1.0*kilocalories_per_mole/angstroms**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    #add restraint to the particles
    for k in indices_int:
        restraint.addParticle(k, parm.positions[k].value_in_unit(u.nanometers))
    system.addForce(restraint)



    g = open(system_name + 'final_NPT.xml', 'w+')
    g.write(XmlSerializer.serialize(integrator))
    g.close()

    # what gets saved from simulation runs

    dcdReporter = DCDReporter(system_name + 'final_NPT.dcd',
                              5000)  ##how often in number of steps should the output file be written
    dataReporter = StateDataReporter(system_name + 'final_NPT.log',  ## other information about the state
                                     5000,
                                     totalSteps=n_steps,
                                     step=True,
                                     speed=True,
                                     progress=True,
                                     elapsedTime=True,
                                     remainingTime=True,
                                     potentialEnergy=True,kineticEnergy=True,totalEnergy=True,
                                     temperature=True,
                                     volume=True,
                                     density=True,
                                     separator='	')
    checkpointReporter = CheckpointReporter(system_name + 'final_NPT.chk', 500000)
    restartReporter = RestartReporter(system_name + 'final_NPT.rst7', reportInterval=500000, netcdf=True)

    print('Creating Simulation object...')

    simulation = Simulation(topology, system, integrator, platform, platformProperties)  # GPU code
    simulation.context.setPositions(positions)  # specifies initial atom positions
    if parm.box_vectors is not None:  ##None is the singeton of Nonetype in python refers to the presence/absence of a value. S						   o (is ==) and (is not !=)
        simulation.context.setPeriodicBoxVectors(*parm.box_vectors)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.reporters.append(dcdReporter)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(checkpointReporter)
    simulation.reporters.append(restartReporter)

    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    simulation.saveState(system_name + 'final_NPT_begin.xml')
    simulation.step(n_steps)
    print('Done final NPT at', pz, 'atm')


    # SAVE-SIMULATION-STATE-----------------------------------------------------------

    print('Saving final state')
    final_state = simulation.context.getState(getForces=True, getVelocities=True, getPositions=True, getEnergy=True,
                                              getParameterDerivatives=True, getParameters=True)
    outfile = open(system_name + 'final_NPT_done.xml', 'w+')
    outfile.write(XmlSerializer.serialize(final_state))
    outfile.close()


openmm_env_path = '/home/adrijad2/om77'

import os
import sys
from openmm.app import *
from openmm import *
from simtk.unit import *
from sys import stdout
from openmm import XmlSerializer
from parmed.amber import LoadParm  # Loads the parmed library we just installed.
from parmed.openmm import RestartReporter
from parmed import unit as u
from mdtraj.reporters import DCDReporter
from openmmtools import states
import re
import numpy as np

folder = '/home/adrijad2/steps/step1_make_polymer_memb/prep_pol/TMPMA95_FOMA5'
os.chdir(folder)


parm_name = 'TMPMA95_FOMA5_15_88_annealing.prmtop'
ambermin_rst = 'TMPMA95_FOMA5_15_88_annealing_min.rst7'
system_name = 'TMPMA05_FOMA5_15_88_annealing_'
run_openmm_uniax_NPT1 (parm_name, ambermin_rst, system_name, fix_bb=False, fix_oth=False)
for i in range(1,11):
	run_openmm_800K_NVT2 (parm_name, system_name, i, fix_bb=False, fix_oth=False)
	run_openmm_300K_NVT3 (parm_name, system_name, fix_bb=False, fix_oth=False)
	run_openmm_uniax_NPT4(parm_name, system_name, fix_bb=False, fix_oth=False)
	print('done with loop number',i)
run_openmm_final_NPT(parm_name, system_name, fix_bb=False, fix_oth=False)


