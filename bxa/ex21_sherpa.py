"""
This script simulates NuSTAR spectra and fits the simulating model with BXA to generate 
posterior distributions for Exercise 2.1 of the tutorial

The NuSTAR spectral files used for the simulations can be found 
at the following link:
https://www.nustar.caltech.edu/system/media_files/binaries/32/original/nustar_point_sources_for_proposers.tgz?1404172745
(see https://www.nustar.caltech.edu/page/response_files for info)
"""

import numpy as np
import bxa.sherpa as bxa

set_xsabund("wilm")
set_xsxsect("vern")
set_xscosmo(67.3, 0., 0.685)
set_analysis("energy")
set_stat("wstat")

## set up a model
model = xstbabs.tbabs * xspowerlaw.pl

set_par(tbabs.nh, val = 10., frozen = False, min = 1.e-2, max = 1.e4)
set_par(pl.PhoIndex, val = 1.9, min = -3., max = 3.)
set_par(pl.norm, val = 5.e-3, min = 1.e-8, max = 1.)

## generate a population
np.random.seed(42)

## the number of sources to simulate
population_size = 30

## distributions being simulated from
gam_values = np.random.normal(loc = 1.8, scale = 0.5, size = population_size)
lognorm_values = np.random.uniform(low = -4., high = -2., size = population_size)
lognh_values = np.random.uniform(low = 22., high = 23., size = population_size)

for i, (gam, lognorm, lognh) in enumerate(zip(gam_values, lognorm_values, lognh_values)):
    ## load the parameter values to be simulated
    tbabs.nh = 10. ** (lognh - 22.)
    pl.PhoIndex = gam
    pl.norm = 10. ** lognorm

    set_source(1, model)
    fakepha_kwargs = {}
    fakepha_kwargs["rmf"] = unpack_rmf("nustar.rmf")
    fakepha_kwargs["arf"] = unpack_arf("point_30arcsecRad_1arcminOA.arf")
    fakepha_kwargs["backscal"] = 1.
    fakepha_kwargs["grouped"] = False
    fakepha_kwargs["exposure"] = 80.e3

    fake_pha(1, **fakepha_kwargs)
    save_pha(1, "sim%d_nu.pi" %(i), clobber = True)

    load_pha(1, "sim%d_nu.pi" %(i))
    ignore_id(1, "0.:0.5,8.:")
    set_source(1, model)


    ## free parameters
    parameters = [tbabs.nh, pl.PhoIndex, pl.norm]

    ## priors
    priors = [bxa.create_loguniform_prior_for(m, m.TBabs.nH),
              bxa.create_uniform_prior_for(pl.PhoIndex),
              bxa.create_loguniform_prior_for(pl.norm)
              ]
    priorfunction = bxa.create_prior_function(priors)

    solver = bxa.BXASolver(prior=priorfunction, parameters=parameters, outputfiles_basename = "fitsim%d/" %(i))

    solver.run(resume = True)
