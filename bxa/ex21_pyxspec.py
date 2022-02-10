"""
This script simulates NuSTAR spectra and fits the simulating model with BXA to generate 
posterior distributions for Exercise 2.1 of the tutorial

The NuSTAR spectral files used for the simulations can be found 
at the following link:
https://www.nustar.caltech.edu/system/media_files/binaries/32/original/nustar_point_sources_for_proposers.tgz?1404172745
(see https://www.nustar.caltech.edu/page/response_files for info)
"""

import numpy as np
from xspec import *
import bxa.xspec as bxa

Plot.xAxis = "keV"
Plot.device = "/null"
Xset.abund = "wilm"
Fit.statMethod = "cstat"

## set up a model
m = Model("tbabs * powerlaw")
    
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
    AllModels(1)(1).values = (10. ** (lognh - 22.), 0.1, 1.e-2, 1.e-2, 1.e4, 1.e4)
    AllModels(1)(2).values = (gam, 0.01, -3., -3., 3., 3.)
    AllModels(1)(3).values = (10. ** lognorm, 0.01, 1.e-8, 1.e-8, 1.e0, 1.e0)
    exposure = 80.e3
    fakeit_kwargs = {}
    fakeit_kwargs["response"] = "nustar.rmf"
    fakeit_kwargs["arf"] = "point_30arcsecRad_1arcminOA.arf"
    fakeit_kwargs["background"] = ""
    fakeit_kwargs["exposure"] = exposure
    fakeit_kwargs["fileName"] = "sim%d_nu.pi" %(i)
    AllData.fakeit(1, FakeitSettings(**fakeit_kwargs))
    AllData.ignore("1:0.-3. 78.-**")

    transformations = [bxa.create_uniform_prior_for(m, m.powerlaw.PhoIndex),
                       bxa.create_loguniform_prior_for(m, m.powerlaw.norm),
                       bxa.create_loguniform_prior_for(m, m.TBabs.nH),
                       ]
    solver = bxa.BXASolver(transformations = transformations, outputfiles_basename = "fitsim%d/" %(i))
    solver.run(resume = True)



