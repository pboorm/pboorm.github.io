"""
BXA Tutorial 2023

Session 4

Exercise 4.1 - generate the data

The data will be generated with a Gaussian PhoIndex distribution from model1

WARNING: each BXA directory can be a few MB, so be careful when simulating many of them!
"""

import os, subprocess, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from xspec import *
import bxa.xspec as bxa
from bxa.xspec.solver import set_parameters


def bin_spec(srcfile, outfile, backfile="", respfile="", grouptype="min", groupscale=1):
	"""
	Function to bin the spectra generated using ftgrouppha
	"""
	ftgrouppha_str = "ftgrouppha"
	ftgrouppha_str += " infile=%s" %(srcfile)
	ftgrouppha_str += " outfile=%s" %(outfile)
	ftgrouppha_str += " backfile=%s" %(backfile)
	ftgrouppha_str += " respfile=%s" %(respfile)
	ftgrouppha_str += " grouptype=%s" %(grouptype)
	ftgrouppha_str += " groupscale=%d" %(groupscale)
	ftgrouppha_str += " clobber=yes"
	subprocess.call(ftgrouppha_str, shell = True)

	if os.path.isfile(outfile):
		return outfile
	else:
		print("%s not created?" %(binned_fileName))
		sys.exit()

def sim_spec():
	"""
	Function to simulate spectra from the currently-loaded datafiles
	"""
	fakeit_kwargs = {}
	fakeit_kwargs["response"] = AllData(1).response.rmf
	fakeit_kwargs["arf"] = AllData(1).response.arf
	fakeit_kwargs["exposure"] = AllData(1).exposure
	fakeit_kwargs["correction"] = "1."
	fakeit_kwargs["backExposure"] = AllData(1).background.exposure
	fakeit_kwargs["fileName"] = "sim_ex41.pha"
	fakeit_kwargs["background"] = AllData(1).background.fileName

	fakeit_settings = FakeitSettings(**fakeit_kwargs)
	AllData.fakeit(1, fakeit_settings, applyStats=True)

	bin_fname = bin_spec(fakeit_kwargs["fileName"],
						 fakeit_kwargs["fileName"].replace(".", "_bmin5."),
						 fakeit_kwargs["fileName"].replace(".", "_bkg."),
						 "","bmin",5)

	return bin_fname

## some useful initial settings
Fit.statMethod = "cstat"
Xset.abund = "wilm"
Plot.xAxis = "keV"
Plot.setRebin(2., 1000)
Plot.device = "/xw"
np.random.seed(42)


## the size of the population
population_size = 30

## generate a parent histogram of NH values
fractions = {}
fractions = [0.3, 0.05, 0.15, 0.5]
bins = [20., 21., 22., 23., 24.]
cdf = np.append(0., np.cumsum(fractions) / np.sum(fractions))
interp_cdf = scipy.interpolate.interp1d(cdf, bins, bounds_error = False)
rands = np.random.uniform(size = population_size)
lognhe22_population = interp_cdf(rands) - 22.

## simulate PhoIndex values from a parent Gaussian distribution
PhoIndex_population = np.random.normal(1.9, 0.1, population_size)
lognorm_population = np.random.uniform(-4.8, -2.8, population_size)
sim_df = pd.DataFrame(data = {"PhoIndex": PhoIndex_population, "log(nH)": lognhe22_population, "log(norm)": lognorm_population})

## model to simulate from
sim_model = Model("TBabs*zTBabs*zpowerlw")

run_dir = "./ex4.1_data/"
if not os.path.isdir(run_dir):
	os.mkdir(run_dir)

for i, row in sim_df.iterrows():
	row = row.to_dict()
	outputfiles_basename = "%s/sim_%d" %(run_dir, i)
	
	## first simulate data from the original model fit
	sim_model.TBabs.nH.values = (0.04, -1.) ## Milky Way absorption
	sim_model.zTBabs.Redshift.values = (0.05, -1.)
	sim_model.zTBabs.nH.values = (10 ** (row["log(nH)"]), 0.01, 0.01, 0.01, 200., 200.)
	sim_model.zpowerlw.PhoIndex = (row["PhoIndex"], 0.1, -3., -2., 9., 10.)
	sim_model.zpowerlw.Redshift.link = "p%d" %(sim_model.zTBabs.nH.index)
	sim_model.zpowerlw.norm = (10 ** (row["log(norm)"]), 0.01, 1.e-10, 1.e-10, 1., 1.)

	sim_fname = sim_spec()
	AllData("1:1 %s" %(bin_fname))
	AllData.ignore("1:0.-3. 78.-**")
	Plot("ldata")

	priors = {}
	prior1 = bxa.create_uniform_prior_for(AllModels(1), AllModels(1).zpowerlw.PhoIndex)
	prior2 = bxa.create_loguniform_prior_for(AllModels(1), AllModels(1).zTBabs.nH)
	prior3 = bxa.create_loguniform_prior_for(AllModels(1), AllModels(1).zpowerlw.norm)
	solver = bxa.BXASolver(transformations=[prior1, prior2, prior3], outputfiles_basename=outputfiles_basename)
	set_parameters(transformations=solver.transformations, values=row.values)
	
	results = solver.run(resume=True)


sim_df.to_csv("./%s/simulation_index.csv" %(run_dir), index=False)




































