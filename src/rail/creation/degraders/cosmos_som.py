"""Degrader that emulate specz selection with SOM"""

# import os
import numpy as np
# import pandas as pd
# import pickle
# import tables_io
import qp
from rail.creation.noisifier import Noisifier
# from rail.estimation.algos.somoclu_som import SOMocluInformer
from somoclu import Somoclu
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import TableHandle, PqHandle
from rail.estimation.algos.somoclu_som import get_bmus
from ceci.config import StageParameter as Param

default_noncolor_cols = ["i", "redshift"]
default_noncolor_nondet = [28.62, -1.0]
default_color_cols = ['u', 'g', 'r', 'i', 'z', 'y']
default_colorcol_nondet = [27.79, 29.04, 29.06, 28.62, 27.98, 27.05]


class SOMCOSMOSPZ(Noisifier):
    """Class that creates a mock COSMOS 30-band sample by training a SOM on
    the COSMOS 30-band data, finding the best cell for each galaxy, taking
    the `lp_zbest` value and histogramming those values in a cell and
    creating a qp Ensemble for the cell.  Then, for each input galaxy in
    that cell, draw a random redshift from the ensemble with ens.rvs() to
    construct a rough equivalent redshift distribution.

    The input is assumed to be a pandas dataframe, and the output will be
    another pandas dataframe
    """

    name = "SOMCOSMOSPZ"
    config_options = Noisifier.config_options.copy()
    config_options.update(nondetect_val=SHARED_PARAMS,
                          zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          noncolor_cols=Param(list, default_noncolor_cols, msg="data columns used for SOM, can be a single band if"
                                              "you will also be using colordata in 'color_cols', or can be as many as you want"),
                          noncolor_nondet=Param(list, default_noncolor_nondet, msg="list of nondetect replacement values for the non-color cols"),
                          color_cols=Param(list, default_color_cols, msg="columns that will be differenced to make"
                                           " colors.  This will be done in order, so put in increasing WL order"),
                          color_nondet=Param(list, default_colorcol_nondet, msg="list of nondetect replacement vals for color columns"),
                          som_size=Param(list, [32, 32], msg="tuple containing the size (x, y) of the SOM"),
                          seed=Param(int, 12345, msg="Random number seed"),
                          cosmos_pz_col=Param(str, "lp_zbest", msg="name of column for photoz in COSMOS file"),
                          n_epochs=Param(int, 10, msg="number of training epochs."))

    inputs = [('spec_data', PqHandle),
              ('input', PqHandle),
              ]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def make_data_selection(self, df):
        """make the data to train the som or input to som"""
        df = df.copy()
        usecols = self.config.noncolor_cols.copy()
        colcols = self.config.color_cols
        ncol = len(self.config.color_cols)
        for i in range(ncol - 1):
            tmpname = f"{colcols[i]}" + "min" + f"{colcols[i+1]}"
            usecols.append(tmpname)
            df[tmpname] = df[colcols[i]] - df[colcols[i + 1]]
        # NEED TO SELECT JUST USED ROWS AND OUTPUT TO NUMPY
        subdf = df[usecols].to_numpy()

        return subdf

    def _initNoiseModel(self):
        self.rng = np.random.default_rng(self.config.seed)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
    
    def _addNoise(self):
        """code to do the main SOM-based selection"""
        cosmos_data = self.get_data('spec_data')
        input_data = self.get_data('input')
        ngal = len(input_data[self.config.color_cols[0]])
        pzcol = f"mock_{self.config.cosmos_pz_col}"
        input_data[pzcol] = np.zeros(ngal)
        # do some checks on whether data is set up properly and remove non-detects
        check_vals = self.config.noncolor_cols + self.config.color_cols
        check_lims = self.config.noncolor_nondet + self.config.color_nondet
        for data in (cosmos_data, input_data):
            for val, lim in zip(check_vals, check_lims):
                if val not in data.keys():  # pragma: no cover
                    raise KeyError(f"required key {val} not present in input data file!")
                if np.isnan(self.config.nondetect_val):  # pragma: no cover
                    mask = np.isnan(data[val])
                else:
                    mask = np.logical_or(np.isinf(data[val]), np.isclose(data[val], self.config.nondetect_val))
                # data[val][mask] = lim
                data.loc[mask, val] = np.float32(lim)

        cosmossomdata = self.make_data_selection(cosmos_data)
        photsomdata = self.make_data_selection(input_data)

        SOM = Somoclu(self.config.som_size[0], self.config.som_size[1],
                      gridtype='rectangular', compactsupport=False,
                      maptype='planar', initialization='pca')

        SOM.train(photsomdata, epochs=self.config.n_epochs,)

        phot_bmu_coords = get_bmus(SOM, photsomdata).T
        cosmos_bmu_coords = get_bmus(SOM, cosmossomdata).T


        for i in range(self.config.som_size[0]):
            for j in range(self.config.som_size[1]):
                subset = np.logical_and(phot_bmu_coords[0] == i, phot_bmu_coords[1] == j)
                subsetidx = np.where(subset)[0]
                howmany = np.sum(subset)

                cos_mask = np.logical_and(cosmos_bmu_coords[0] == i, cosmos_bmu_coords[1] == j)
                tmppz = np.array(cosmos_data[self.config.cosmos_pz_col][cos_mask])
                tmphist, bins = np.histogram(tmppz, bins=self.zgrid)
                ens = qp.Ensemble(qp.hist, data=dict(bins=self.zgrid, pdfs=tmphist))
                tmppzs = np.array(ens.rvs(howmany)).flatten()
                #input_data.iloc[subsetidx, pzcol] = tmppzs
                input_data[pzcol][subset] = tmppzs
        self.add_data('output', input_data)
