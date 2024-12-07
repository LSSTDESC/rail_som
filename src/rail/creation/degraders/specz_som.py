"""Selector that emulate specz selection with SOM"""

# import os
import numpy as np
# import pandas as pd
# import pickle
# import tables_io
from rail.creation.selector import Selector
# from rail.estimation.algos.somoclu_som import SOMocluInformer
from somoclu import Somoclu
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import TableHandle
from rail.estimation.algos.somoclu_som import get_bmus
from ceci.config import StageParameter as Param

default_noncolor_cols = ["i", "redshift"]
default_noncolor_nondet = [28.62, -1.0]
default_color_cols = ['u', 'g', 'r', 'i', 'z', 'y']
default_colorcol_nondet = [27.79, 29.04, 29.06, 28.62, 27.98, 27.05]


class SOMSpeczDegrader(Selector):
    """Class that creates a specz sample by training
    a SOM on data with spec-z, classifying all galaxies
    from a larger sample via the SOM, then selecting the
    same number of galaxies in each SOM cell as there
    are in the specz sample.  If fewer galaxies are
    available in the large sample for a cell, it just
    takes as many as possible, so you can still mismatch
    the distribution numbers"""

    name = "SOMSpeczDegrader"
    config_options = Selector.config_options.copy()
    config_options.update(nondetect_val=SHARED_PARAMS,
                          noncolor_cols=Param(list, default_noncolor_cols, msg="data columns used for SOM, can be a single band if"
                                              "you will also be using colordata in 'color_cols', or can be as many as you want"),
                          noncolor_nondet=Param(list, default_noncolor_nondet, msg="list of nondetect replacement values for the non-color cols"),
                          color_cols=Param(list, default_color_cols, msg="columns that will be differenced to make"
                                           " colors.  This will be done in order, so put in increasing WL order"),
                          color_nondet=Param(list, default_colorcol_nondet, msg="list of nondetect replacement vals for color columns"),
                          som_size=Param(tuple, (32, 32), msg="tuple containing the size (x, y) of the SOM"),
                          n_epochs=Param(int, 10, msg="number of training epochs."))

    inputs = [('spec_data', TableHandle),
              ('input', TableHandle),
              ]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # if self.config.redshift_cut < 0:
        #     raise ValueError("redshift cut must be positive")

    def make_data_selection(self, df):
        """make the data to train the som or input to som"""
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

    def _select(self):
        """code to do the main SOM-based selection"""
        spec_data = self.get_data('spec_data')
        deep_data = self.get_data('input')
        # do some checks on whether data is set up properly and remove non-detects
        check_vals = self.config.noncolor_cols + self.config.color_cols
        check_lims = self.config.noncolor_nondet + self.config.color_nondet
        for data in (spec_data, deep_data):
            print("doing checks")
            for val, lim in zip(check_vals, check_lims):
                if val not in data.keys():  # pragma: no cover
                    raise KeyError(f"required key {val} not present in input data file!")
                if np.isnan(self.config.nondetect_val):  # pragma: no cover
                    mask = np.isnan(data[val])
                else:
                    mask = np.logical_or(np.isinf(data[val]), np.isclose(data[val], self.config.nondetect_val))
                # data[val][mask] = lim
                data.loc[mask, val] = lim

        specsomdata = self.make_data_selection(spec_data)
        photsomdata = self.make_data_selection(deep_data)

        SOM = Somoclu(self.config.som_size[0], self.config.som_size[1],
                      gridtype='rectangular', compactsupport=False,
                      maptype='planar', initialization='pca')

        SOM.train(photsomdata, epochs=self.config.n_epochs,)

        phot_bmu_coords = get_bmus(SOM, photsomdata).T
        spec_bmu_coords = get_bmus(SOM, specsomdata).T

        total_mask = np.zeros(len(deep_data), dtype=bool)

        for i in range(self.config.som_size[0]):
            for j in range(self.config.som_size[1]):
                subset = np.logical_and(phot_bmu_coords[0] == i, phot_bmu_coords[1] == j)
                subsetidx = np.where(subset)[0]
                lengal = np.sum(subset)
                howmany = np.sum(np.logical_and(spec_bmu_coords[0] == i, spec_bmu_coords[1] == j))
                if howmany > lengal:
                    howmany = lengal
                perm = np.random.permutation(lengal)
                total_mask[subsetidx[perm][:howmany]] = True

        return total_mask
