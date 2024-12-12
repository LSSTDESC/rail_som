import os
import numpy as np
import pandas as pd
import pytest
from rail.core.stage import RailStage
from rail.core.data import DATA_STORE, PqHandle
from rail.creation.degraders.specz_som import SOMSpeczDegrader

def test_SOMSpeczDegrader():
    """test of the specz subset degrader"""
    nspec = 1000
    ninput = 10000
    columns = ['redshift', 'u', 'g', 'r', 'i', 'z', 'y']
    specdict = {}
    inputdict = {}
    rng = np.random.default_rng(1138)
    for ii, col in enumerate(columns):
        if ii == 0:
            # numbers from 0 to 3
            specdict[col] = rng.random(nspec) * 3.0
            inputdict[col] = rng.random(ninput) * 3.0
        else:
            # numbers from 14 to 26
            specdict[col] = rng.random(nspec) * 12.0 + 14.0
            inputdict[col] = rng.random(ninput) * 12.0 + 14.0
    specdf = pd.DataFrame(specdict)
    inputdf = pd.DataFrame(inputdict)

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    spec_data = DS.add_data("spec_data", specdf, PqHandle)
    input_data = DS.add_data("input_data", inputdf, PqHandle)


    noncol_cols = ['i', 'redshift']
    col_cols = ['u', 'g', 'r', 'i', 'z', 'y']
    
    noncol_nondet = [28.62, -1.0 ]
    col_nondet = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    som_dict = dict(color_cols=col_cols,
                    noncolor_cols=noncol_cols,
                    nondetect_val=99.0,
                    noncolor_nondet=noncol_nondet,
                    color_nondet=col_nondet)

    som_degrade = SOMSpeczDegrader.make_stage(name="roman_som_degrader", 
                                              output="test_degraded_som.pq", 
                                              **som_dict)
    cutdf = som_degrade(input_data)
    for col in columns:
        assert col in cutdf().keys()
