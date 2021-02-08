import numpy as np
from astromodels import Blackbody, Log_normal, Model, PointSource, Powerlaw
from threeML import (BayesianAnalysis, DataList, DispersionSpectrumLike,
                     display_spectrum_model_counts
                     debug_mode)
from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import OGIPResponse
from twopc import compute_ppc

debug_mode()


def test_all():
    response = OGIPResponse(
        get_path_of_data_file("datasets/ogip_powerlaw.rsp"))

    np.random.seed(1234)

    # rescale the functions for the response
    source_function = Blackbody(K=1e-7, kT=500.0)
    background_function = Powerlaw(K=1, index=-1.5, piv=1.0e3)
    spectrum_generator = DispersionSpectrumLike.from_function(
        "fake",
        source_function=source_function,
        background_function=background_function,
        response=response)

    source_function.K.prior = Log_normal(mu=np.log(1e-7), sigma=1)
    source_function.kT.prior = Log_normal(mu=np.log(300), sigma=2)

    ps = PointSource("demo", 0, 0, spectral_shape=source_function)

    model = Model(ps)


    ba = BayesianAnalysis(model, DataList(spectrum_generator))


    ba.set_sampler()


    ba.sample(quiet=True)


    ppc = compute_ppc(ba,
                  ba.results,
                  n_sims=500, 
                  file_name="my_ppc.h5",
                  overwrite=True,
                  return_ppc=True)


    ppc.fake.plot(bkg_subtract=True);



    ppc.fake.plot(bkg_subtract=False);



