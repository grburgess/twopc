from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astromodels import clone_model
from threeML import BayesianAnalysis, DataList
from threeML.analysis_results import BayesianResults
from threeML.io.logging import silence_console_log
from threeML.utils.progress_bar import tqdm


def compute_ppc(analysis: BayesianAnalysis,
                result: BayesianResults,
                n_sims: int,
                file_name: str,
                overwrite: bool = False,
                return_ppc: bool = False
                ) -> Union["PPC", None]:
    """
    Compute a posterior predictive check from a 3ML DispersionLike
    Plugin. The resulting posterior data simulations are stored
    in an HDF5 file which can be read by the PPC class

    :param analysis: 3ML bayesian analysis object
    :param result: 3ML analysis result
    :param n_sims: the number of posterior simulations to create
    :param file_name: the filename to save to
    :param overwrite: to overwrite an existsing file
    :param return_ppc: if true, PPC object will be return directy
    :returns: None
    :rtype:

    """

    p = Path(file_name)

    if p.exists() and (not overwrite):

        raise RuntimeError(f"{file_name} already exists!")

    with h5py.File(file_name, 'w', libver='latest') as database:

        # first we collect the real data data and save it so that we will not have to
        # look it up in the future

        data_names = []

        database.attrs['n_sims'] = n_sims

        for data in analysis.data_list.values():

            data_names.append(data.name)
            grp = database.create_group(data.name)
            grp.attrs['exposure'] = data.exposure
            grp.create_dataset(
                'ebounds', data=data.response.ebounds, compression='lzf')
            grp.create_dataset(
                'obs_counts', data=data.observed_counts, compression='lzf')
            grp.create_dataset(
                'bkg_counts', data=data.background_counts, compression='lzf')
            grp.create_dataset('mask', data=data.mask, compression='lzf')

        # select random draws from the posterior

        choices = np.random.choice(
            len(result.samples.T), replace=False, size=n_sims)

        # for each posterior sample

        with silence_console_log(and_progress_bars=False):

            for j, choice in enumerate(tqdm(choices, desc="sampling posterior")):

                # get the parameters of the choice

                params = result.samples.T[choice]

                # set the analysis free parameters to the value of the posterior
                for i, (k, v) in enumerate(analysis.likelihood_model.free_parameters.items()):
                    v.value = params[i]

                # create simulated data sets with these free parameters
                sim_dl = DataList(*[data.get_simulated_dataset()
                                    for data in analysis.data_list.values()])

                # set the model of the simulated data to the model of the simulation
                for i, data in enumerate(sim_dl.values()):

                    # clone the model for saftey's sake
                    # and set the model. For now we do nothing with this

                    data.set_model(clone_model(analysis.likelihood_model))

                    # store the PPC data in the file
                    grp = database[data_names[i]]
                    grp.create_dataset('ppc_counts_%d' %
                                       j, data=data.observed_counts, compression='lzf')
                    grp.create_dataset('ppc_background_counts_%d' %
                                       j, data=data.background_counts, compression='lzf')
                # sim_dls.append(sim_dl)
        if return_ppc:

            return PPC(file_name)


class PPC(object):

    def __init__(self, file_name: str):
        """
        Reads a PPC HDF5 created by compute_ppc. This applies to DispersionLike
        data types only. Each detector is read from the file and an associated
        detector attribute is added to the class allowing the user to access the
        observed and PPC information of the detector


        :param filename: the file name to read
        :returns:
        :rtype:
>
        """

        # open the file

        with h5py.File(file_name, 'r') as f:

            n_sims = f.attrs['n_sims']

            dets = f.keys()

            # go thru all the detectors and grab
            # their data

            self._det_list = {}

            for d in dets:

                ppc_counts = []
                ppc_bkg = []
                obs_counts = f[d]['obs_counts'][()]
                background_counts = f[d]['bkg_counts'][()]
                mask = f[d]['mask'][()]

                ebounds = f[d]['ebounds'][()]

                exposure = f[d].attrs['exposure']

                # scroll thru the PPCS and build up PPC matrix

                for n in range(n_sims):
                    ppc_counts.append(f[d]['ppc_counts_%d' % n][()].tolist())
                    ppc_bkg.append(
                        f[d]['ppc_background_counts_%d' % n][()].tolist())

                # build a detector object and attach it to the class
                det_obj = PPCDetector(d, obs_counts, background_counts, mask, ebounds, exposure, np.array(ppc_counts),
                                      np.array(ppc_bkg))

                setattr(self, d, det_obj)

                self._det_list[d] = det_obj

            self._n_sims = n_sims
            self._dets = dets
            self._filename = file_name

    @ property
    def n_sims(self) -> int:

        return self._n_sims

    @ property
    def detectors(self) -> List[str]:
        return self._det_list


class PPCDetector(object):

    def __init__(self, name: str,
                 obs_counts: np.ndarray,
                 obs_background: np.ndarray,
                 mask: np.ndarray,
                 ebounds: np.ndarray,
                 exposure,
                 ppc_counts,
                 ppc_background):
        """
        This is simply a container object that stores the observed and PPC information of each detector for examination

        :param name:
        :param obs_counts:
        :param obs_background:
        :param mask:
        :param ebounds:
        :param exposure:
        :param ppc_counts:
        :param ppc_background:
        :returns:
        :rtype:

        """

        self._exposure = exposure
        self._obs_counts = obs_counts
        self._obs_background = obs_background
        self._mask = mask
        self._ebounds = ebounds
        self._channel_width = ebounds[1:] - ebounds[:-1]
        self._ppc_counts = ppc_counts
        self._ppc_background = ppc_background
        self._name = name

        self._max_energy = ebounds.max()
        self._min_energy = ebounds.min()

    @ property
    def name(self) -> str:
        return self._name

    @ property
    def obs_counts(self) -> np.ndarray:
        return self._obs_counts

    @ property
    def obs_background(self) -> np.ndarray:
        return self._obs_background

    @ property
    def mask(self) -> np.ndarray:
        return self._mask

    @ property
    def ebounds(self) -> np.ndarray:
        return self._ebounds

    @ property
    def channel_width(self) -> np.ndarray:
        return self._channel_width

    @ property
    def exposure(self) -> float:
        return self._exposure

    @ property
    def ppc_counts(self) -> np.ndarray:
        return self._ppc_counts

    @ property
    def ppc_background(self) -> np.ndarray:
        return self._ppc_background

    def _get_channel_bounds(self, ene_min: float, ene_max: float) -> Tuple[float, float]:

        if ene_min <= self._min_energy:

            lo = 0

        else:

            lo = np.searchsorted(self._ebounds, ene_min) - 1

        if ene_max >= self._max_energy:

            hi = -1

        else:

            hi = np.searchsorted(self._ebounds, ene_max) - 1

        return lo, hi

    def get_integral_ppc(self, ene_min: float, ene_max: float) -> Dict[str, Any]:

        lo, hi = self._get_channel_bounds(ene_min, ene_max)

        counts = self._ppc_counts[:, lo:hi].sum(axis=1)
        bkg = self._ppc_background[:, lo:hi].sum(axis=1)

        obs = self._obs_counts[lo:hi].sum()
        obs_bkg = self._obs_background[lo:hi].sum()

        return dict(ppc_counts=counts,
                    ppc_background=bkg,
                    obs_counts=obs,
                    obs_bkg=obs_bkg
                    )

    def plot(self,
             bkg_subtract: bool = False,
             ax=None, levels: List[float] = [95, 75, 55],
             colors: List[str] = ['#ABB2B9', '#566573', '#17202A'],
             lc: str = '#FFD100',
             lw: float = .9,
             **kwargs

             ):
        """FIXME! briefly describe function

        :param bkg_subtract:
        :param ax:
        :param levels:
        :param 75:
        :param 55]:
        :param colors:
        :param '#566573':
        :param '#17202A']:
        :param lc:
        :param lw:
        :returns:
        :rtype:

        """

        assert len(levels) == len(
            colors), 'Number of levels and number of colors MUST be the same'

        if ax is None:

            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        # compute all the percentiles

        ppc_low = []
        ppc_high = []

        for level in levels:

            if bkg_subtract:

                tmp_low = np.percentile((self._ppc_counts - self._ppc_background) /
                                        self._channel_width / self._exposure, 50. - level / 2., axis=0)
                tmp_high = np.percentile((self._ppc_counts - self._ppc_background) /
                                         self._channel_width / self._exposure, 50. + level / 2., axis=0)

                ppc_low.append(tmp_low)
                ppc_high.append(tmp_high)

            else:

                tmp_low = np.percentile(
                    self._ppc_counts / self._channel_width / self._exposure, 50. - level / 2., axis=0)
                tmp_high = np.percentile(
                    self._ppc_counts / self._channel_width / self._exposure, 50. + level / 2., axis=0)

                ppc_low.append(tmp_low)
                ppc_high.append(tmp_high)

        if bkg_subtract:

            true_rate = (self._obs_counts - self._obs_background) / \
                self._channel_width / self._exposure

        else:
            true_rate = self._obs_counts / self._channel_width / self._exposure

        # colors = [light,mid,dark]

        for j, (lo, hi) in enumerate(zip(ppc_low, ppc_high)):

            for i in range(len(self._ebounds) - 1):
                if self._mask[i]:

                    ax.fill_between(
                        [self._ebounds[i], self._ebounds[i + 1]], lo[i], hi[i], color=colors[j])

        n_chan = len(self._ebounds) - 1

        for i in range(len(self._ebounds) - 1):
            if self._mask[i]:

                ax.hlines(true_rate[i], self._ebounds[i],
                          self._ebounds[i + 1], color=lc, lw=lw)

                if i < n_chan - 1:
                    if self._mask[i + 1]:

                        ax.vlines(
                            self._ebounds[i + 1], true_rate[i], true_rate[i + 1], color=lc, lw=lw)

        ax.set_xscale('log')
        ax.set_yscale('log')

        if bkg_subtract:

            ax.set_ylabel(r'Net Rate [cnt s$^{-1}$ keV$^{-1}$]')

        else:
            ax.set_ylabel(r'Rate [cnt s$^{-1}$ keV$^{-1}$]')
        ax.set_xlabel(r'Energy [keV]')

        return fig
