---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Creating PPC from GBM SpectrumLike fits

A very useful way to check the quality of Bayesian fits is through posterior predicitve checks.
We can create these with the machinery inside of 3ML for certain plugins (perhaps more in the future). 



```python
import numpy as np
from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import OGIPResponse
from threeML import silence_warnings, DispersionSpectrumLike, display_spectrum_model_counts
from threeML import set_threeML_style, BayesianAnalysis, DataList, threeML_config
from astromodels import Blackbody, Powerlaw, PointSource, Model, Log_normal

threeML_config.plugins.ogip.data_plot.counts_color = "#FCE902"
threeML_config.plugins.ogip.data_plot.background_color = "#CC0000"

```


```python nbsphinx="hidden"
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)

silence_warnings()

set_threeML_style()

threeML_config.plugins.ogip.data_plot.counts_color = "#FCE902"
threeML_config.plugins.ogip.data_plot.background_color = "#CC0000"


threeML_config.plugins.ogip.fit_plot.background_mpl_kwargs = dict(ls="-", lw=.7)
threeML_config.interface.multi_progress_cmap = "Reds"
```
## Generate some synthetic data

First, lets use 3ML to generate some fake data. Here we will have a background spectrum that is a power law and a source that is a black body.

### Grab a demo response from the 3ML library

```python
# we will use a demo response
response = OGIPResponse(get_path_of_data_file("datasets/ogip_powerlaw.rsp"))
```

### Simulate the data

We will mimic the some normal c-stat style data.

```python
np.random.seed(1234)

# rescale the functions for the response
source_function = Blackbody(K=1e-7, kT=500.0)
background_function = Powerlaw(K=1, index=-1.5, piv=1.0e3)
spectrum_generator = DispersionSpectrumLike.from_function(
    "fake",
    source_function=source_function,
    background_function=background_function,
    response=response,
)

fig = spectrum_generator.view_count_spectrum()
```

## Fit the data 

We will quickly fit the data via Bayesian posterior sampling... After all you need a posterior to do posterior predictive checks!


```python
source_function.K.prior = Log_normal(mu = np.log(1e-7), sigma = 1 )
source_function.kT.prior = Log_normal(mu = np.log(300), sigma = 2 )

ps = PointSource("demo", 0, 0, spectral_shape=source_function)

model = Model(ps)
```

```python
ba = BayesianAnalysis(model, DataList(spectrum_generator))
```

```python
ba.set_sampler("emcee")
ba.sampler.setup(n_iterations=1000, n_walkers="50")
ba.sample(quiet=False)
```

Ok, we have a pretty decent fit. But is it?

```python
ba.restore_median_fit()
fig = display_spectrum_model_counts(ba,
                                    data_color="#CC0000",
                                    model_color="#FCE902",
                                    background_color="k",
                                    show_background=True,
                                    source_only=False,
                                    min_rate=10)
ax = fig.get_axes()[0]

_ = ax.set_ylim(1e-3)

```

<!-- #region -->
## Compute the PPCs

We want to check the validiity of the fit via posterior predicitive checks (PPCs).
Essentially:

$$\pi\left(y^{\mathrm{rep}} \mid y\right)=\int \mathrm{d} \theta \pi(\theta \mid y) \pi\left(y^{\mathrm{rep}} \mid \theta\right)$$


So, we will randomly sample the posterior, fold those point of the posterior model throught the likelihood and sample new counts. As this is an expensive process, we will want to save this to disk (in an HDF5 file).

<!-- #endregion -->

```python
from twopc import compute_ppc
```

```python
ppc = compute_ppc(ba,
                  ba.results,
                  n_sims=1000, 
                  file_name="my_ppc.h5",
                  overwrite=True,
                  return_ppc=True)
```

Lets take a look at the results of the background substracted rate

```python
fig = ppc.fake.plot(bkg_subtract=True,colors=["#FF1919","#CC0000","#7F0000"]);
```

And the PPCs with source + background

```python
fig = ppc.fake.plot(bkg_subtract=False,colors=["#FF1919","#CC0000","#7F0000"]);
```

** AWESOME! ** It looks like our fit and accurately produce future data! 

```python

```
