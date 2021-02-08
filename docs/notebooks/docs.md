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

```python
import numpy as np
from threeML.io.package_data import get_path_of_data_file
from threeML.utils.OGIP.response import OGIPResponse
from threeML import silence_warnings, DispersionSpectrumLike, display_spectrum_model_counts
from threeML import set_threeML_style, BayesianAnalysis, DataList
from astromodels import Blackbody, Powerlaw, PointSource, Model, Log_normal
silence_warnings()
%matplotlib notebook

set_threeML_style()
```

## Generate some synthetic data

```python
# we will use a demo response
response = OGIPResponse(get_path_of_data_file("datasets/ogip_powerlaw.rsp"))
```

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
ba.set_sampler()
```

```python
ba.sample(quiet=True)
```

```python
fig = display_spectrum_model_counts(ba, show_background=True, source_only=False, min_rate=10)
ax = fig.get_axes()[0]

ax.set_ylim(1e-3)

```

## Compute the PPCs

```python
from twopc import compute_ppc
```

```python
ppc = compute_ppc(ba,
                  ba.results,
                  n_sims=500, 
                  file_name="my_ppc.h5",
                  overwrite=True,
                  return_ppc=True)
```

```python
ppc.fake.plot(bkg_subtract=True);
```

```python

```
