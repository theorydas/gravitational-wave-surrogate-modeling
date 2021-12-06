# pySurrogate/Examples

Tutorial examples in the usage of pySurrogate and thought process behind the creation of these models.

## Contents

`gravitational waves.ipynb`
A time domain surrogate model that bypasses SEOBNRv4_opt in calculating waveforms of non-spin precessing black hole mergers of mass ratios $q \in [5, 100]$ up to a few dozens of cycles before merger includig ringdown. The model includes strain decomposition, data conditioning and a latent space projection using the greedy basis construction algorithm ROMpy which is mapped onto physical parameters with gaussian processes.

Dependencies include: `pycbc`, `scipy`, `sklearn`, `forked-rompy`.