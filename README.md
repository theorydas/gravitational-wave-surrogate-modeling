# pySurrogate

Inspired by the use of machine learning in the field of gravitational wave astronomy as means of accurately and fastly bypassing costly simulators that would have otherwise took ~days-weeks to provide with results, the pySurrogate package aims to streamline the creation of predictive models which learn to map input parameters X to multi-dimensional array outputs Y. The package combines class wrappers with experience on the creation of surrogates to guide the user by effectively splitting the surrogate creation process into three main components.

1. Initializing the X input data through unit and physical transformations or reshaping.
2. Transfering (and decoding) the Y input data to (and from) a reduced latent space representation and applying other physical transformations.
3. Creating a regressor model which maps the outputs of the first two steps together.

## Contents

* `pySurrogate.py` - Currently the only file featured that defines all relevant wrapper classes.

## Documentation

A thorough documentation of the entire code can be found [here](https://chalk-impulse-d39.notion.site/pySurrogate-b481dfb3e82d4302ad2f8468db1d1886).

## Requirements

Most of the wrapper classes will require `numpy` arrays as inputs or will generate ones by default. Although <ins>not required</ins>, the package is designed to work best with `sklearn`-like scalers.

## TODOs

* Provide example notebooks and plots.
* Improve documentation with example usage.