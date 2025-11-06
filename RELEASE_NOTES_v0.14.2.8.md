# ESBMTK v0.14.2.8 Release Notes

**Release Date:** November 6th, 2025

## Overview

ESBMTK 0.14.2.8 is a maintenance and feature enhancement release that improves plotting capabilities, data handling robustness, and adds new features for gas exchange connections. This release continues the 0.14.2.x series which requires Python 3.12.

## What's New

### Bug Fixes

- **Fixed `reverse_time` display for plots with more than one x-axis**: Resolved issues with time axis rendering when multiple x-axes are present in plots
- **Reservoir mass calculations**: Now updated once integration finishes for improved accuracy
- **CSV data encoding**: Enhanced robustness when reading third-party generated CSV data with improved encoding error detection (handles UTF-7 and other exotic encodings from various Excel versions)

### New Features

- **Gas Exchange Connections**: Now support the `scale` keyword for more flexible modeling
- **Unit-registered output variables**: Added support for unit-registered output variables. By adding `_u` suffix to variable names (e.g., `M.D_b.PO4.c_u` instead of `M.D_b.PO4.c`), concentrations will be displayed with units included
- **`reverse_time` keyword expansion**: Signal, ExternalData, and plot functions now accept the `reverse_time` keyword, useful for models running forward in time from a past starting point
- **Enhanced logging**: Warnings are now written to the log file for better debugging and monitoring
- **Plot customization**: ExternalData now accepts the `plot_args` keyword, allowing arbitrary arguments to be passed to plot commands via dictionary (e.g., `plot_args = {"alpha": 0.5}`)

## Installation

ESBMTK 0.14.2.8 is available via:

- **PyPI**: `pip install esbmtk`
- **conda-forge**: `conda install -c conda-forge esbmtk`
- **GitHub**: `git clone https://github.com/uliw/esbmtk.git`

## Requirements

- Python 3.12 or higher (required as of v0.14.2.1)

## Upgrading

Users upgrading from earlier versions should review the [Changelog](https://esbmtk.readthedocs.io/en/latest/changelog.html) for breaking changes introduced in the 0.14.x series.

## Documentation

- Manual: https://esbmtk.readthedocs.io/
- Publication: https://gmd.copernicus.org/articles/18/1155/2025/
- Sample code: https://github.com/uliw/ESBMTK-Examples

## Previous Releases in 0.14.2.x Series

- **0.14.2.2** (May 1st, 2025): Mostly a bugfix release with isotope-related unit tests
- **0.14.2.1**: Major update requiring Python 3.12, cleaned up repository structure, reworked class definitions, fixed isotope calculation regressions, added debug keyword, changed default solver from BDF to LSODA

## Citation

If you use ESBMTK in your research, please cite:

Wortmann et al. (2025) - https://gmd.copernicus.org/articles/18/1155/2025/

## Contributors

Maintained by Ulrich G. Wortmann (uli.wortmann@utoronto.ca)

## License

GPL-3.0-or-later

---

For complete details, see the [full Changelog](https://github.com/uliw/esbmtk/blob/master/CHANGELOG.rst).
