ESBMTK 0.14.2.8 is a maintenance and feature enhancement release that improves plotting capabilities, data handling robustness, and adds new features for gas exchange connections.

## Key Changes

### Bug Fixes
- Fixed `reverse_time` display for plots with more than one x-axis
- Reservoir mass calculations are now updated once integration finishes
- Enhanced robustness when reading third-party generated CSV data with improved encoding error detection (handles UTF-7 and other exotic encodings)

### New Features
- **Gas Exchange Connections** now support the `scale` keyword
- **Unit-registered output variables**: Add `_u` suffix to variable names (e.g., `M.D_b.PO4.c_u`) to display concentrations with units
- **`reverse_time` keyword**: Now supported by Signal, ExternalData, and plot functions
- **Enhanced logging**: Warnings are now written to the log file
- **Plot customization**: ExternalData now accepts the `plot_args` keyword for passing arbitrary arguments (e.g., `plot_args = {"alpha": 0.5}`)

## Requirements
- Python 3.12 or higher

## Installation
```bash
# PyPI
pip install esbmtk

# conda-forge
conda install -c conda-forge esbmtk
```

## Documentation
- Manual: https://esbmtk.readthedocs.io/
- Publication: https://gmd.copernicus.org/articles/18/1155/2025/
- Examples: https://github.com/uliw/ESBMTK-Examples

## Citation
If you use ESBMTK in your research, please cite: Wortmann et al. (2025) - https://gmd.copernicus.org/articles/18/1155/2025/

---
Full changelog: https://github.com/uliw/esbmtk/blob/master/CHANGELOG.rst
