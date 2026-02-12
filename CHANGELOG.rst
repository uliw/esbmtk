    :Author: Uli Wortmann

.. contents::

1 Changelog
-----------

1.1 0.14.2.8 Nov 2025
~~~~~~~~~~~~~~~~~~~~~

1.1.1 New Features:
^^^^^^^^^^^^^^^^^^^

- Gas Exchange Connections now allow the ``scale`` keyword

- Signal, ExternalData, and plot now accept the ``reverse_time`` keyword

- Warnings are now written to the log file

- ExternalData now accepts the ``plot_args`` keyword which allows to pass arbitrary arguments to a plot command in the form of a dictionary (e.g., ``plot_args = {"alpha": -.5"}`` ).

- Unit-registered output variables: append \_u to reservoir attributes
  (e.g., M.S\ :sub:`b.c`\ \ :sub:`u`\, M.S\ :sub:`b.m`\ \ :sub:`u`\, M.time\ :sub:`u`\) to retrieve values with units
  attached.

- Improved toolchain settings and linting configuration added to
  pyproject.toml (ruff, basedpyright) and updated requirements for
  development tooling.

.. _bug-fixes:

1.1.2 Bug fixes
^^^^^^^^^^^^^^^

- Fix reverse\ :sub:`time`\ plot display for figures with multiple x-axes (plots
  now correctly invert/time-format only axes marked as reversible).

- Reservoir mass bookkeeping fixed: reservoir mass values are updated at
  the end of integration and \_u suffixed properties reflect units
  correctly.

- Multiple plotting fixes: improved unit-conversion logic for plotting
  functions, fewer formatting errors, and more consistent legends.

.. _tests:

1.1.3 Tests
^^^^^^^^^^^

- Added many new integration/unit tests and CSV fixtures under tests/:

  - Signal/ExternalData parsing and plotting (including ``reverse_time``
    cases).

  - Isotope-signal handling (``epsilon_only``, mixed units).

  - Misc. fixes to existing tests to match new behavior.

.. _internal-developer-changes:

1.1.4 Internal / developer changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- src/esbmtk/\*init\*.py reorganized --- package public API exports were
  restructured and some symbol names/locations changed; initialization
  now uses a centralized unit registry.

- Added ``src/esbmtk/initialize_unit_registry.py`` to centralize Q\_/ureg.

- Added ``src/esbmtk/version.py`` to provide package version without
  circular imports.

- Reworked ODE-constant bookkeeping: new toc/doc/gcc mechanism for
  tracking constants passed to the ODE backend (affects how
  scale/rate/delta/epsilon constants are added).

- Replaced many print statements with structured logging (logging
  module) and improved run() logging + resource reporting.

- Updated pyproject.toml, requirements.txt, and setup.cfg for Python >=
  3.12 and new dev dependencies (ruff, basedpyright, sphinx, pytest,
  tox, etc).

.. _breaking-changes-migration-notes:

1.1.5 Breaking changes / migration notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Python requirement: ESBMTK now requires Python 3.12+.

- Public API reshuffle:

  - **init** reorganized; some imports that previously came from top-level
    esbmtk may have moved or been renamed. Users importing internal
    names should re-check imports

  - Unit registry: Q\_ and ureg are now provided by
    ``esbmtk.initialize_unit_registry`` (available via from esbmtk import
    Q\_, ureg). If you previously created your own UnitRegistry, adopt
    the package-provided registry.

- Connections:

  - Chaining of ``scale_with_flux`` references is still unsupported and will
    not work --- do not create flux A scaled by B when B itself is
    ``scale_with_flux`` of C.

- ODE constants and indexing changed:

  - Connections now register rate/scale/delta/epsilon constants into the
    global ODE constant table (toc/doc). If you generate or parse the
    equations file programmatically, check for changed constant
    indices/keys.

1.2 Older versions
~~~~~~~~~~~~~~~~~~

  - ExternalData now accepts the ``plot_args`` keyword which allows to pass arbitrary arguments to a plot command in the form of a dictionary (e.g., ``plot_args = {"alpha": -.5"}`` ).

- 0.14.2.2 May 1\ :sup:`st`\ 2025, mostly  a bugfix release that adds isotope related unit test.

- 0.14.2.1 ESBMTK now requires python 3.12. Cleaned up repository structure. This may result in import errors. Please get in touch if it does. Reworked many class definitions to improve readability and documentation. Fixed a variety of regressions for isotope calculations that were introduces in 0.14.1.x  Added ``debug`` keyword to the Model object. Expanded the debug information in the equations file. The default solver has changed from BDF to LSODA.

- 0.14.1.4 Fixed a regression that caused incorrect ``zcc`` values in the post-processing step. Added  the ``reverse_time`` keyword to the plot function. This is most useful for models that run forward in time from past starting point.

- 0.14.1.x reworked the structure of the equations file, as well as the code that generates forcing functions. ``scale_by_concentration`` connections now correct for density. Both changes results in small changes to the numerical solutions. Also added link to the GMD publication describing the library.

- 0.14.0.x simplified the setup of the gas exchange connections, documentation improvements, ``number_of_max_datapoints`` keyword has been removed. Currently, this results in a huge memory demand for jobs that integrate over millions of years with small time steps (say 10 years)

- 0.13.0.8 added new testing framework, and 16 unit tests. Updated the documentation

- 0.13.0.7 the isotope fractionation factor has been renamed from ``alpha`` to ``epsilon``

- 0.13.0.5 ``tinmestep`` keyword is now deprecated and should be replaced with ``max_timestep``. ``min_time_step`` is now a configurable option.

- 0.13.0.x The equations file is now a temporary file with a random name. This allows for the parallel execution of model code.

- 0.13.0.x plot() will now return the figure and axes handles, and accepts the ``no_show`` option. This allows for the addition of manual plot elements in post-processing

- May 10th v 0.13.0.0 fixes an error in the solubility calculation for
  oxygen and carbon dioxide. This will change the steady state results
  for pCO2 in existing models by about 4 ppm. This version renamed many classes.
  Existing models may need some editing.

  - Element -> ElementProperties

  - Species -> SpeciesProperties

  - Reservoir -> Species

  - ReservoirGroup -> Reservoir

  - ConnectionGroup -> ConnectionProperties

  - Connection -> Connect

  - SourceGroup -> SourceProperties

  - SinkGroup -> SinkProperties

  Since the respective ConnectionProperty, SourceProperty and SinkProperty
  objects which correlate with the former ConnectionGroup, SourceGroup
  SinkGroup classes. As such, existing code must be changed from

  .. code:: ipython

      Sink(
      name="burial",
      species=M.PO4,
      register=M,  #
      )

  to

  .. code:: ipython

      SinkProperties(
      name="burial",
      species=[M.PO4],
      )

  and

  .. code:: ipython

      Connection(  #
      source=M.S_b,  # source of flux
      sink=M.D_b,  # target of flux
      ctype="scale_with_concentration",
      scale=M.S_b.volume / tau,
      id="primary_production",
      )

to

.. code:: ipython

    ConnectionProperties(  #
    source=M.S_b,  # source of flux
    sink=M.D_b,  # target of flux
    ctype="scale_with_concentration",
    scale=M.S_b.volume / tau,
    id="primary_production",
    species=[M.PO4],  # apply this only to PO4
    )

- May 1st, v 0.12.0.28 ESBMTK can now be installed via conda. Various
  documentation updates

- Dec. v 0.12.0.x This is a breaking change that requires the following
  updates to the model definition.

  - Models that use isotope calculations need to ensure that sources and
    sink also specify the isotope keyword.

  - Weathering and Gas-exchange have now become connection properties,
    see the examples in the online documentation

  - Models that used carbonate\ :sub:`system`\ \ :sub:`1`\ \ :sub:`pp`\() no longer need to call this
    specifically, as this function is now called automatically

- Oct. 12\ :sup:`th`\, 2023 v 0.11.0.2 This is a breaking change. Added support
  to specify box area and volume explicitly, rather than as a function
  of hypsography. This is likely to affect existing geoemtry definitions
  since the (area/total area) parameter has changed meaning The area
  fraction is now calcualted automatically, and unless you split the
  model in specific basins the last parameter in the geometry list
  should always be 1 (i.e., [0, -350, 1]).

  Equilibrium constants are now calculated by pyCO2SYS. This facilitates
  a wide selection of parametrizations via the ``opt_k_carbonic`` and
  ``opt_pH_scale`` keywords in the Model definition. Options and defaults
  are the same as for pyCO2SYS.

- Oct. 30\ :sup:`th`\, 2023 v 0.10.0.11 This is a breaking change.
  Remineralization and photosynthesis must be implemented via functions,
  rather than transport connections. CS1 and CS2 are retired, and
  replaced by photosynthesis, organic-matter remineralization and
  carbonate-dissolution functions. I've started writing a user guide,
  see `https://esbmtk.readthedocs.io/en/latest/ESBMTK-Tutorial.html <https://esbmtk.readthedocs.io/en/latest/ESBMTK-Tutorial.html>`_

So far, only the very basics are covered. More to come!

- July 28\ :sup:`th`\, 2023, v 0.9.0.1 The ODEPACk backend is now fully
  functional, and the basic API is more or less stable.

- Nov. 11\ :sup:`th`\2022, v 0.9.0.0 Moved to odepack based backend. Removed
  now defunct code. The odepack backend does not yet support isotope
  calculations.

- 0.8.0.0

  - Cleanup of naming scheme which is now strictly hierarchical.

  - Bulk connection dictionaries now have to be specified as
    ``source_to_sink`` instead of ``source2sink``.

  - The connection naming scheme has been revamped. Please see
    ``esbmtk.connect.__set_name__()`` documentation for details.

  - Model concentration units must now match 'mole/liter' or 'mol/kg'.
    Concentrations can still be specified as ``mmol/l`` or ``mmol/kg``, but
    model output will be in mole/liter or kg. At present, the model does
    not provide for the automatic conversion of mol/l to mol/kg. Thus
    you must specify units in a consistent way.

  - The SeawaterConstants class now always returns values as mol/kg
    solution. Caveat Emptor.

  - The SeawaterConstants class no longer accepts the 'model' keyword

  - All of his will break existing models.

  - Models assume by default that they deal with ideal water, i.e.,
    where the density equals one. To work with seawater, you must set
    ``ideal_water=False``. In that case, you should also set the
    ``concentration_unit`` keyword to ``'mol/kg'`` (solution).

  - Several classes now require the "register" keyword. You may need to
    fix your code accordingly

- The flux and connection summary methods can be filtered by more than
  one keyword. Provide a filter string in the following format
  ``"keyword_1 keyword_2`` and it will only return results that match both
  keywords.

- Removed the dependency on the nptyping and number libraries

- 0.7.3.9 Moved to setuptools build system. Lost of code fixes wrt
  isotope calculations, minor fixes in the carbonate module.

- March 2\ :sup:`nd`\0.7.3.4 ``Flux_summary`` now supports an ``exclude`` keyword.
  Hot fixed an error in the gas exchange code, which affected the total
  mass of atmosphere calculations. For the time being, the mass of the
  atmosphere is treated as constant.

- 0.7.3.0 Flux data is no longer kept by default. This results in huge
  memory savings. esbmtk now requires python 3.9 or higher, and also
  depends on ``os`` and ``psutil``. the scale with flux process now uses the
  ``ref_flux`` keyword instead of ``ref_reservoirs``. Models must adapt
  their scripts accordingly. esbmtk objects no longer provide delta
  values by default. Rather they need to be calculated in the
  post-processing step via ``M.get_delta_values()``. The ``f_0`` keyword in
  the weathering connection is now called ``rate``. Using the old keyword
  will result in a unit error.

- January 8\ :sup:`th`\0.7.2.2 Fixed several isotope calculation regressions.
  Added 31 Unit tests.
