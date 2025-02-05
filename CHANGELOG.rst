    :Author: Uli Wortmann

.. contents::

1 Changelog
-----------

- 0.14.0.x simplified the setup of the gas exchange connections, documentation improvements, ``number_of_max_datapoints`` keyword has been removed.

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
