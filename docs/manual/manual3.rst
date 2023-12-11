==========================================
Seawater & Carbon Chemistry & Gas Exchange
==========================================




1 Seawater & Carbon Chemistry & Gas Exchange
--------------------------------------------

ESBMTK provides several classes that abstract the handling of basin geometry, seawater chemistry and air-sea gas exchange.

1.1 Hypsography
~~~~~~~~~~~~~~~

For many modeling tasks, it is important to have knowledge of a globally averaged hypsometric curve. ESBMTK will automatically create a suitable hypsography instance if a :py:class:`esbmtk.esbmtk.Reservoir()` or :py:class:`esbmtk.extended_classes.ReservoirGroup()` instance is specified with the geometry keyword as in the following example where the first list item denotes the upper depth datum, the second list item, the lower depth datum, and the last list item denotes the fraction of the total ocean area if the upper boundary would be at sealevel.

.. code:: ipython

    Reservoir(
        name="S_b",  # Name of reservoir group
        geometry=[-200, -800, 1],  # upper, lower, fraction
        concentration="1 mmol/kg",
        species=M.DIC,
        register=M,
    )
    print(f"M.S_b.area = {M.S_b.area:.2e}") # surface area at upper depth datum
    print(f"M.S_b.sed_area = {M.S_b.sed_area:.2e}") # surface between upper and lower datum
    print(f"M.S_b.volume = {M.S_b.volume:.2e}") # total volume

This will register 3 new instance variables, and also create a hypsometry instance at the model level that provides access to the following methods:

.. code:: ipython

    #return the ocean area at a given depth in m**2
    print(f"M.hyp.area(0) = {M.hyp.area(0):.2e}")

    # return the area between 2 depth datums in m**2
    print(f"M.hyp.area_dz(0, -200) = {M.hyp.area_dz(0, -200):.2e}")

    # return the volume between 2 depth datums in m**3
    print(f"M.hyp.volume(0,-200) = {M.hyp.volume(0,-200):.2e}")

    # return the total surface area of earth in m**2
    print(f"M.hyp.sa = {M.hyp.sa:.2e}")

Internally, the hypsometric data is parameterized as a spline function that provides a reasonable fit between -6000 mbsl to 1000 above sealevel. The data was fitted against hypsometric data derived from 
Scripps’ SRTM15+V2.5.5 grid (Tozer et al., 2019, `https://doi.org/10.1029/2019EA000658 <https://doi.org/10.1029/2019EA000658>`_), which was down-sampled to a 5 minute grid before processing the hypsometry. The following figure shows a comparison between the spline fit, and the actual data. The file ``hypsometry.py`` provides further examples.

.. _hyp:

.. figure:: ./hyp.png
    :width: 300


    Comparison between spline fit, and the actual data.

1.2 Seawater
~~~~~~~~~~~~

ESBMTK provides a :py:class:`esbmtk.seawater.seawaterConstants()` class that will be automatically instantiated when a :py:class:`esbmtk.extended_classes.ReservoirGroup()` instance 
definition includes the ``seawater_parameters`` keyword. This keyword expects a dictionary that specifies temperature, salinity, and pressure for a given ``Reservoirgroup``. The class methods and instance variables are accessible via the ``swc`` instance.

.. code:: ipython

    ReservoirGroup(
        name="S_b",  # box name
        geometry=[-200, -800, 1],  # upper, lower, fraction
        concentration={M.DIC: "2220 umol/kg", M.TA: "2300 umol/kg"},
        seawater_parameters={
            "T": 25,  # Deg celsius
            "P": 0,  # Bar
            "S": 35,  # PSU
        },
        register=M,
    )
    # Acess the sewater_parameters with the swc instance
    print(f"M.S_b.density = {M.S_b.swc.density:.2e}")

Apart from density, this class will provide access to a host of instance parameters, e.g., equilibrium constants - see :py:meth:`esbmtk.seawater.seaWaterConstants.update_parameters()` for the currently defined names. Most of these values are computed by ``pyCO2SYS`` (`https://doi.org/10.5194/gmd-15-15-2022 <https://doi.org/10.5194/gmd-15-15-2022>`_). Using  ``pyCO2SYS`` provides access to a variety of parametrizations for the respective equilibrium constants, various pH scales, as well a different methods to calculate buffer factors. Unless explicitly specified in the model definition, ESBMTK uses the defaults set by pyCO2SYS. Note that when using the seawater class, the model concentration unit must be set to ``mol/kg`` as in the following example:

.. code:: ipython

    M = Model(
        stop="6 Myr",  # end time of model
        timestep="1 kyr",  # upper limit of time step
        element=["Carbon"],  # list of element definitions
        concentration_unit="mol/kg",
        opt_k_carbonic=13,  # Use Millero 2006
        opt_pH_scale=1,  # 1:total, 3:free scale
        opt_buffers_mode=2, # carbonate, borate water alkalinity only
    )

1.2.1 Caveats
^^^^^^^^^^^^^

- Seawater Parameters are only computed once when the ``ReservoirGroup`` is instantiated, in order to provide an initial steady state. Subsequent changes to seawater chemistry or physical parameters do not affect the initial state.

- The ``swc`` instance provides a ``show()`` method listing most values. However, that list may not be comprehensive.

- See the pyCO2SYS documentation for list of parameters and options `https://pyco2sys.readthedocs.io/en/latest/ <https://pyco2sys.readthedocs.io/en/latest/>`_

- The code example ``seawater_example.py`` in the examples directory

1.3 Carbon Chemistry
~~~~~~~~~~~~~~~~~~~~

1.4 Gas Exchange
~~~~~~~~~~~~~~~~

1.5 Odds and Ends
~~~~~~~~~~~~~~~~~

1.5.1 Weathering
^^^^^^^^^^^^^^^^

1.5.2 Adding your own functions to ESBMTK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.5.3 Calling model specific functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
