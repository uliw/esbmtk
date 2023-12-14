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

1.3.1 pH
^^^^^^^^

Unless explicitly requested (see above), pH will be reported on the total scale. The hydrogen ion concentration ([H\ :sup:`+`\]) is computed by pyCO2SYS based on the initial DIC and total alkalinity (TA) concentrations. Subsequent hydrogen concentration calculations use the iterative approach of Follows et al. 2005 (`https://doi.org/10.1016/j.ocemod.2005.05.004 <https://doi.org/10.1016/j.ocemod.2005.05.004>`_). 

Provided that the model has terms for DIC and TA, pH calculations for a given :py:class:`esbmtk.extended_classes.ReservoirGroup()` instance are added using the :py:func:`esbmtk.utility_functions.add_carbonate_system_1()` function:

.. code:: ipython

    box_names = [A_sb, I_sb, P_sb, H_sb]  # list of ReservoirGroup handles
    add_carbonate_system_1(box_names)

This will create Reservoirs :py:class:`esbmtk.esbmtk.Reservoir()` instances for ``Hplus`` and ``CO2aq``. After running the model, the resulting concentration data is available in the usual manner as:

.. code:: ipython

    A_sb.Hplus.c
    A_sb.CO2aq.c

The remaining carbonate species are calculated during post-processing (see the :py:func:`esbmtk.post_processing.carbonate_system_1_pp.()` function) and are available as

.. code:: ipython

    A_sb.pH
    A_sb.HCO3
    A_sb.CO3
    A_sb.Omega

1.3.1.1 Notes:
::::::::::::::

- that the resulting concentration data depends on the choice of equilibrium constants and how they are calculated (see the ``opt_k_carbonic``, ``opt_buffers_mode`` keywords above).

- Also note that the data from post-processing is currently available as :py:class:`esbmtk.extended_classes.VectorField()` instance, rather than as :py:class:`esbmtk.esbmtk.Reservoir()` instance.

- Reservoirs that use carbonate system 2 (see below), do not need to use carbonate system 1

- ESBMTK will print a warning message of the pH changes by more than 0.01 units per time step. However, this is only a crude measure, since the solver also uses interpolation between integration steps. So this may not catch all possible scenarios.

1.3.2 Carbonate burial and dissolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Carbonate burial and dissolution use the parametrization proposed by Boudreau et al. 2010 (`https://doi.org/10.1029/2009gb003654 <https://doi.org/10.1029/2009gb003654>`_). The current ESBMTK implementation  has the following short comings:

- It only considers Calcium dissolution/burial (although it would be easy to add a Aragonite)

- Results will only be correct as long as the depth of the saturation horizon remains below the upper depth datum of the deep-water box. Future versions will address this limitation

The following figure provides an overview of the parametrizations and variables used by the  :py:func:`esbmtk.bio_pump_functions0.carbonate_chemisrty.carbonate_system_2()` and :py:func:`esbmtk.bio_pump_functions0.carbonate_chemisrty.add_carbonate_system_2()` functions.

.. _boudreau:

.. figure:: ./boudreau.png
    :width: 800


    Overview of the the parametrizations and variables used by the :py:func:`esbmtk.bio_pump_functions0.carbonate_chemisrty.carbonate_system_2()` and :py:func:`esbmtk.bio_pump_functions0.carbonate_chemisrty.add_carbonate_system_2()` functions. Image Credit: Tina Tsan & Mahruk Niazi

Provided a given model has data for DIC & TA, and that the carbonate export flux is known, ``carbonate_system_2`` can be added to a ReservoirGroup instance in the following way:

.. code:: ipython

    surface_boxes: list = [M.L_b]
    deep_boxes: list = [M.D_b]
    export_fluxes: list = M.flux_summary(filter_by="PIC_DIC L_b", return_list=True)

    add_carbonate_system_2(
            r_db=deep_boxes,  # list of reservoir groups
            r_sb=surface_boxes,  # list of reservoir groups
            carbonate_export_fluxes=export_fluxes,  # list of export fluxes
            z0=-200,  # depth of shelf
            alpha=alpha,  # dissolution coefficient, typically around 0.6
        )

Notes:

- boxes and fluxes are lists, since in some models there is more than one surface box (e.g., models that resolve individual ocean basins)

- ESBMTK only considers the sediment area to 6000 mbsl. The area contributed by the elevations below 6000 mbsl is negligible, and this constrain simplifies the hypsographic fit.

- The total sediment area of a given ``ReservoirGroup`` is known provided the box-geometry was specified correctly.

- The :py:func:`esbmtk.bio_pump_functions0.carbonate_chemisrty.carbonate_system_2()` function only returns [H\ :sup:`+`\] and the dissolution flux for  given box. It does not return the burial flux.

- Please study the actual model implementations provided in the examples folder.

1.3.3 Post processing
^^^^^^^^^^^^^^^^^^^^^

As with ``carbonate_system_1`` the remaining carbonate species are not part of the equation system, rather they are calculated once a solution has been found. Since the solver does not store the carbonate export fluxes, one first has to calculate the relevant fluxes from the concentration data in the model solution. This however model dependent (i.e., export productivity as a function of residence time, or as function of upwelling flux), and as such post-processing of ``carbonate_system_2``  is not done automatically, but has to be initiated manually, e.g., like this:

.. code:: ipython

    # get CaCO3_export in mol/year
    CaCO3_export = M.CaCO3_export.to(f"{M.f_unit}").magnitude
    carbonate_system_2_pp(
        M.D_b,  # ReservoirGroup
        CaCO3_export,  # CaCO3 export flux
        200,  # z0
        6000,  # zmax
    )

This will compute all carbonate species similar to ``carbonate_system_1_pp``, and in addition calculate:

.. code:: ipython

    M.D_b.Fburial  # CaCO3 burial flux mol/year
    M.D_b.Fdiss  # CaCO3 dissolution flux mol/year
    M.D_b.zsat  # Saturation depth in mbsl
    M.D_b.zcc  # CCD depth in mbsl
    M.D_b.zsnow  # Snowline depth in mbsl

see  the :py:func:`esbmtk.post_processing.carbonate_system_2_pp.()` function for details

1.4 Gas Exchange
~~~~~~~~~~~~~~~~

1.5 Weathering
~~~~~~~~~~~~~~