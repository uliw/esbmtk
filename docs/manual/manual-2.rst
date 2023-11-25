=============
ESBMTK Manual
=============




1 Implementing more complex models
----------------------------------

1.1 Adding model forcing
~~~~~~~~~~~~~~~~~~~~~~~~

ESBMTK realizes model forcing through the ``Signal`` class. Once defined, a signal instance can be associated with a ``Connection`` instance and will then act on the associated connection.
This class provides the following methods to create a signal:

- ``square()``, ``pyramid()``, ``bell()``  These are defined by specifying the signal start time (relative to the model time), its size (as mass) and duration, or as duration and magnitude (see the example below)

- ``filename()`` a string pointing to a CSV file that specifies the following columns: ``Time [yr]``, ``Rate/Scale [units]``, ``delta value [dimensionless]`` The class will attempt to convert the data into the correct model units. This process is however not very robust.

The default is to add the signal to a given connection. It is however also possible to use the signal data as a scaling factor. Signals are cumulative, i.e., complex signals are created by adding one signal to another (i.e., Snew = S1 + S2). Using the P-cycle model from the previous chapter (see ``po4_1.py`` in the examples directory) we can add a signal by first defining a signal instance, and then associating the instance with a weathering connection instance:

.. code:: ipython

    Signal(
        name="CR",  # Signal name
        species=M.PO4,  # Species
        start="3 Myrs",
        shape="pyramid",
        duration="1 Myrs",
        mass="45 Pmol",
        register=M,
    )

    Connection(
        source=M.weathering,  # source of flux
        sink=M.sb,  # target of flux
        rate=F_w,  # rate of flux
        id="river",  # connection id
        signal=M.CR,
    )

This will result in the following output:

.. _sig:

.. figure:: ./po4_1_with_signal.png
    :width: 300


    Example output for the ``CR`` signal above. See ``po4_1_with_signal.py`` in the examples directory.

1.2 Working with multiple species
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic building blocks introduced so far, are sufficient to create a model, but not necessarily convenient when a model contains more than one species. ESBMTK addresses this through ``ReservoirGroup`` class, which allows to group of several ``Reservoir`` instances. A  ``ReservoirGroup`` shares common properties. e.g., the volume and name of a given box, as well as the connection properties. In other words, in a multi-species model, one does not have to specify connections for each species, rather, it is sufficient to specify the connection type for the  ``ReservoirGroup`` instance. Similarly, there are classes to group sources, sinks and connections.

Using the previous example of a simple P-cycle model, we now express the P-cycling as a function of photosynthetic organic matter (OM) production and remineralization. First, we import the new classes and we additionally load the species definitions for carbon.

.. code:: ipython

    from esbmtk import (
        ReservoirGroup,  # the reservoir class
        ConnectionGroup,  # the connection class
        SourceGroup,  # the source class
        SinkGroup,  # sink class
    )
    M = Model(
        stop="6 Myr",  # end time of model
        timestep="1 kyr",  # upper limit of time step
        element=["Phosphor", "Carbon"],  # list of species definitions
    )

Setting up a group source, is similar to a single Source, except that we now specify a species list:

.. code:: ipython

    SourceGroup(
        name="weathering",
        species=[M.PO4, M.DIC],
        register=M,  # i.e., the instance will be available as M.weathering
    )

Defining a ``Reservoirgroup`` follows the same pattern, except that we use a dictionary so that we can specify the initial concentrations for each species as well:

.. code:: ipython

    ReservoirGroup(
        name="S_b",
        volume="3E16 m**3",  # surface box volume
        concentration={M.DIC: "0 umol/l", M.PO4: "0 umol/l"},
        register=M,
    )

The ``ConnectionGroup`` definition is equally straightforward, and the following expression will apply the thermohaline downwelling to all species in the ``M.S_b`` group.

.. code:: ipython

    ConnectionGroup(  # thermohaline downwelling
        source=M.S_b,  # source of flux
        sink=M.D_b,  # target of flux
        ctype="scale_with_concentration",
        scale=thc,
        id="downwelling_PO4",
    )

It is also possible, to specify individual rates or scales using a dictionary, as in this example that sets two different weathering fluxes:

.. code:: ipython

    ConnectionGroup(
        source=M.weathering,  # source of flux
        sink=M.S_b,  # target of flux
        rate={M.DIC: F_w_OM, M.PO4: F_w_PO4},  # rate of flux
        ctype="regular",
        id="river",  # connection id
    )

The following code defines primary production and its effects on DIC in the surface and deep box. The example is a bit contrived but demonstrates the principle. Note the use of the ``ref_reservoirs`` keyword and ``Redfield`` ratio

.. code:: ipython

    # Primary production as a function of P-concentration
    Connection(  #
        source=M.S_b.DIC,  # source of flux
        sink=M.D_b.DIC,  # target of flux
        ref_reservoirs=M.S_b.PO4,
        ctype="scale_with_concentration",
        scale=Redfield * M.S_b.volume / tau,
        id="OM_production",
    )

One can now proceed to define the particulate phosphate transport as a function of organic matter export

.. code:: ipython

    pl = data_summaries(
        M,  # model instance 
        [M.DIC, M.PO4],  # Species list 
        [M.S_b, M.D_b],  # ReservoirGroup list
        M,
    )
    M.plot(pl, fn="po4_2.png")

which results in the below plot. The full code is available in the examples directory as ``po4_2.py``

.. _po4_2:

.. figure:: ./po4_2.png
    :width: 300


    Output of ``po4_2.py`` demonstrating the use of the ``data_summaries()`` function

1.3 Adding isotopes
~~~~~~~~~~~~~~~~~~~

1.4 Using many boxes
~~~~~~~~~~~~~~~~~~~~