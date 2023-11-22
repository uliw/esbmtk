=============
ESBMTK Manual
=============




1 Implementing more complex models
----------------------------------

1.1 Adding model forcing
~~~~~~~~~~~~~~~~~~~~~~~~

ESBMTK realizes model forcing through the ``Signal`` class. Once defined, a signal instance can be associated with a ``Connection`` instance and will then act on the associated connection.
This class provides the following methods to create a signal:

- ``square()``, ``pyramid()``, ``bell()``  These are defined by specifying the signal startime (relative to the model time), it's size (as mass) and duration, or as duration and magnitude (see the example below)

- ``filename()`` a string pointing to a CSV file that specifies the following columns: ``Time [yr]``, ``Rate/Scale [units]``, ``delta value [dimensionless]`` The class will attempt to convert the data into the correct model units. This process is however not very robust.

The default is to add the signal to a given connection. It is however also possible to use the signal data as a scaling factor. Signals are cumulative, i.e., complex signals ar created by adding one signal to another (i.e., Snew = S1 + S2). Using the P-cycle model from the previous chapter (see ``po4_1.py`` in the examples directory) we can add a signal by first defining a signal instance, and then associating the instance with weathering connection instance:

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

1.3 Adding isotopes
~~~~~~~~~~~~~~~~~~~

1.4 Using many boxes
~~~~~~~~~~~~~~~~~~~~
