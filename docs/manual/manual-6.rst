


Debugging and Tips & Tricks
---------------------------

Key spelling errors
~~~~~~~~~~~~~~~~~~~

ESBMTK does some rudimentary error checking on the key-value pairs that are used to initialize the ESBMTK objects. It will catch typos if a key is misspelled, e.g., in the following example, we use ``elements`` instead of ``element``, and this will result in stack trace where the first entry points to the location in the model code where the error occurs,, and where the last line indicates that ``elements`` is not a valid keyword

.. code:: ipython

    from esbmtk import Model

    M = Model(
        stop="6 Myr",
        timestep="1 kyr",
        elements=["Carbon", "Oxygen"])

::

    ---------------------------------------------------------------------------
    KeywordError                              Traceback (most recent call last)
    Cell In[4], line 3
          1 from esbmtk import Model
    ----> 3 M = Model(
          4     stop="6 Myr",
          5     timestep="1 kyr",
          6     elements=["Carbon", "Oxygen"],
          7 )

    File ~/user/python-scripts/esbmtk/src/esbmtk/esbmtk.py:191, in Model.__init__(self, **kwargs)
        186 # provide a list of absolutely required keywords
        187 self.lrk: tp.List[str] = [
        188     "stop",
        189     ["max_timestep", "timestep"],
        190 ]
    --> 191 self.__initialize_keyword_variables__(kwargs)
        193 self.name = "M"
        195 # empty list which will hold all reservoir references

    File ~/user/python-scripts/esbmtk/src/esbmtk/esbmtk_base.py:94, in input_parsing.__initialize_keyword_variables__(self, kwargs)
         92 self.__check_mandatory_keywords__(self.lrk, kwargs)
         93 self.__register_variable_names__(self.defaults, kwargs)
    ---> 94 self.__update_dict_entries__(self.defaults, kwargs)
         95 self.update = True

    File ~/user/python-scripts/esbmtk/src/esbmtk/esbmtk_base.py:146, in input_parsing.__update_dict_entries__(self, defaults, kwargs)
        144 for key, value in kwargs.items():
        145     if key not in defaults:
    --> 146         raise KeywordError(f"{key} is not a valid keyword")
        148     if not isinstance(value, defaults[key][1]):
        149         raise InputError(
        150             f"{value} for {key} must be of type {defaults[key][1]}, not {type(value)}"
        151         )

    KeywordError: 

    elements is not a valid keyword

Value errors
~~~~~~~~~~~~

ESBMTK will test if the provided value is of the right type, (i.e., a number, or a string). So in the following example, it will raise 
an input error since ``time_step`` is given without a time unit.

.. code:: ipython

    from esbmtk import Model

    M = Model(
        stop="6 Myr",
        timestep=1,
        elements=["Carbon", "Oxygen"])

::

    ---------------------------------------------------------------------------
    InputError                                Traceback (most recent call last)
    Cell In[5], line 3
          1 from esbmtk import Model
    ----> 3 M = Model(
          4     stop="6 Myr",
          5     timestep=1,
          6     elements=["Carbon", "Oxygen"],
          7 )
    ...
    ...
    ...

    InputError: 

    1 for timestep must be of type (<class 'str'>, <class 'pint.Quantity'>), not <class 'int'>

Key-type errors
~~~~~~~~~~~~~~~

ESBMTK will check that absolutely  necessary keys are provided, e.g., omitting the ``timestep`` key will result in a ``Missingkeyworderror``.

.. code:: ipython

    from esbmtk import Model

    M = Model(
        stop="6 Myr",
        elements=["Carbon", "Oxygen"])

::

    ---------------------------------------------------------------------------
    MissingKeywordError                       Traceback (most recent call last)
    Cell In[7], line 3
          1 from esbmtk import Model
    ----> 3 M = Model(
          4     stop="6 Myr",
          5     elements=["Carbon", "Oxygen"],
          6 )
    ...
    ...
    ...

    MissingKeywordError: 

    timestep is a mandatory keyword

However, ESBMTK classes like  ``Connectionproperties`` accept a large range of keywords, and presently ESBMTK has no mechanism to test if all of these are suitable to the given connection or not. A typical mistake is shown in the following example that defines a connection where the flux is based on the concentration in the source reservoir. 

.. code:: ipython

    Species2Species(  # Surface to deep box
        source=M.L_b.PO4,
        sink=M.D_b.PO4,
        ctype="scale_with_concentration",
        scale=1,
        id="productivity")

It is a common mistake to replace the ``scale`` keyword with the ``rate`` keyword. This error will not be caught by ESBMTK since ``rate`` is valid keyword for other connection types. This can result in difficult to track errors. 

Using introspection
~~~~~~~~~~~~~~~~~~~

All ESBMTK objects maintain state, it is thus possible to inspect them. If we create e.g., the following model 

.. code:: ipython

    from esbmtk import Model, Reservoir

    M = Model(
        stop="6 Myr",
        timestep="1 kyr",
        element=["Carbon"])

    Reservoir(
        name="S_b",  # box name
        volume="3E16 m**3",  # surface box volume
        concentration={M.DIC: "2200 umol/l"},  # initial concentration
    )



we can query the parameters that we used to create the Reservoir instance by printing the model instance.

.. code:: ipython

    print(M)

::

    M (Model)
      stop = 6 Myr
      timestep = 1 kyr
      element = ['Carbon']


Since ESBMTK follows a hierarchical structure we can query the element properties for Phosphor like this

.. code:: ipython

    print(M.Carbon)

::

    Carbon (ElementProperties)
      mass_unit = mol
      li_label = C^{12}$S
      hi_label = C^{13}$S
      d_label = $\delta^{13}$C
      d_scale = mUr VPDB
      r = 0.0112372
      reference = https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf
      full_name = M.Carbon


and the DIC reservoir in the surface box as

.. code:: ipython

    print(M.S_b.DIC)

::

    DIC (Species)
      delta = None
      concentration = 2200 umol/l
      isotopes = False
      plot = None
      volume = 2.9999999999999996e+19 liter
      groupname = S_b
      rtype = regular
      full_name = M.S_b.DIC


to get a list of species that are defined for ``M.Carbon`` , we can use the ``vars()`` function (this results in a long list, so it is not shown here)

.. code:: ipython

    vars(M.Carbon)

Accessing ESBMTK objects that are created implicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspecting objects that are explicitly defined is straightforward, however, connection objects like:

.. code:: ipython

    Species2Species(  # Surface to deep box
        source=M.L_b.PO4,
        sink=M.D_b.PO4,
        ctype="scale_with_concentration",
        scale=1,
        id="productivity")

do not have an obvious name. The same is true for Flux instances. I.e., if we want scale the flux of organic bound phosphor as a function of the primary production of organic matter (OM), we need to know how the reference the OM flux. ESBMTK provides a lookup function for fluxes

.. code:: ipython

    M.flux_summary(
        filter_by="productivity",
        exclude="H_b", # optional
        return_list=True, # optional
    )

as well as for connections:

.. code:: ipython

    M.flux_summary(
        filter_by="productivity",
        return_list=True, # optional
    )

The returned objects can then be inspected as usual.

Inspecting the model equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under normal circumstances the model equations are transient and created on the fly. It is however possible to create a permanent version and write the equations to the file ``equations.py`` . To enable this feature, on has to set the ``debug_equations_file`` key

.. code:: ipython

    M = Model(
        stop="6 Myr",
        timestep="1 kyr",
        elements=["Carbon", "Oxygen"],
        debug_equations_file=True,
    )

Running the model will now create ``equations.py`` in the respective project directory. Subsequent runs will query whether to re-use the equations file from the previous run, or to create a new one. Re-using the old file is particularly useful when creating your own extensions, as it allows to edit the equations file manually (i.e, to set breakpoints, or add print statement to trace the solver etc.)
