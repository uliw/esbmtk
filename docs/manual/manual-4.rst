


Extending ESBMTK
----------------

The Element and Species Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESBMTK uses the :py:class:`esbmtk.esbmtk.Species()` and :py:class:`esbmtk.esbmtk.Element()` class primarily to control plot labeling. Each ``Species`` instance is a child of an ``Element`` instance. Within the model hierarchy,  one would access e.g., ``DIC`` as ``M.Carbon.DIC`` . However, this results in a lot of redundant code, so the ``Species`` instances are also registered with the ``Model`` instance.

.. code:: ipython

    from esbmtk import Model

    M = Model(stop="6 Myr", timestep="1 kyr", element=["Carbon", "Oxygen"])
    # Access using complete hirarchy
    print(M.Carbon.DIC)
    # Access using shorthand
    print(M.DIC)

The distinction between Element and Species exists to group information that is common to all species of a given element. The current entry for Oxygen reads, e.g., like this

.. code:: ipython

    def Oxygen(model: Model) -> None:
        """Common Properties of Oxygen

        Parameters
        ----------
        model : Model
            Model instance

        """
        eh = Element(
            name="Oxygen",
            model=model,  # model handle
            mass_unit="mol",  # base mass unit
            li_label="$^{16$}O",  # Name of light isotope
            hi_label="$^{18}$)",  # Name of heavy isotope
            d_label=r"$\delta^{18}$)",  # Name of isotope delta
            d_scale="mUr VSMOV",  # 
            r=2005.201e-6,  # https://nucleus.iaea.org/rpst/documents/vsmow_slap.pdf
            register=model,
        )

and the associated species definitions are:

.. code:: ipython

    Species(name="O", element=eh, display_as="O", register=eh)
    Species(name="O2", element=eh, display_as=r"O$_{2}$", register=eh)
    Species(name="OH", element=eh, display_as=r"OH$^{-}$", register=eh)

Note that the variable ``eh`` is used to associate the ``Species`` instance with the ``Element`` instance. Upon startup, ESBMTK loads all predefined species definitions for each element named in the ``element_list`` keyword and registers them with the model instance. See the file ``species_definitions.py`` in the source-code for the currently defined elements and species (`https://github.com/uliw/esbmtk/blob/master/src/esbmtk/species_definitions.py <https://github.com/uliw/esbmtk/blob/master/src/esbmtk/species_definitions.py>`_)

To see a list of all known species for a given element use the ``list_species`` method of the Element instance

.. code:: ipython

    M.Oxygen.list_species()

Modifying/Extending an existing Species/Element definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modifying and existing definition is done after the model has been loaded, but
before running the solver. The following two lines, show, e.g, how to change the
isotope scale of Oxygen from mUR to permil, and how to set the plot concentration unit of O2 to :math:`\mu` mol:

.. code:: ipython

    M.Oxygen.d_scale="\u2030"
    M.Oxygen.O2.scale_to="umol"

see the :py:class:`esbmtk.esbmtk.Species()` and :py:class:`esbmtk.esbmtk.Element()` definitions for a full list of implemented properties.

Adding custom Species definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new species follow the examples in the ``species_definitions.py`` source code file. Provided you loaded ``Oxygen`` in the model definition, defining a new species instance for dissolved oxygen would look like this

.. code:: ipython

    from esbmtk import Species
    Species(
        name="O2aq",
        element=M.Oxygen,
        display_as=r"[O$_{2}$]$_{aq}$",
    )
    M.O2aq = M.Oxygen.O2aq  # register shorthand with model
    print(M.O2aq)

Adding a new Element and its species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, I use Boron to demonstrate how to add a new element and its respective species. Note, however, that Boron is already part of ESBMTK, for this example it is simply not loaded.

.. code:: ipython

    from esbmtk import Model, Element, Species

    M = Model(stop="6 Myr", timestep="1 kyr")

    Element(
        name="Boron",
        model=M,  # model handle
        mass_unit="mmol",  # base mass unit
        li_label=r"$^{11$}B",  # Name of light isotope
        hi_label=r"$^{10$}B",  # Name of heavy isotope
        d_label=r"$\delta{11}B",  # Name of isotope delta
        d_scale="mUr SRM951",  # Isotope scale.
        r=0.26888,  # isotopic abundance ratio for species
        register=M,
    )

    Species(name="B", element=M.Boron, display_as="B")
    Species(name="BOH", element=M.Boron, display_as="BOH")
    Species(name="BOH3", element=M.Boron, display_as=r"B(OH)$_{3}$")
    Species(name="BOH4", element=M.Boron, display_as=r"B(OH)$_{4}^{-}$")

    # register the species shorthands with the model.
    for sp in M.Boron.lsp:
        setattr(M, sp.name, sp)

    # verify the sucess
    print(M.BOH3)

Note that in the above example, we leverage that ``Element`` instances keep track of their species in the ``lsp`` variable. Provided that none of the species was defined previously, we can thus simply loop over the list of species to register them with the model.

Adding custom functions to ESBMTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESBMTK has some rudimentary support to add custom functions. This is currently not very user-friendly, and a better interface may become available in the future.
Adding a custom function to ESBMTK requires the following considerations:

- ESBMTK must be able to import the function so that it can be used in the equation system

- ESBMTK must have a way to assign the correct input & output variables to the function call

- Since we only declare a function and not a complete connection object, it is up to the user code to make sure that function parameters like scale factors (see below) are in the correct units, and of type ``Number`` (rather than string or quantity). Likewise, it is up to the user-provided code to ensure that the returned values have the correct sign.

- The function signature of any custom function must adhere to a format, where the first argument(s) are of type float, and the second argument is a tuple (which can be empty):

.. code:: ipython

    def custom(c0:float, t: tuple)  # valid
    def custom(c0:float, c1:float, t: tuple) # valid
    def custom(c0:float, c1:int, t: tuple) # invalid

The reason behind this rigid scheme has to do with memory management, but it is typically easy to adhere to them.

A worked example
^^^^^^^^^^^^^^^^

Let's consider a simple case where we define a custom function ``my_burial()`` that returns a flux as a function of concentration. For this, we need a parameter that passes a concentration, and a parameter that passes a scaling factor. Since both are float, we could use this signature with an empty tuple

.. code:: ipython

    def my_burial(concentration: float, scale: float, t: tuple) -> float:

However, to demonstrate the use of a tuple to pass one or more parameters, I will pass the scaling factor as a tuple in the below example:

.. code:: ipython

    def my_burial(concentration: float, p: tuple) -> float:
        """Calculate a flux as a function of concentration

        Parameters
        ----------
        concentration : float
            substance concentration
        p : tuple
            where the first element is the scaling factor

        Returns
        -------
        float
            flux in model mass unit / time

        Notes: the scale information is passed as a tuple, so we need
        extract it from the tuple before using it

        f is a burial flux, so we need to return a negative number.
        """
        (scale,) = p

        f = concentration * scale

        return -f

ESBMTK needs to import this function into the code that builds the equation system, so this requires that we place this function into a module file (e.g., ``my_functions.py``), and that we register this file and any custom functions with the model code. ESBMTK provides the ``register_user_function()`` function which is used like this

.. code:: ipython

    register_user_function(M, "my_functions", "my_burial")

Note that the last argument can also be a list of function names.

Next, we need to create code that maps the model variables required by ``my_burial()`` to the actual function call. Most of this work is done by the :py:class:`esbmtk.extended_classes.ExternalCode()` class. In the following example, we wrap this task into a dedicated function, but this is not a hard requirement. I add this function to the ``my_functions.py`` file, but you can also keep it with the code that defines the model.  Since we want to use this function to calculate a flux between two reservoirs (or a sink/source), we need to pass the source and sink reservoirs, as well as the species and the scale information, to ``add_my_burial()``.

Notes on the below code:

- If ``my_buria()`` is defined in the same file as ``add_my_burial()`` there is no need to import ``my_burial()``

- The ``function_input_data`` keyword requires the ``Reservoir`` instance, not the array with the concentration values (i.e., ``Reservoir.c``). More than one argument can be given.

- The ``return_values`` keyword expects a dictionary. If the return value is a flux, the dictionary key must be preceded by ``F_``. The key format must be ``{Reservoir.full_name}.{Species.name}``. The ``id_string`` must be unique within the model, and must not contain blanks or dots. If the return value is a Reservoir, the dictionary entry reads like this  ``{f"R_{rg.full_name}.Hplus": rg.swc.hplus},`` where dictionary value is used to set the initial condition.

- In the last step, the ``register_return_values`` parses the return value dictionary and creates the necessary :py:class:`esbmtk.esbmtk.Flux()` or :py:class:`esbmtk.esbmtk.Reservoir()` instances. This step may move to the init-section of the :py:class:`esbmtk.extended_classes.ExternalCode()` class definition in a future version.

.. code:: ipython

    def add_my_burial(source, sink, species, scale) -> None:
        """This function initializes a user supplied function
        so that it can be used within the ESBMTK eco-system

        Parameters
        ----------
        source : Source | Reservoir | ReservoirGroup
            A source
        sink : Sink | Reservoir | ReservoirGroup
            A sink
        species : Species
            A model species
        scale : float
            A scaling factor

        """
        from esbmtk import ExternalCode, register_return_values

        p = (scale,)  # convert float into tuple
        ec = ExternalCode(
            name="mb",
            species=source.species,
            function=my_burial,
            fname="my_burial",
            function_input_data=[source],
            function_params=p,
            register=source,
            return_values=[
                {f"F_{sink.full_name}.{species.name}": "id_string"},
            ],
        )

        register_return_values(ec, source)

Once these functions are defined, we can use them in the model definition as follows

.. code:: ipython

    # register the new module and function with the model
    register_user_function(M, "my_functions", "my_burial")

    # import the add_my_burial into this script file
    from my_functions import add_my_burial

    # add the my_burial_function to the model objects.
    add_my_burial(
        M.D_b,  # Source
        M.burial,  # Sink
        M.PO4,  # Species
        M.D_b.volume.magnitude / 2000.0,  # Scale
    )

Note that  ``M.D_b.volume.magnitude`` is not a number but a quantity. So one needs to query the numerical value with ``.magnitude``  or add code to  ``add_my_burial`` to query the type of the input arguments and convert as necessary.

The file ``user_defined_functions.py`` in the ``examples`` directory shows a working example. 

Debugging custom function integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current custom function integration interface is not very user-friendly and often requires investigating the actual ``equations.py`` file. In the default operating mode, ESBMTK will recreate this file for each model run, so that print statements and breakpoints that have been placed in ``equations.py`` have no effect.
Use the ``parse_model`` keyword in the model instance to keep the edited ``equations.py`` for the next run:

.. code:: ipython

    M = Model(
        stop="1000 yr",  # end time of model
        timestep="1 yr",  # upper limit of time step
        element=["Phosphor"],  # list of element definitions
        parse_model=False,  # do not overwrite equations.py
    )
