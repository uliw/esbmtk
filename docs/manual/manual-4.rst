


1 Extending ESBMTK
------------------

1.1 The Element and Species Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESBMTK uses the :py:class:`esbmtk.esbmtk.Species()` and :py:class:`esbmtk.esbmtk.Element()` class primarily to control plot labeling. Each ``Species`` instance is a child of an ``Element`` instance. Within the model hierarchy,  one would access e.g., ``DIC`` as ``M.Carbon.DIC`` . However this results in a lot of redundant code, so the ``Species`` instances are also registered with the ``Model`` instance.

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

Note that the variable ``eh`` is used to associate the ``Species`` instance with the ``Element`` instance. Upon startup, ESBMTK loads all predefined species definition for each element named in the ``element_list`` keyword, registers them with the model instance. See the file ``species_definitions.py`` in the source-code for the currently defined elements and species (`https://github.com/uliw/esbmtk/blob/staging/src/esbmtk/species_definitions.py <https://github.com/uliw/esbmtk/blob/staging/src/esbmtk/species_definitions.py>`_)

To see a list of all known species for a given element use the ``list_species`` method of the Element instance

.. code:: ipython

    M.Oxygen.list_species()

1.1.1 Modifying/Extending an existing Species/Element definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modifying and existing definition is done after the model has been loaded, but
before running the solver. The following two lines, show, e.g, how to change the
isotope scale of Oxygen from mUR to permil, and how to set the plot concentration unit of O2 to :math:`\mu` mol:

.. code:: ipython

    M.Oxygen.d_scale="\u2030"
    M.Oxygen.O2.scale_to="umol"

see the :py:class:`esbmtk.esbmtk.Species()` and :py:class:`esbmtk.esbmtk.Element()` definitions for a full list of implemented properties.

1.1.2 Adding your own Species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

1.1.3 Adding a new Element and its species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example I use Boron to demonstrate how to add a new element and its respective species. Note however that Boron is already part of ESBMTK, for this example it is simply not loaded.

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

1.1.4 Adding your own functions to ESBMTK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.1.5 Calling model specific functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
