def my_burial(concentration: float, p: tuple) -> float:
    """Calculate a flux as a function of concentration

    Parameters
    ----------
    concentration : float
        substance concentration
    p : tuple
        tuple where the first element is the scaling factor

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
        ftype="std",
        function_input_data=[source],
        function_params=p,
        register=source,
        return_values=[
            {f"F_{sink.full_name}.{species.name}": "id_string"},
        ],
    )

    register_return_values(ec, source)
