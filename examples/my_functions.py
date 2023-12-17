def my_burial(concentration: float, scale: float) -> float:
    """Calculate a flux as a function of concentration

    Parameters
    ----------
    concentration : float
        substance concentration
    scale : float
        scaling factor

    Returns
    -------
    float
        flux

    """
    f = concentration * scale

def init_carbonate_system_1(source, sink, species, scale)-> None:
    """TBD

    Parameters
    ----------
    r : Reservoir
        Reservoir
    
    """
    
    from esbmtk import ExternalCode

    p = (scale,)  # must be a tuple!
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
            {f"F_{sink.full_name}": "any string"},
        ],
    )
    # Add to the list of functions to be imported in ode backend
    source.mo.lpc_i.append(ec.fname)  

def add_my_burial(source, sink, species, scale)-> None:
    """ TBD """

    init_carbonate_system_1(source, sink, species, scale)
    
