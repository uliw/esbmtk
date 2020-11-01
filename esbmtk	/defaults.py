from esbmtk import esbmtkBase

class Carbon(esbmtkBase):
    """ Some often used definitions
    
    """
    from esbmtk import Model
    from typing import Dict
    from nptyping import Float, NDArray
    
    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """

        from esbmtk import Model, Element, Species
        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "model": Model,
            "name":str,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["model","name"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({"model"  : "a model handle",})
        self.__validateandregister__(kwargs)  # initialize keyword values

        eh = Element(
             name="C",                  # Element Name
             model=self.model,          # Model handle
             mass_unit="mmol",          # base mass unit
             li_label="C^{12$S",        # Name of light isotope
             hi_label="C^{13}$S",       # Name of heavy isotope
             d_label="$\delta^{13}$C",       # Name of isotope delta
             d_scale="VPDB",            # Isotope scale. End of plot labels
             r=0.0112372,  # VPDB C13/C12 ratio https://www-pub.iaea.org/MTCD/publications/PDF/te_825_prn.pdf
        )

        # add species
        Species(name="CO2", element=eh)  # Name & element handle
        Species(name="DIC", element=eh)
        Species(name="OM", element=eh)
        Species(name="CaCO3", element=eh)
