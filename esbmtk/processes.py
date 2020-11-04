"""
     esbmtk: A general purpose Earth Science box model toolkit
     Copyright (C), 2020 Ulrich G. Wortmann

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

class Process(esbmtkBase):
    """This class defines template for process which acts on one or more
     reservoir flux combinations. To use it, you need to create an
     subclass which defines the actual process implementation in their
     call method. See 'PassiveFlux as example'
    """

    
    def __init__(self, **kwargs :Dict[str, any]) -> None:
        """
          Create a new process object with a given process type and options
          """

        self.__defaultnames__()      # default kwargs names
        self.__initerrormessages__() # default error messages
        self.bem.update({"rate": "a string"})
        self.__validateandregister__(kwargs)  # initialize keyword values
        self.__postinit__()          # do some housekeeping

    def __postinit__(self) -> None:
        """ Do some housekeeping for the process class
          """

        # legacy name aliases
        self.n: str = self.name  # display name of species
        self.r: Reservoir = self.reservoir
        self.f: Flux = self.flux
        self.m: Model = self.r.sp.mo  # the model handle

        # Create a list of fluxes wich texclude the flux this process
        # will be acting upon
        self.fws :List[Flux] = self.r.lof.copy()
        self.fws.remove(self.f)  # remove this handle

        self.rm0 :float = self.r.m[0]  # the initial reservoir mass
        self.direction :Dict[Flux,int] = self.r.lio[self.f.n]
        

    def __defaultnames__(self) -> None:
        """Set up the default names and dicts for the process class. This
          allows us to extend these values without modifying the entire init process"""


        # provide a dict of known keywords and types
        self.lkk: Dict[str, any] = {
            "name": str,
            "reservoir": Reservoir,
            "flux": Flux,
            "rate": Number,
            "delta": Number,
            "lt": Flux,
            "alpha": Number,
            "scale": Number,
            "ref": Flux,
        }

        # provide a list of absolutely required keywords
        self.lrk: list[str] = ["name"]

        # list of default values if none provided
        self.lod: Dict[str, any] = {}

        # default type hints
        self.scale :t
        self.delta :Number
        self.alpha :Number
        

    def register(self, reservoir :Reservoir, flux :Flux) -> None:
        """Register the flux/reservoir pair we are acting upon, and register
          the process with the reservoir
          """

        # register the reservoir flux combination we are acting on
        self.f :Flux = flux
        self.r :Reservoir = reservoir
        # add this process to the list of processes acting on this reservoir
        reservoir.lop.append(self)
        flux.lop.append(self)

    def describe(self) -> None:
        """Print basic data about this process """
        print(f"\t\tProcess: {self.n}", end="")
        for key, value in self.kwargs.items():
            print(f", {key} = {value}", end="")

        print("")

    def show_figure(self, x, y) -> None:
        """ Apply the current process to the vector x, and show the result as y.
          The resulting figure will be automatically saved.

          Example:
               process_name.show_figure(x,y)
          """
        pass
