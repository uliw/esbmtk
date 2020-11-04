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

class SomeClass(esbmtkBase):
    """
    A ESBMTK class template
    
    Example:
             ShelfArea(name = "Name"         # handle
                       data = s-handle       # data object handle
                       shelf_break = -250    # optional, defaults to -250
                       ocean_area  = 3.61E14 # optional, defaults to 3.61E14 m^2)

    The instance provides the following data
    
    Name.sa    = shelf area as function of sealevel and time in m^2
    Name.sap   = shelf area in percent relative to z=0
    Name.sbz0  = open ocean area up to the shelf break
    Name.sa0   = shelf area for z=0

    Methods:

    Name.plot() will plot the current shelf area data
    Name.plotvsl() will plot the shelf area versus the sealevel data 
"""

    def __init__(self, **kwargs: Dict[str, any]) -> None:
        """ Initialize this instance """

        # dict of all known keywords and their type
        self.lkk: Dict[str, any] = {
            "name": str,
            "data": ExternalData,
            "shelf_break" :int,
            "ocean_area": float,
        }

        # provide a list of absolutely required keywords
        self.lrk: list = ["name", "data"]
        # list of default values if none provided
        self.lod: Dict[str, any] = {"shelf_break":-250,
                                    "ocean_area": 3.61E14}

        # provide a dictionary entry for a keyword specific error message
        # see esbmtkBase.__initerrormessages__()
        self.__initerrormessages__()
        self.bem.update({"ocean_area"  : "a number",
                         "shelf_break" : "an integer number"})
        
        self.__validateandregister__(kwargs)  # initialize keyword values
