import numpy as np
import pytest

from esbmtk import (
    Model,
    Signal,
)


class TestSignal:
    case = "data"  # forward | backwards
    oxidation_flux = "pyrite_oxidation.csv"
    burial_flux = "pyrite_burial.csv"
    M = Model(
        stop="200 kyr",  # end time of model
        max_timestep="10 yr",  # upper limit of time step
        element=["Carbon"],  # list of species definitions
    )

    M.PW_DIC = Signal(  # Pyrite weathering
        name="PW_DIC",
        species=M.DIC,
        filename=f"{case}/{oxidation_flux}",
        scale=4,
        stype="addition",
        register=M,
    )

    M.PB_DIC = Signal(  # Pyrite burial
        name="PB_DIC",
        species=M.DIC,
        filename=f"{case}/{burial_flux}",
        scale=4,
        stype="addition",
        register=M,
    )

    M.PBW_DIC = Signal(
        name="PBW_DIC",
        species=M.DIC,
        filename=f"{case}/koelling_pyrite_data_200kyr.csv",
        scale=4,
        stype="addition",
        register=M,
    )

    def test_one(self):
        """Test signal addition by checking their
        interpolated values
        """
        self.M.new_sig = self.M.PW_DIC + self.M.PB_DIC
        x = self.M.PB_DIC.m * 0
        y = x
        for i, t in enumerate(self.M.time):
            x[i] = self.M.new_sig(t)[0]
            y[i] = self.M.PBW_DIC(t)[0]

        assert np.sum(np.abs(x) - np.abs(y)) < 0.01

    def test_two(self):
        """Test manual addition of signal values by testing
        their data directly"""
        self.M.new_sig = self.M.PW_DIC + self.M.PB_DIC
        a = self.M.PBW_DIC.m
        b = self.M.PW_DIC.m + self.M.PB_DIC.m
        assert np.sum(np.abs(a) - np.abs(b)) < 0.01
