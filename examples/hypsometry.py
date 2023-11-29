import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pl
from esbmtk import (
    Model,  # the model class
    Reservoir,
    ExternalData,
    DataField,
)

# define the basic model parameters
M = Model(
    stop="6 Myr",  # end time of model
    timestep="1 kyr",  # upper limit of time step
    element=["Carbon"],  # list of element definitions
    concentration_unit="mol/kg",
)

Reservoir(
    name="S_b",  # Name of reservoir group
    geometry=[-200, -800, 1],  # upper, lower, fraction
    concentration="1 mmol/kg",
    species=M.DIC,
    register=M,
)
print(f"M.S_b.area = {M.S_b.area:.2e}")
print(f"M.S_b.sed_area = {M.S_b.sed_area:.2e}")
print(f"M.S_b.volume = {M.S_b.volume:.2e}\n")

# return the ocean area at a given depth in m**2
print(f"M.hyp.area(0) = {M.hyp.area(0):.2e}")

# return the area between 2 depth datums in m**2
print(f"M.hyp.area_dz(0, -200) = {M.hyp.area_dz(0, -200):.2e}")

# return the volume between 2 depth datums in m**3
print(f"M.hyp.volume(0,-200) = {M.hyp.volume(0,-200):.2e}")

# return the total surface area of earth in m**3
print(f"M.hyp.sa = {M.hyp.sa:.2e}")

# get figure of data
fn: str = "Hypsometric_Curve_05m.csv"  # file name
cwd: pl.Path = pl.Path.cwd()  # get the current working directory
fqfn: pl.Path = pl.Path(f"{cwd}/{fn}")  # fully qualified file name

if not fqfn.exists():  # check if file exist
    raise FileNotFoundError(f"Cannot find file {fqfn}")

df: pd.DataFrame = pd.read_csv(fqfn)  # read csv data

x = df.iloc[:, 0]
y = df.iloc[:, 1]

elevation = np.arange(1000, -6001, -1)
plt.style.use(["ggplot"])
fig, ax = plt.subplots()
ax.plot(y, x, color="C1", label="Data", linewidth=3)  # create a line plot
ax.plot(
    M.hyp.hypdata, elevation, color="C0", label="Spline", linewidth=1
)  # create a line plot
ax.set_xlabel("Surface Area [%]")
ax.set_ylabel("Depth [m]")
ax.axis([1, 0, -6000, 1000])
ticks = ax.get_xticks() * 100
ticks = ticks.astype(int)
ax.xaxis.set_major_formatter("{x%.0f}")
ax.set_xticklabels(ticks)
plt.legend()
plt.savefig("hyp.png")
plt.show()  # display figure
