"""Test the signal_from_csv output.

Here we test following combinations
|Name | Signal time reverse | plot time reverse |
| M0  | False               | False            |
| M1  | True                | False            |
| M2  | False               | True             |
| M3  | True                | True             |
"""

import pytest
from signal_from_csv_reverse_test import basemodel
from esbmtk import Signal

M = basemodel()  # get model handle
M0 = basemodel()  # get model handle
M1 = basemodel()  # get model handle
M2 = basemodel()  # get model handle
M3 = basemodel()  # get model handle


def run_model(M, sr, pr, i):
    Signal(
        name="CR",  # Signal name
        species=M.DIC,  # SpeciesProperties
        filename="signal_from_csv_data_reverse.csv",
        reverse_time=sr,
    )
    M.Conn_weathering_to_S_b_DIC_river.signal = M.CR
    M.run()
    plt, fig, axs = M.plot(
        [M.CR, M.S_b.DIC, M.D_b.DIC],
        reverse_time=pr,
        no_show=True,
        fn=f"M{i}_{sr}_{pr}.pdf",
    )
    # M.plot(
    #     [M.CR, M.S_b.DIC, M.D_b.DIC],
    #     reverse_time=pr,
    #     fn=f"M{i}_{sr}_{pr}.pdf",
    # )
    # return 1, 1, 1
    return plt, fig, axs


models = [M0, M1, M2, M3]
sr = [False, True, False, True]
pr = [False, False, True, True]

plt = []
fig = []
axs = []

for i, M in enumerate(models):
    p, f, a = run_model(M, sr[i], pr[i], i)
    plt.append(p)
    fig.append(f)
    axs.append(a)

coords = []
for ax in axs:
    for line in ax[0].get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # Combine as coordinate pairs
        coords.append(list(zip(xdata, ydata)))

# at 250
test_values = [  # result, reference value
    (M0.D_b.DIC.c[839] * 1e6, 22.111),
    (M1.D_b.DIC.c[839] * 1e6, 50.500),
    (M2.D_b.DIC.c[839] * 1e6, 22.111),
    (M3.D_b.DIC.c[839] * 1e6, 50.500),
    (coords[0][250][1], 88920000000),  # forward only
    (coords[1][250][1], 0),  # signal reversed, plot forward
    (coords[2][250][1], 88920000000),  # signal forward, plot reversed
    (coords[3][250][1], 0),  # signal forward, plot reversed
]


# run tests
@pytest.mark.parametrize("test_input, expected", test_values)
def test_values(test_input, expected):
    t = 1e-1  # +- 1 mu mol is good enough
    assert abs(expected) * (1 - t) <= abs(test_input) <= abs(expected) * (1 + t)
