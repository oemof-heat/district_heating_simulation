from collections import OrderedDict
import itertools as it
import os.path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


lamb_func = (
    lambda eps, D, Re: 1.325 / (np.log(eps / (3.7 * D) + 5.74 / (Re ** 0.9))) ** 2
)


def single_pipe(args):
    r"""
    Calculate hydraulic and thermal properties of a single feed and return
    district heating pipe.
    """
    Q_cons, DT_drop, DT_prod_in, k, L, D, c, rho, eps, mu = args
    Q_cons *= 1e6  # MW to W
    eta_pump = 0.7
    pressure_loss_cons = 1e5

    # hydraulic part
    # mass flow is determined by consumer mass flow
    m = Q_cons * 1 / (c * DT_drop)

    # pressure loss
    v = 4 * m / (rho * np.pi * D ** 2)
    Re = D * v * rho / mu
    lamb = lamb_func(eps, D, Re)

    pressure_loss_dis = lamb * 8 * L * 1 / (rho * np.pi ** 2 * D ** 5) * m ** 2
    Dp_pump = 2 * pressure_loss_dis + pressure_loss_cons
    P_pump = Dp_pump * m * 1 / rho

    # heat losses in feedin and return pipe
    exponent = k * np.pi * L * D * 1 / (c * m)
    DT_cons_in = DT_prod_in * np.exp(-1 * exponent)
    DT_prod_r = (DT_cons_in - DT_drop) * np.exp(-1 * exponent)

    Q_prod = c * m * (DT_prod_in - DT_prod_r)
    Q_loss = Q_prod - Q_cons
    perc_loss = 100 * Q_loss * 1 / Q_prod

    # Change units
    Dp_pump_bar = 1e-5 * Dp_pump
    P_pump_kW = 1e-3 * 1 / eta_pump * P_pump
    Q_loss_MW = 1e-6 * Q_loss

    if DT_prod_r > 0:
        return np.array(
            [
                Q_prod,
                DT_cons_in,
                DT_prod_r,
                v,
                Dp_pump_bar,
                P_pump_kW,
                Q_loss_MW,
                perc_loss,
            ]
        )
    else:
        return None


def generic_sampling(input_dict, results_dict, function):
    r"""
    n-dimensional full sampling, storing as xarray.

    Parameters
    ----------
    input_dict : OrderedDict
        Ordered dictionary containing the ranges of the
        dimensions.

    results_dict : OrderedDict
        Ordered dictionary containing the dimensions and
        coordinates of the results of the function.

    function : function
        Function to be sampled.

    Returns
    -------
    results : xarray.DataArray

    sampling : np.array

    indices :
    """
    join_dicts = OrderedDict(list(input_dict.items()) + list(results_dict.items()))
    dims = join_dicts.keys()
    coords = join_dicts.values()
    results = xr.DataArray(
        np.empty([len(v) for v in join_dicts.values()]), dims=dims, coords=coords
    )

    sampling = np.array(list(it.product(*input_dict.values())))
    indices = np.array(
        list(it.product(*[np.arange(len(v)) for v in input_dict.values()]))
    )

    for i in range(len(sampling)):
        result = function(sampling[i])
        results[tuple(indices[i])] = result

    return results, sampling, indices


input_dict = OrderedDict(
    [
        ('Q_cons', np.arange(1, 6, 2)),
        ('DT_drop', [10]),
        ('DT_prod_in', [70, 90, 110]),
        ('k', np.arange(1, 3, 1)),
        ('L', [1000]),
        ('D', [0.25, 0.3, 0.4]),
        ('c', [4230]),
        ('rho', [951]),
        ('eps', [0.01e-3]),
        ('mu', [0.255e-3]),
    ]
)

result_dict = OrderedDict(
    [
        (
            'results',
            [
                'Q_prod',
                'DT_cons_in',
                'DT_prod_r',
                'v [m/s]',
                'pressure_loss [bar]',
                'P_pump [kW]',
                'Q_loss [MW]',
                'loss [%]',
            ],
        )
    ]
)


def plot_data():
    r"""

    """
    fig, axs = plt.subplots(5, 3, figsize=(9, 12))

    coords = ['D', 'DT_prod_in', 'k']
    ylim = [(0, 3), (0, 5), (0, 100), (0.1, 0.5), (0, 40)]
    colors = [
        sns.color_palette("hls", len(sam_results[coord])).as_hex() for coord in coords
    ]
    labels = [
        [f'{coord}={D}' for D in sam_results.coords[coord].values] for coord in coords
    ]
    titles = [
        'Flow velocity',
        'Pressure loss $\Delta P_{pump}$',
        'Pump power',
        'Heat loss',
        'Relative heat loss',
    ]

    for i, c in enumerate(
        ['v [m/s]', 'pressure_loss [bar]', 'P_pump [kW]', 'Q_loss [MW]', 'loss [%]']
    ):

        for ii, vv in enumerate(sam_results.coords['D'].values):
            sam_results.squeeze().isel(k=1, DT_prod_in=-1).sel(results=c, D=vv).plot(
                ax=axs[i, 0], marker='.', markersize=10, color=colors[0][ii]
            )

            axs[i, 0].set_ylim(ylim[i])
            axs[i, 0].set_xlabel('Q_cons [MW]')
            axs[i, 0].set_ylabel(c)
            axs[i, 0].set_title('', size='15')
        axs[0, 0].set_title('Column A \n', size='15')
        axs[-1, 0].set_xlabel('Q_cons [MW] \n\n DT_prod_in=110°C \n k=1.5 W/(m²K)')

        for ii, vv in enumerate(sam_results.coords['DT_prod_in'].values):
            sam_results.squeeze().isel(k=1, DT_prod_in=ii).sel(results=c, D=0.25).plot(
                ax=axs[i, 1], marker='.', markersize=10, color=colors[1][ii]
            )

            axs[i, 1].set_ylim(ylim[i])
            axs[i, 1].set_xlabel('Q_cons [MW]')
            axs[i, 1].set_title(titles[i], size='15')
        axs[0, 1].set_title('Column B \n' + titles[0], size='15')
        axs[-1, 1].set_xlabel('Q_cons [MW] \n\n D=0.25 \n k=1.5 W/(m²K)')

        for ii, vv in enumerate(sam_results.coords['k'].values):
            sam_results.squeeze().isel(k=ii, DT_prod_in=-1).sel(results=c, D=0.25).plot(
                ax=axs[i, 2], marker='.', markersize=10, color=colors[2][ii]
            )

            axs[i, 2].set_ylim(ylim[i])
            axs[i, 2].set_xlabel('Q_cons [MW]')
            axs[i, 2].set_title('', size='15')
        axs[0, 2].set_title('Column C \n', size='15')
        axs[-1, 2].set_xlabel('Q_cons [MW] \n\n D=0.25 m \n DT_prod_in=110°C')

    plt.subplots_adjust(hspace=0.7)

    for i, coord in enumerate(coords):
        axs[-1, i].legend(
            axs[-1, i],
            labels=labels[i],
            loc='lower center',
            bbox_to_anchor=(0.5, -1.8),
            ncol=1,
        )

    # plt.tight_layout()
    fig.savefig('single_pipe_calculations.pdf', bbox_inches="tight")


if os.path.isfile('single_pipe_calculations.nc'):
    print('File exists')
    sam_results = xr.open_dataarray('single_pipe_calculations.nc')

else:
    print('Calculate')
    sam_results = generic_sampling(input_dict, result_dict, single_pipe)[0]
    sam_results.to_netcdf('single_pipe_calculations.nc')

plot_data()
