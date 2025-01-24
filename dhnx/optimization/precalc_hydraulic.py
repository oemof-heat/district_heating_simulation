# -*- coding: utf-8

"""
This module is designed to hold functions for pre-calculation of hydraulic
parameters for the district heating network dimensioning.

The aim is to calculate the maximum heat transport capacity
of pipelines given a maximum pressure drop per meter, roughness of the pipes
inner surface, and an estimated delta T of the forward and return pipes.

The equations and values used for the calculation can be found here:
http://www.math-tech.at/Beispiele/upload/gra_Druckverlust_in_Rohrleitungen.PDF
https://www.schweizer-fn.de/stroemung/rauhigkeit/rauhigkeit.php

This file is part of project dhnx (). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location:

SPDX-License-Identifier: MIT
"""

import logging
import math

import numpy as np
from scipy.optimize import fsolve

try:
    from CoolProp.CoolProp import PropsSI

except ImportError:
    print("Need to install CoolProp to use the hydraulic "
          "pre-calculation module.")

logger = logging.getLogger(__name__)  # Create a logger for this module


def eq_smooth(x, R_e):
    r"""
    Calculation of the pressure drop of hydraulic smooth surfaces.
    (Prandtl & Karman)

    .. eq_smooth_equation:

    :math:`f(x) = x-2 \cdot log\Big(\frac{Re}{2,51x}\Big)`

    Parameters
    ----------
    x: numeric
        :math:`x`: function variable [-]
    R_e: numeric
        :math:`Re`: Reynolds number [-]

    Returns
    -------
    Equation : numeric

    """
    return x - 2 * np.log10(R_e / (x * 2.51))


def eq_transition(x, R_e, k, d_i):
    r"""
    Equation to be solved for the transition range
    between a smooth and rough pipe surface (Prandtl-Colebrook)

    .. eq_transition_equation

    :math:`f(x)=x+2\cdot log \big(  \frac{2,51x}{Re} \cdot \frac{k}{3,71d_i} \big)`

    Parameters
    ----------
    x : numeric
        :math:`x`: function variable () [-]

    R_e: numeric
        :math:`Re`: Reynolds number [-]

    k : numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    d_i : numeric
        :math:`d_i`: inner diameter [m]

    Returns
    -------
    Equation : numeric

    """

    return x + 2 * np.log10((2.51 * x) / R_e + k / (3.71 * d_i))


def calc_k_v(d_v, d):
    r"""
    Calcutlates the kinematic viscosity for given density and dynamic viscosity

    .. calc_k_v_equation:

    :math:`\nu = \frac{\eta}{\rho}`

    Parameters
    ----------
    d_v: numeric
        :math:`\eta`: dynamic viskosity [kg/(m*s)]

    d: numeric
        :math:`\rho`: density [kg/m³]

    Returns
    -------
    kinematic viscosity [m²/s] : numeric

    """
    return d_v / d


def calc_Re(v, d_i, k_v):
    r"""
    Calculates the Reynolds number for a given velocity, inner diameter
    and kinematic viscosity

    .. calc_Re_equation:

    :math:`\frac{v \cdot d_i}{\nu}`

    Parameters
    ----------
    v: numeric
        :math:`v`: flow velocity [m/s]

    d_i: numeric
        :math:`d_i`: inner pipe diameter [m]

    k_v: numeric
        :math:`\nu`: kinematic viscosity [m²/s]

    Returns
    -------
    Reynolds number [-]: numeric

    """
    return v * d_i / k_v


def calc_lambda_laminar(Re):
    r"""
    Calculates the Darcy friction factor for a given Reynolds number
    for a laminar flow

    .. calc_lam_lam_equation:

    :math:`\lambda=\frac{64}{Re}`

    Parameters
    ----------
    Re: numeric
        :math:`Re`: Reynolds number [-]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    return 64 / Re


def calc_d_p(lam, length, d_i, d, v):
    r"""
    Calculates the pressure drop in a pipe for a given Darcy friction factor

    .. calc_d_p_equation:

    :math:`\Delta p = \lambda \frac{l}{d_i} \frac{\rho}{2} v^2`

    Parameters
    ----------
    lam: numeric
        :math:`\lambda`: Darcy friction factor [-]

    length: numeric
        :math:`l`: length of the pipe [m]

    d_i : numeric
        :math:`d_i`: inner pipe diameter [m]

    d: numeric
        :math:`\rho`: density [kg/m³]

    v: numeric
        :math:`v`: flow velocity [m/s]

    Returns
    -------
    Pressure drop [Pa]: numeric
    """
    return lam * length / d_i * d / 2 * v ** 2


def calc_lambda_turb1(Re):
    r"""
    Calculates the Darcy friction factor for a given Reynolds number
    for a turbulent flow, a smooth pipe and a Reynolds number smaller
    than 10^5 (Blasius)

    .. calc_lam_turb1_equation

    :math:`\lambda = 0,3164\cdot Re ^{-0,25}`

    Parameters
    ----------
    Re: numeric
        :math:`Re`: Reynolds number [-]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    return 0.3164 * Re ** (-0.25)


def calc_lambda_turb2(Re):
    r"""
    Calculates the Darcy friction factor for a given Reynolds number
    for a turbulent flow, a smooth pipe and a Reynolds number
    between 10^5 and 10^6 (Nikuradse)

    .. calc_lam_turb1_equation

    :math:`\lambda = 0,0032 + 0,221 \cdot Re ^{-0,237}`

    Parameters
    ----------
    Re: numeric
        :math:`Re`: Reynolds number [-]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    return 0.0032 + 0.221 * Re ** (-0.237)


def calc_lambda_turb3(Re):
    r"""
    Calculates the Darcy friction factor for a given Reynolds number
    for a turbulent flow, a smooth pipe and a Reynolds number higher than 10^6.
    For a formula, see :func:`~precalc_hydraulic.eq_smooth`.

    Parameters
    ----------
    Re: numeric
        :math:`Re`: Reynolds number [-]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    lam_init = 0.3164 / (Re ** 0.25)

    x = fsolve(eq_smooth, x0=lam_init, args=Re)

    return 1 / x[0] ** 2


def calc_lambda_rough(d_i, k):
    r"""
    Calculates the Darcy friction factor for a turbulent flow
    and a rough inner pipe surface (Prandtl & Nikuradse)

    .. calc_lam_rough_equation

    :math:`\lambda =\frac{1}{(2 log(3,71\frac{d_i}{k}))^2}`

    Parameters
    ----------
    d_i : numeric
        :math:`d_i`: inner pipe diameter [m]

    k : numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    return 1 / ((-2 * np.log10(k / (3.71 * d_i))) ** 2)


def calc_lambda_transition(R_e, k, d_i):
    r"""
    Calculates the Darcy friction factor for a given Reynolds number
    for a turbulent flow and the transition area between a rough and
    smooth pipe surface.

    See also :func:`~precalc_hydraulic.eq_transition`.

    Parameters
    ----------
    R_e: numeric
        :math:`Re`: Reynolds number [-]

    k : numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    d_i : numeric
        :math:`d_i`: inner pipe diameter [m]

    Returns
    -------
    Darcy friction factor [-] : numeric

    """
    lam_init = 0.25 / R_e ** 0.2
    x = fsolve(eq_transition, x0=lam_init, args=(R_e, k, d_i))
    return 1 / x[0] ** 2


def delta_p(v, d_i, k=0.1, T_medium=90, length=1,
            pressure=101325, R_crit=2320, fluid='IF97::Water'):

    r"""
    Function to calculate the pressure loss in a pipeline

    Parameters
    ----------
    v : numeric
        :math:`v`: flow velocity [m/s]

    d_i : numeric
        :math:`d_i`: inner pipe diameter [m]

    k : numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    T_medium : numeric
        :math:`T_{medium}`: fluid temperature [°C]

    length : numeric
        :math:`l`: length of the pipe [m]

    pressure : numeric
        :math:`p`: pressure in the pipe [Pa]

    R_crit : numeric
        :math:`Re_{crit}`: critical Reynolds number between laminar and turbulent flow [-]

    fluid : str
        name of the fluid used

    Returns
    -------
    Pressure drop [Pa] : numeric

    """
    k = k * 0.001

    # get density of water [kg/m^3]
    d = PropsSI('D', 'T', T_medium + 273.15, 'P', pressure, fluid)
    # dynamic viscosity eta [kg/(m*s)]
    d_v = PropsSI('V', 'T', T_medium + 273.15, 'P', pressure, fluid)
    k_v = calc_k_v(d_v, d)

    # Reynolds number
    R_e = calc_Re(v, d_i, k_v)

    if R_e < R_crit:  # laminar flow
        lam = calc_lambda_laminar(R_e)
        d_p = calc_d_p(lam, length, d_i, d, v)

    else:  # turbulent flow

        if R_e * k / d_i < 65:
            # Smooth pipe

            if R_e < 10**5:
                lam = calc_lambda_turb1(R_e)

            elif R_e >= 10**5 and R_e < 10**6:
                lam = calc_lambda_turb2(R_e)

            else:
                # Re > 10^6
                lam = calc_lambda_turb3(R_e)

        elif R_e * k / d_i > 1300:
            # Rough pipe
            lam = calc_lambda_rough(d_i, k)

        else:
            # Transition range 65 < Re * k/d < 1300
            lam = calc_lambda_transition(R_e, k, d_i)

        d_p = calc_d_p(lam, length, d_i, d, v)

    return d_p


def calc_v(vol_flow, d_i):
    r"""
    Calculates the velocity for a given volume flow and inner diameter of a pipe.

    .. calc_v_equation:

    :math:`v_{flow}=\frac{\dot{V}}{(\frac{d_i}{2})^2*\pi}`

    Parameters
    ----------
    vol_flow: numeric
        :math:`\dot{V}`: volume flow [m³/h]

    d_i: numeric
        :math:`d_i`: inner diameter [m]

    Returns
    -------
    flow velocity [m/s] : numeric

    """
    return vol_flow / ((d_i * 0.5)**2 * math.pi * 3600)


def v_max_secant(d_i, T_average, k=0.1, p_max=100, p_epsilon=1,
                 v_0=1, v_1=2,
                 pressure=101325, fluid='IF97::Water'):
    r"""Calculates the maximum velocity via iterative approach
    using the secant method.

    The two different starting values v_0 and v_1 should be in the
    area of the maximum flow velocity, as iteration starts from there.

    Parameters
    ----------
    d_i: numeric
        :math:`d_i`: inner diameter [m]

    T_average: numeric
        :math:`T_{av}`: average temperature [°C]

    k: numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    p_max: numeric
        :math:`p_{max}`: maximum pressure drop in pipeline [Pa]

    p_epsilon: numeric
        :math:`p_\epsilon`: accuracy of pressure [Pa]

    v_0: numeric
        :math:`v_0`: first value of initial guess for maximum flow velocity [m/s]

    v_1: numeric
        :math:`v_1`: second value of initial guess for maximum flow velocity [m/s]

    pressure: numeric
        :math:`p`: pressure level [pa]

    fluid: str
        type of fluid, default: 'IF97::Water'

    Returns
    -------
    maximum flow velocity [m/s] : numeric

    """
    p_new = 0
    v_new = 0
    n = 0
    while n < 100:
        n += 1

        p_0 = delta_p(v_0, k=k, d_i=d_i, T_medium=T_average,
                      pressure=pressure, fluid=fluid)

        p_1 = delta_p(v_1, k=k, d_i=d_i, T_medium=T_average,
                      pressure=pressure, fluid=fluid)

        v_new = v_1 - (p_1 - p_max) * (v_1 - v_0) / (p_1 - p_0)

        p_new = delta_p(v_new, k=k, d_i=d_i, T_medium=T_average,
                        pressure=pressure, fluid=fluid)

        if abs(p_new - p_max) < p_epsilon:
            break

        else:
            v_0 = v_1
            v_1 = v_new

    logger.info(
        "Maximum flow velocity calculated. Iterations: %d, "
        "Flow velocity: %.4f [m/s], Pressure drop: %.4f [Pa/m]"
        % (n, v_new, p_new)
    )

    return v_new


def v_max_bisection(d_i, T_average, k=0.1, p_max=100,
                    p_epsilon=0.1, v_epsilon=0.001,
                    v_0=0.01, v_1=10,
                    pressure=101325, fluid='IF97::Water'):
    r"""Calculates the maximum velocity via bisection for a given pressure drop.

    The two starting values `v_0` and `v_1` need to be given,
    with `v_0` below the expected flow velocity and `v_1` above.
    These are the starting values for the bi-section method.

    If either of the stop-criteria `p_epsilon` or `v_epsilon` is reached,
    the iterative calculation is stopped.

    Parameters
    ----------
    d_i: numeric
        :math:`d_i`: inner diameter [m]

    T_average: numeric
        :math:`T_{av}`: average temperature [°C]

    k: numeric
        :math:`k`: roughness of inner pipeline surface [mm]

    p_max: numeric
        :math:`p_{max}`: maximum pressure drop in pipeline [Pa]

    p_epsilon: numeric
        :math:`p_\epsilon`: accuracy of pressure [Pa]

    v_epsilon: numeric
        :math:`v_\epsilon`: accuracy of velocity [m/s]

    v_0 : numeric
        :math:`v_0`: first value of initial guess for maximum flow velocity [m/s]

    v_1: numeric
        :math:`v_1`: second value of initial guess for maximum flow velocity [m/s]

    pressure: numeric
        :math:`p`: pressure level [Pa]

    fluid: str
        type of fluid, default: 'IF97::Water'

    Returns
    -------
    maximum flow velocity [m/s] : numeric

    """
    p_0 = delta_p(v_0, k=k, d_i=d_i, T_medium=T_average,
                  pressure=pressure, fluid=fluid)

    p_1 = delta_p(v_1, k=k, d_i=d_i, T_medium=T_average,
                  pressure=pressure, fluid=fluid)

    if (p_0 - p_max) * (p_1 - p_max) >= 0:
        raise AttributeError(
            "The initial guesses `v_0` and `v_1` must be "
            "below and above the expected flow velocity."
        )

    p_new = 0
    v_new = 0
    n = 0
    while n < 200:
        n += 1

        p_0 = delta_p(v_0, k=k, d_i=d_i, T_medium=T_average,
                      pressure=pressure, fluid=fluid)

        p_1 = delta_p(v_1, k=k, d_i=d_i, T_medium=T_average,
                      pressure=pressure, fluid=fluid)

        v_new = 0.5 * (v_1 + v_0)

        p_new = delta_p(v_new, k=k, d_i=d_i, T_medium=T_average,
                        pressure=pressure, fluid=fluid)

        if abs(p_new - p_max) < p_epsilon:
            logger.info("Bi-section method: p_epsilon criterion reached.")
            break

        if abs(v_1 - v_0) < v_epsilon:  # wieso v_1 und v_0?
            logger.info("Bi-section method: v_epsilon criterion reached.")
            break

        else:
            # no stop criteria reached
            # check if p_new is above or below p_max
            if (p_0 - p_max) * (p_new - p_max) < 0:
                v_1 = v_new
            else:
                v_0 = v_new

    logger.info(
        "Maximum flow velocity calculated. Iterations: %d, "
        "Flow velocity: %.4f [m/s], Pressure drop: %.4f [Pa/m]"
        % (n, v_new, p_new)
    )

    return v_new


def calc_power(T_vl=80, T_rl=50, mf=3, p=101325):
    r"""
    Function to calculate the thermal power based on mass flow and temperature difference.

    .. calc_power_equation:

    :math:`P_{th} = \dot{m} \cdot (c_{p_{VL}} \cdot T_{VL} - c_{p_{RL}} \cdot T_{RL})`

    Parameters
    ----------
    T_vl: numeric
        :math:`T_{VL}`: forward temperature [°C]

    T_rl: numeric
        :math:`T_{RL}`: return temperature [C°]

    mf: numeric
        :math:`\dot{m}`: mass flow [kg/s]

    p: numeric
        :math:`p`: pressure [Pa]

    Returns
    -------
    thermal power [W] : numeric

    """
    cp_vl = PropsSI('C', 'T', T_vl + 273.15, 'P', p, 'IF97::Water')

    cp_rl = PropsSI('C', 'T', T_rl + 273.15, 'P', p, 'IF97::Water')

    return mf * (cp_vl * (T_vl + 273.15) - cp_rl * (T_rl + 273.15))


def calc_mass_flow(v, di, T_av, p=101325):
    r"""
    Calculates the mass flow in a pipe for a given density, diameter and flow velocity.
    The average temperature is needed for a correct value of the density.

    .. calc_mass_flow_equation:

    :math:`\dot{m} = \pi \rho_{T_av}  v  \big( \frac{d_i}{2}  \big) ^2`

    Parameters
    ----------
    v : numeric
        :math:`v`: flow velocity [m/s]

    di : numeric
        :math:`d_i`: inner diameter [m]

    T_av : numeric
        :math:`T_av`: temperature level [°C]

    p: numeric
        :math:`p`: pressure [Pa]

    Returns
    -------
    mass flow [kg/s] : numeric

    """
    rho = PropsSI('D', 'T', T_av + 273.15, 'P', p, 'IF97::Water')

    return rho * v * (0.5 * di) ** 2 * math.pi


def calc_mass_flow_P(P, T_av, delta_T, p=101325):
    r"""
    Calculates the mass flow in a pipe for a given power and heat difference.
    The average temperature is needed for a correct value of the heat capacity.

    .. calc_mass_flow_P_equation:

    :math:`\dot{m} = \frac{P}{c_{P_{T_{av}}} \cdot \Delta T}`

    Parameters
    ----------
    P : numeric
        :math:`P`: power [W]

    T_av : numeric
        :math:`T_{av}`: average temperature [°C]

    delta_T : numeric
        :math:`\Delta T`: temperature difference [K]

    p: numeric
        :math:`p`: pressure [Pa]

    Returns
    -------
    mass flow [kg/s]: numeric

    """
    cp = PropsSI('C', 'T', T_av + 273.15, 'P', p, 'IF97::Water')

    return P / (cp * delta_T)


def calc_v_mf(mf, di, T_av, p=101325):
    r"""
    Calculates the flow velocity for a given mass flow and inner diameter.
    The average temperature is needed for a correct value of the density.

    .. calc_v_mf_equation:

    :math:`v = \frac{\dot{m}}{\pi \rho \cdot \big( \frac{d_i}{2} \big)^2 }`

    Parameters
    ----------
    mf : numeric
        :math:`dot{m}`: mass flow [kg/s]

    di : numeric
        :math:`d_i`: inner diameter [m]

    T_av : numeric
        :math:`T_{av}`: average temperature [°C]

     p: numeric
        :math:`p`: pressure [Pa]

    Returns
    -------
    flow velocity [m/s]: numeric

    """
    rho = PropsSI(
        'D', 'T', T_av + 273.15, 'P', p, 'IF97::Water')  # [kg/m^3]

    return mf / (rho * (0.5 * di) ** 2 * math.pi)


def calc_pipe_loss(temp_average, u_value, temp_ground=10):
    r"""
    Calculates the heat loss of a DHS pipe trench.

    Temperatures must be given in the same unit, K or °C.

    .. calc_pipe_loss_equation:

    :math:`P_{loss} = (T_{average} - T_{ground}) \cdot U`

    Parameters
    ----------
    temp_average : float
        :math:`T_{average}`: Average temperature of medium in
        (if u_value relates to forward and return pipe,
        the average temperature of forward and return must be given.)
    u_value : float
        :math:`U`: Heat transmission coefficient of whole trench in W/(m*K)
        (u_value of forward and return pipe must be summed up, if total
        heat loss should be calculated.)
    temp_ground : float
        :math:`T_{ground}`: Temperature of surrounding, e.g. ground.

    Returns
    -------
    Heat loss of pipe trench [W/m]: float
    """
    return (temp_average - temp_ground) * u_value
