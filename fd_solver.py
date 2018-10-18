"""This module contain functions to calculate option pricing
using different finite difference methods.

This will work for 1 factor(asset) and
2 factors(asset and interest rate) models.

-*- coding: utf-8 -*-
Author: Agam Shah(shahagam4@gmail.com)

Reference "Paul-Wilmott-on-Quantitative-Finance"
"""


import math
import numpy as np
# from grid import Grid



def solve(grid, asset_volatility, interest_rate, strike_price,
          current_stock_price, excercise_type="European",
          solving_method="Explicit", extrapolation_grid=None,
          current_time=0.0, option_type="Call",
          interest_rate_volatility=None, interest_rate_drift=None):
    """This is wrapper function for option pricing.
    It takes generic input for option pricing.

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    current_stock_price : float
        the current price of asset where one want to find option value
    excercise_type : {"European","American"}
        specify excercise type
    solving_method : {"Explicit","FullyImplicit","CrankNicolson","CrankNicolsonLU",
                      "CrankNicolsonSORwithOptimalW","CrankNicolsonDouglas"}
    extrapolation_grid : Grid class object, optional
        Specify the value if one want to improve accuracy using
        Richardson's extrapolation
    current_time : float
        time at which one want to calculate the option value,
        default is t=0.0
        for 2 factor this should be always 0.0
    option_type : {"Call","Put"}
        specify option type
    interest_rate_volatility : float
        value of the interest rate volatility
    interest_rate_drift : float
        value of the interest rate drift

    Returns
    -------
    final_option_value : float
        for 1-factor returns option value at given stock price and time
    option_value_matrix : numpy_array
        for 2-factor returns option value matrix at t=0.0
    """
    final_option_value = -1
    option_value_matrix = None
    extrapolation_option_value = None
    if interest_rate_volatility is not None:
        option_value_matrix = two_factor_Explicit(
            grid, asset_volatility, strike_price, excercise_type,
            interest_rate_volatility, interest_rate_drift, option_type)
        return option_value_matrix
    else:
        if extrapolation_grid is None:

            if current_time == 0.0:

                if solving_method == "Explicit":
                    option_value_matrix = Explicit_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "FullyImplicit":
                    option_value_matrix = fully_implicit_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolson":
                    option_value_matrix = crank_nicolson_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonLU":
                    option_value_matrix = crank_nicolson_lu_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonSORwithOptimalW":
                    option_value_matrix = crankNicolsonSORwithOptimalW_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonDouglas":
                    option_value_matrix = crank_nicolson_douglas_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                
                final_option_value = interpolate_two_d(grid, option_value_matrix,
                                         current_stock_price)
            else:

                if solving_method == "Explicit":
                    option_value_matrix = Explicit_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "FullyImplicit":
                    option_value_matrix = fully_implicit_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolson":
                    option_value_matrix = crank_nicolson_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonLU":
                    option_value_matrix = crank_nicolson_lu_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonSORwithOptimalW":
                    option_value_matrix = crankNicolsonSORwithOptimalW_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                elif solving_method == "CrankNicolsonDouglas":
                    option_value_matrix = crank_nicolson_douglas_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)

                final_option_value = interpolate_three_d(grid, option_value_matrix,
                                           current_stock_price, current_time)
        else:
            if current_time == 0.0:

                if solving_method == "Explicit":
                    option_value_matrix = Explicit_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = Explicit_two_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "FullyImplicit":
                    option_value_matrix = fully_implicit_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = fully_implicit_two_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolson":
                    option_value_matrix = crank_nicolson_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = crank_nicolson_two_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolsonLU":
                    option_value_matrix = crank_nicolson_lu_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = crank_nicolson_lu_two_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolsonSORwithOptimalW":
                    option_value_matrix = crankNicolsonSORwithOptimalW_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value =\
                        crankNicolsonSORwithOptimalW_two_d(
                            extrapolation_grid, asset_volatility,
                            interest_rate, strike_price, excercise_type,
                            option_type)
                elif solving_method == "CrankNicolsonDouglas":
                    option_value_matrix = crank_nicolson_douglas_two_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value =\
                        crank_nicolson_douglas_two_d(
                            extrapolation_grid, asset_volatility,
                            interest_rate, strike_price, excercise_type,
                            option_type)

                option_value_grid = interpolate_two_d(
                    grid, option_value_matrix, current_stock_price)
                option_value_extrapolation_grid = interpolate_two_d(
                    extrapolation_grid, extrapolation_option_value,
                    current_stock_price)

                final_option_value = richardson_extrapolation(
                    grid, option_value_grid, extrapolation_grid,
                    option_value_extrapolation_grid)
            else:

                if solving_method == "Explicit":
                    option_value_matrix = Explicit_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = Explicit_three_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "FullyImplicit":
                    option_value_matrix = fully_implicit_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = fully_implicit_three_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolson":
                    option_value_matrix = crank_nicolson_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = crank_nicolson_three_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolsonLU":
                    option_value_matrix = crank_nicolson_lu_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value = crank_nicolson_lu_three_d(
                        extrapolation_grid, asset_volatility, interest_rate,
                        strike_price, excercise_type, option_type)
                elif solving_method == "CrankNicolsonSORwithOptimalW":
                    option_value_matrix = crankNicolsonSORwithOptimalW_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value =\
                        crankNicolsonSORwithOptimalW_three_d(
                            extrapolation_grid, asset_volatility,
                            interest_rate, strike_price, excercise_type,
                            option_type)
                elif solving_method == "CrankNicolsonDouglas":
                    option_value_matrix = crank_nicolson_douglas_three_d(
                        grid, asset_volatility, interest_rate, strike_price,
                        excercise_type, option_type)
                    extrapolation_option_value =\
                        crank_nicolson_douglas_three_d(
                            extrapolation_grid, asset_volatility,
                            interest_rate, strike_price, excercise_type,
                            option_type)

                option_value_grid = interpolate_three_d(grid, option_value_matrix,
                                                        current_stock_price,
                                                        current_time)
                option_value_extrapolation_grid = interpolate_three_d(
                    extrapolation_grid, extrapolation_option_value,
                    current_stock_price, current_time)

                final_option_value = richardson_extrapolation(
                    grid, option_value_grid, extrapolation_grid,
                    option_value_extrapolation_grid)
    return final_option_value



def richardson_extrapolation(grid, option_value_grid, extrapolation_grid,
                             option_value_extrapolation_grid):
    """Richardson's extrapolation method take two non similar grids
    (means different asset step size grids) and give option value with
    improved accuracy.

    For more information refer section 78.8

    Parameters
    ----------
    grid : Grid class object
        Grid class object for first grid
    option_value_grid : float
        option value correspond to first grid
    extrapolation_grid : Grid class object
        Grid class object for second grid
    option_value_extrapolation_grid : float
        option value correspond to second grid

    Returns
    -------
    float
        returns extrapolated(more accurate) option value

    """
    asset_step_size_grid = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    asset_step_size_second_grid = float(
        (extrapolation_grid.maximum_asset_value -
         extrapolation_grid.minimum_asset_value) /
        extrapolation_grid.number_of_asset_step)
    print(option_value_grid, option_value_extrapolation_grid)

    option_value = ((asset_step_size_second_grid**2 * option_value_grid -
                     asset_step_size_grid**2 * option_value_extrapolation_grid) /
                    (asset_step_size_second_grid**2-asset_step_size_grid**2))

    return option_value


def interpolate_two_d(grid, option_value_matrix, current_stock_price):
    """To estimate the option value at point in between.
    It uses bilinear interpolation to estimate value.

    For more information refer section 77.16

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    option_value_matrix : numpy_array
        one dimensional option value array
    current_stock_price : float
        stock price at which you want to find option value

    Returns
    -------
    float
        returns interpolated option value

    """
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    lower_asset_index = math.floor(
        (current_stock_price-grid.minimum_asset_value)/asset_step_size)
    upper_asset_index = lower_asset_index+1
    option_value = ((option_value_matrix[lower_asset_index] *
                     (grid.minimum_asset_value+asset_step_size *
                      upper_asset_index-current_stock_price) +
                     option_value_matrix[upper_asset_index] *
                     (current_stock_price -
                      (grid.minimum_asset_value + asset_step_size *
                       lower_asset_index)))/asset_step_size)

    return option_value


def interpolate_three_d(grid, option_value_matrix, current_stock_price, current_time):
    """To estimate the option value at point in between.
    It uses bilinear interpolation to estimate value.
    This also find option value at in between time point.

    For more information refer section 77.16

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    option_value_matrix : numpy_array
        two dimensional option value array
    current_stock_price : float
        stock price at which you want to find option value
    current_time : float
        time at which you want to find option value

    Returns
    -------
    float
        returns interpolated option value

    """
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    lower_asset_index = math.floor(
        (current_stock_price-grid.minimum_asset_value)/asset_step_size)
    upper_asset_index = lower_asset_index + 1
    time_step_size = grid.expiration_time / grid.number_of_time_step
    lower_time_index = math.floor(current_time / time_step_size)
    upper_time_index = lower_time_index + 1

    area3 = ((current_stock_price -
              (grid.minimum_asset_value+asset_step_size * lower_asset_index)) *
             (current_time-lower_time_index*time_step_size))
    area4 = ((current_stock_price -
              (grid.minimum_asset_value+asset_step_size * lower_asset_index)) *
             (upper_time_index*time_step_size-current_time))
    area2 = (grid.minimum_asset_value+asset_step_size*upper_asset_index -
             current_stock_price)*(current_time-lower_time_index*time_step_size)
    area1 = (grid.minimum_asset_value+asset_step_size*upper_asset_index -
             current_stock_price)*(upper_time_index*time_step_size-current_time)
    option_value_1 = option_value_matrix[lower_asset_index, lower_time_index]
    option_value_2 = option_value_matrix[lower_asset_index, upper_time_index]
    option_value_3 = option_value_matrix[upper_asset_index, upper_time_index]
    option_value_4 = option_value_matrix[upper_asset_index, lower_time_index]

    option_value = (area1*option_value_1+area2*option_value_2+area3 *
                    option_value_3+area4*option_value_4)/(area1+area2+area3+area4)
    return option_value


def Explicit_three_d(grid, asset_volatility, interest_rate, strike_price,
                     excercise_type="European", option_type="Call"):
    """This function uses Explicit finite-difference method.
    This is for 1 factor model.

    For more information refer section 77.11

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value,
                              grid.number_of_asset_step + 1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    if((int(grid.expiration_time /
            (asset_volatility**2*grid.number_of_asset_step**2)) + 1) >
       grid.number_of_time_step):
        print("numer Of Time Steps Changed")
        grid.number_of_time_step = (int(
            grid.expiration_time/(asset_volatility**2 *
                                  grid.number_of_asset_step**2))+1)

    time_step_size = grid.expiration_time / grid.number_of_time_step

    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    for k in range(1, grid.number_of_time_step+1):
        for i in range(1, grid.number_of_asset_step):
            delta = (option_value[i+1, k-1] -
                     option_value[i-1, k-1])/(2*asset_step_size)
            gamma = ((option_value[i+1, k-1] -
                      2*option_value[i, k-1]+option_value[i-1, k-1]) /
                     (asset_step_size*asset_step_size))
            theta = -0.5*asset_volatility**2*asset_value[i]**2*gamma - \
                interest_rate*asset_value[i]*delta + \
                interest_rate*option_value[i, k-1]
            option_value[i, k] = option_value[i, k-1]-time_step_size*theta
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        option_value[grid.number_of_asset_step, k] = 2 * \
            option_value[grid.number_of_asset_step-1, k] - \
            option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def Explicit_two_d(grid, asset_volatility, interest_rate, strike_price,
                   excercise_type="European", option_type="Call"):
    """This function uses Explicit finite-difference method.
    This is for 1 factor model.

    For more information refer section 77.11

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value) /
        grid.number_of_asset_step)
    if((int(grid.expiration_time /
            (asset_volatility**2*grid.number_of_asset_step**2))+1) >
       grid.number_of_time_step):

        print("numer Of Time Steps Changed")

        grid.number_of_time_step = (int(
            grid.expiration_time /
            (asset_volatility**2*grid.number_of_asset_step**2))+1)

    time_step_size = grid.expiration_time / grid.number_of_time_step

    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    option_value_with_greeks = np.zeros((grid.number_of_asset_step + 1, 6))

    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]
        option_value_with_greeks[i, 0] = asset_value[i]
        option_value_with_greeks[i, 1] = pay_off[i]

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        for i in range(1, grid.number_of_asset_step):
            delta = (option_value_old[i+1]-option_value_old[i-1])/(2*asset_step_size)
            gamma = (option_value_old[i+1]-2*option_value_old[i] +
                     option_value_old[i-1])/(asset_step_size*asset_step_size)
            theta = -0.5*asset_volatility**2*asset_value[i]**2*gamma - \
                interest_rate*asset_value[i]*delta+interest_rate*option_value_old[i]
            option_value_new[i] = option_value_old[i]-time_step_size*theta
        option_value_new[0] = option_value_old[0]*(1-interest_rate*time_step_size)
        option_value_new[grid.number_of_asset_step] = 2 * \
            option_value_new[grid.number_of_asset_step-1] - \
            option_value_new[grid.number_of_asset_step-2]

        option_value_old = list(option_value_new)

        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_old[i] = max(option_value_old[i], pay_off[i])

    for i in range(1, grid.number_of_asset_step):
        option_value_with_greeks[i, 2] = option_value_old[i]
        option_value_with_greeks[i, 3] = (
            option_value_old[i+1]-option_value_old[i-1])/(2*asset_step_size)
        option_value_with_greeks[i, 4] = ((
            option_value_old[i+1]-2*option_value_old[i]+option_value_old[i-1]) /
                                          (asset_step_size*asset_step_size))
        option_value_with_greeks[i, 5] = (
            (-0.5)*asset_volatility**2*asset_value[i]**2 *
            option_value_with_greeks[i, 4] - interest_rate*asset_value[i] *
            option_value_with_greeks[i, 3] + interest_rate*option_value_old[i])

    option_value_with_greeks[0, 2] = option_value_old[0]
    option_value_with_greeks[grid.number_of_asset_step,
                             2] = option_value_old[grid.number_of_asset_step]
    option_value_with_greeks[0, 3] = (
        option_value_old[1]-option_value_old[0])/asset_step_size
    option_value_with_greeks[grid.number_of_asset_step, 3] = ((
        option_value_old[grid.number_of_asset_step] -
        option_value_old[grid.number_of_asset_step-1]) /
                                                              asset_step_size)
    option_value_with_greeks[0, 4] = 0
    option_value_with_greeks[grid.number_of_asset_step, 4] = 0
    option_value_with_greeks[0, 5] = interest_rate*option_value_old[0]
    option_value_with_greeks[grid.number_of_asset_step, 5] = (
        (-0.5)*asset_volatility**2*asset_value[grid.number_of_asset_step]**2 *
        option_value_with_greeks[grid.number_of_asset_step, 4] - interest_rate *
        asset_value[grid.number_of_asset_step] *
        option_value_with_greeks[grid.number_of_asset_step, 3]+interest_rate *
        option_value_old[grid.number_of_asset_step])

    #    return option_value_with_greeks
    return option_value_old


def fully_implicit_three_d(grid, asset_volatility, interest_rate, strike_price,
                           excercise_type="European", option_type="Call"):
    """This function uses fully implicit finite-difference method.
    This is for 1 factor model.

    For more information refer section 78.2

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = -((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/2
        coefficient_b[i-1] = ((asset_volatility**2*i**2+interest_rate)*time_step_size)
        coefficient_c[i-1] = -((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/2

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = coefficient_a[grid.number_of_asset_step-2] - \
        coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1+coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 + \
        coefficient_b[grid.number_of_asset_step-2]+2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = coefficient_c[i]

    matrix_left = (np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) +
                   np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1))

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    for k in range(1, grid.number_of_time_step+1):
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value[0, k] = max(option_value[0, k], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = coefficient_a[0]*option_value[0, k]
        vector_right = np.zeros((1, grid.number_of_asset_step-1))

        vector_right = list(option_value[1:grid.number_of_asset_step, k-1])
        vector_right[0] = vector_right[0]-remainder_vector[0]
        option_value[1:grid.number_of_asset_step, k] = np.linalg.solve(matrix_left, vector_right)

        option_value[grid.number_of_asset_step, k] = 2 * \
            option_value[grid.number_of_asset_step-1, k] - \
            option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def fully_implicit_two_d(grid, asset_volatility, interest_rate, strike_price,
                         excercise_type="European", option_type="Call"):
    """This function uses fully implicit finite-difference method.
    This is for 1 factor model.

    For more information refer section 77.2

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = -((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/2
        coefficient_b[i-1] = ((asset_volatility**2*i**2+interest_rate)*time_step_size)
        coefficient_c[i-1] = -((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/2

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = coefficient_a[grid.number_of_asset_step-2] - \
        coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1+coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 + \
        coefficient_b[grid.number_of_asset_step-2]+2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = coefficient_c[i]

    matrix_left = (np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) +
                   np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1))

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]

    for _ in range(1, grid.number_of_time_step+1):# time(k) loop
        option_value_new[0] = option_value_old[0] * \
            (1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value_new[0] = max(option_value_new[0], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = coefficient_a[0]*option_value_new[0]
        vector_right = np.zeros((1, grid.number_of_asset_step-1))

        vector_right = list(option_value_old[1:grid.number_of_asset_step])
        vector_right[0] = vector_right[0]-remainder_vector[0]
        option_value_new[1:grid.number_of_asset_step] = np.linalg.solve(matrix_left, vector_right)

        option_value_new[grid.number_of_asset_step] = 2 * \
            option_value_new[grid.number_of_asset_step-1] - \
            option_value_new[grid.number_of_asset_step-2]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_new[i] = max(option_value_new[i], pay_off[i])

        option_value_old = list(option_value_new)

    return option_value_new


def crank_nicolson_three_d(grid, asset_volatility, interest_rate, strike_price,
                           excercise_type="European", option_type="Call"):
    """This function uses CRANK–NICOLSON finite-difference method.
    This is for 1 factor model.

    For more information refer section 78.3

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    matrix_left = np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) + \
        np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1)

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    for k in range(1, grid.number_of_time_step + 1):
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value[0, k] = max(option_value[0, k], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value[0, k]
        vector_right = np.matmul(matrix_right, option_value[:, k-1])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        option_value[1:grid.number_of_asset_step, k] = np.linalg.solve(matrix_left, vector_right)

        option_value[grid.number_of_asset_step, k] = 2 * \
            option_value[grid.number_of_asset_step-1, k] - \
            option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def crank_nicolson_two_d(grid, asset_volatility, interest_rate, strike_price,
                         excercise_type="European", option_type="Call"):
    """This function uses fully CRANK–NICOLSON finite-difference method.
    This is for 1 factor model.

    For more information refer section 78.3

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    matrix_left = np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) + \
        np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1)

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        option_value_new[0] = option_value_old[0] * \
            (1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value_new[0] = max(option_value_new[0], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value_new[0]
        vector_right = np.matmul(matrix_right, option_value_old[:])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        option_value_new[1:grid.number_of_asset_step] = np.linalg.solve(matrix_left, vector_right)

        option_value_new[grid.number_of_asset_step] = 2 * \
            option_value_new[grid.number_of_asset_step-1] - \
            option_value_new[grid.number_of_asset_step-2]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_new[i] = max(option_value_new[i], pay_off[i])
        option_value_old = list(option_value_new)

    return option_value_new


def crank_nicolson_lu_three_d(grid, asset_volatility, interest_rate, strike_price,
                              excercise_type="European", option_type="Call"):
    """This function uses CRANK–NICOLSON finite-difference method.
    This also uses LU decomposition.
    This is for 1 factor model.

    For more information refer section 78.3.5

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    new_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_upper_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_lower_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_diagonal[0] = diagonal[0]
    for i in range(1, grid.number_of_asset_step-1):
        new_upper_diagonal[i-1] = super_diagonal[i-1]
        new_lower_diagonal[i] = sub_diagonal[i]/new_diagonal[i-1]
        new_diagonal[i] = diagonal[i]-new_lower_diagonal[i]*super_diagonal[i-1]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    for k in range(1, grid.number_of_time_step + 1):
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value[0, k] = max(option_value[0, k], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value[0, k]

        w = np.zeros(grid.number_of_asset_step - 1)
# vector_right=np.matmul(matrix_right, V[:,k-1][:, None])-remainder_vector
        vector_right = np.matmul(matrix_right, option_value[:, k-1])-remainder_vector

        w[0] = vector_right[0]
        for i in range(1, grid.number_of_asset_step-1):
            w[i] = vector_right[i]-new_lower_diagonal[i]*w[i-1]

        option_value[grid.number_of_asset_step-1, k] =\
            (w[grid.number_of_asset_step-2]/new_diagonal[grid.number_of_asset_step-2])
        for i in range(grid.number_of_asset_step-2, 0, -1):
            option_value[i, k] = (w[i-1]-(new_upper_diagonal[i-1]*option_value[i+1, k])) / \
            new_diagonal[i-1]

        option_value[grid.number_of_asset_step, k] = 2 * \
            option_value[grid.number_of_asset_step-1, k] - \
            option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def crank_nicolson_lu_two_d(grid, asset_volatility, interest_rate, strike_price,
                            excercise_type="European", option_type="Call"):
    """This function uses fully CRANK–NICOLSON finite-difference method.
    This also uses LU decomposition.
    This is for 1 factor model.

    For more information refer section 78.3.5

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    new_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_upper_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_lower_diagonal = np.zeros(grid.number_of_asset_step-1)
    new_diagonal[0] = diagonal[0]
    for i in range(1, grid.number_of_asset_step-1):
        new_upper_diagonal[i-1] = super_diagonal[i-1]
        new_lower_diagonal[i] = sub_diagonal[i]/new_diagonal[i-1]
        new_diagonal[i] = diagonal[i]-new_lower_diagonal[i]*super_diagonal[i-1]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        option_value_new[0] = option_value_old[0]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value_new[0] = max(option_value_new[0], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value_new[0]

        w = np.zeros(grid.number_of_asset_step - 1)
# vector_right=np.matmul(matrix_right, V[:,k-1][:, None])-remainder_vector
        vector_right = np.matmul(matrix_right, option_value_old[:])-remainder_vector

        w[0] = vector_right[0]
        for i in range(1, grid.number_of_asset_step-1):
            w[i] = vector_right[i]-new_lower_diagonal[i]*w[i-1]

        option_value_new[grid.number_of_asset_step - 1] =\
            (w[grid.number_of_asset_step-2]/new_diagonal[grid.number_of_asset_step-2])
        for i in range(grid.number_of_asset_step-2, 0, -1):
            option_value_new[i] = (w[i-1]-(new_upper_diagonal[i-1]*option_value_new[i+1])) / \
            new_diagonal[i-1]

        option_value_new[grid.number_of_asset_step] = 2 * \
            option_value_new[grid.number_of_asset_step-1] - \
            option_value_new[grid.number_of_asset_step-2]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_new[i] = max(option_value_new[i], pay_off[i])
        option_value_old = list(option_value_new)
    print(option_value_new)

    return option_value_new


def crankNicolsonSORwithOptimalW_three_d(grid, asset_volatility, interest_rate,
                                         strike_price,
                                         excercise_type="European",
                                         option_type="Call"):
    """This function uses CRANK–NICOLSON finite-difference method.
    This also uses SOR with optimal w.
    This is for 1 factor model.

    For more information refer section 78.3.6

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    k = 1
    option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
    if excercise_type == "American":
        option_value[0, k] = max(option_value[0, k], pay_off[0])
    remainder_vector = np.zeros(grid.number_of_asset_step-1)
    remainder_vector[0] = -coefficient_a[0]*option_value[0, k]
    vector_right = np.matmul(matrix_right, option_value[:, k-1])
    vector_right[0] = vector_right[0]-remainder_vector[0]

    temp = np.zeros(grid.number_of_asset_step-1)
    temp2 = np.zeros(grid.number_of_asset_step+1)
    temp2[0] = option_value[0, k]
    tol = 0.001
    err = strike_price  # random big number
    temp2[1:grid.number_of_asset_step] = list(
        option_value[1:grid.number_of_asset_step, k-1])

    omega = 1.0
    number_iteration_old = 1000000000
    number_iteration_new = 0
    while omega <= 2.0:
        number_iteration_new = 0
        temp2[1:grid.number_of_asset_step] = list(
            option_value[1:grid.number_of_asset_step, k-1])
        temp2[grid.number_of_asset_step] = 0
        err = strike_price  # random big number
        while err > tol:
            err = 0
            number_iteration_new = number_iteration_new+1
            for i in range(grid.number_of_asset_step-1):
                temp[i] = (temp2[i+1]+(omega/diagonal[i]) *
                           (vector_right[i]-super_diagonal[i]*temp2[i+2]-diagonal[i] *
                            temp2[i+1]-sub_diagonal[i]*temp2[i]))
                if excercise_type == "American":
                    temp[i] = max(temp[i], pay_off[i])
                err = err + (temp[i]-temp2[i+1])**2
                temp2[i+1] = temp[i]
            temp2[grid.number_of_asset_step] = 2 * \
                temp2[grid.number_of_asset_step-1]-temp2[grid.number_of_asset_step-2]

        if number_iteration_new > number_iteration_old:
            omega = omega-0.05
            break
        number_iteration_old = number_iteration_new
        omega = omega+0.05
    # end optimal w

    for k in range(1, grid.number_of_time_step + 1):
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value[0, k] = max(option_value[0, k], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value[0, k]
        vector_right = np.matmul(matrix_right, option_value[:, k-1])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        temp = np.zeros(grid.number_of_asset_step-1)
        err = strike_price  # random big number
        option_value[1:grid.number_of_asset_step,
                     k] = option_value[1:grid.number_of_asset_step, k-1]
        while err > tol:
            err = 0
            for i in range(grid.number_of_asset_step-1):
                temp[i] = (option_value[i+1, k]+(omega/diagonal[i]) *
                           (vector_right[i]-super_diagonal[i] * option_value[i+2, k] -
                            diagonal[i]*option_value[i+1, k]-sub_diagonal[i] *
                            option_value[i, k]))
                if excercise_type == "American":
                    option_value[0, k] = max(option_value[0, k], pay_off[0])
                err = err+(temp[i]-option_value[i+1, k])**2
                option_value[i+1, k] = temp[i]
            option_value[grid.number_of_asset_step, k] = 2 * \
                option_value[grid.number_of_asset_step-1, k] - \
                option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def crankNicolsonSORwithOptimalW_two_d(grid, asset_volatility, interest_rate,
                                       strike_price,
                                       excercise_type="European",
                                       option_type="Call"):
    """This function uses fully CRANK–NICOLSON finite-difference method.
    This also uses SOR with optimal w.
    This is for 1 factor model.

    For more information refer section 78.3.6

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    coefficient_a = np.zeros(grid.number_of_asset_step-1)
    coefficient_b = np.zeros(grid.number_of_asset_step-1)
    coefficient_c = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        coefficient_a[i-1] = ((asset_volatility**2*i**2-interest_rate*i)*time_step_size)/4
        coefficient_b[i-1] = -((asset_volatility**2*i**2+interest_rate)*time_step_size)/2
        coefficient_c[i-1] = ((asset_volatility**2*i**2+interest_rate*i)*time_step_size)/4

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -coefficient_a[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        coefficient_a[grid.number_of_asset_step-2]+coefficient_c[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-coefficient_b[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        coefficient_b[grid.number_of_asset_step-2]-2*coefficient_c[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -coefficient_c[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = coefficient_a[i]
        matrix_right[i, i+1] = 1+coefficient_b[i]
        matrix_right[i, i+2] = coefficient_c[i]

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]

    #k = 1
    option_value_new[0] = option_value_old[0]*(1-interest_rate*time_step_size)
    if excercise_type == "American":
        option_value_new[0] = max(option_value_new[0], pay_off[0])
    remainder_vector = np.zeros(grid.number_of_asset_step-1)
    remainder_vector[0] = -coefficient_a[0]*option_value_new[0]
    vector_right = np.matmul(matrix_right, option_value_old[:])
    vector_right[0] = vector_right[0]-remainder_vector[0]

    temp = np.zeros(grid.number_of_asset_step-1)
    temp2 = np.zeros(grid.number_of_asset_step+1)
    temp2[0] = option_value_new[0]
    tol = 0.001
    err = strike_price  # random big number
    temp2[1:grid.number_of_asset_step] = list(
        option_value_old[1:grid.number_of_asset_step])

    omega = 1.0
    number_iteration_old = 1000000000
    number_iteration_new = 0
    while omega <= 2.0:
        number_iteration_new = 0
        temp2[1:grid.number_of_asset_step] = list(
            option_value_old[1:grid.number_of_asset_step])
        temp2[grid.number_of_asset_step] = 0
        err = strike_price  # random big number
        while err > tol:
            err = 0
            number_iteration_new = number_iteration_new+1
            for i in range(grid.number_of_asset_step-1):
                temp[i] = (temp2[i+1]+(omega/diagonal[i])*(
                    vector_right[i]-super_diagonal[i]*temp2[i+2]-diagonal[i]*temp2[i+1] -
                    sub_diagonal[i]*temp2[i]))
                if excercise_type == "American":
                    temp[i] = max(temp[i], pay_off[i])
                err = err + (temp[i]-temp2[i+1])**2
                temp2[i+1] = temp[i]
            temp2[grid.number_of_asset_step] = 2 * \
                temp2[grid.number_of_asset_step-1]-temp2[grid.number_of_asset_step-2]

        if number_iteration_new > number_iteration_old:
            omega = omega-0.05
            break
        number_iteration_old = number_iteration_new
        omega = omega+0.05
    # end optimal w

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        option_value_new[0] = option_value_old[0]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value_new[0] = max(option_value_new[0], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -coefficient_a[0]*option_value_new[0]
        vector_right = np.matmul(matrix_right, option_value_old[:])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        temp = np.zeros(grid.number_of_asset_step-1)
        err = strike_price  # random big number
        option_value_new[1:grid.number_of_asset_step] =\
            option_value_old[1:grid.number_of_asset_step]
        while err > tol:
            err = 0
            for i in range(grid.number_of_asset_step-1):
                temp[i] = (option_value_new[i+1]+(omega/diagonal[i]) *
                           (vector_right[i]-super_diagonal[i] * option_value_new[i+2]-diagonal[i] *
                            option_value_new[i+1]-sub_diagonal[i]*option_value_new[i]))
                if excercise_type == "American":
                    option_value_new[0] = max(option_value_new[0], pay_off[0])
                err = err+(temp[i]-option_value_new[i+1])**2
                option_value_new[i+1] = temp[i]
            option_value_new[grid.number_of_asset_step] = 2 * \
                option_value_new[grid.number_of_asset_step-1] - \
                option_value_new[grid.number_of_asset_step-2]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_new[i] = max(option_value_new[i], pay_off[i])
        option_value_old = list(option_value_new)
    return option_value_new


def crank_nicolson_douglas_three_d(grid, asset_volatility, interest_rate, strike_price,
                                   excercise_type="European", option_type="Call"):
    """This function uses CRANK–NICOLSON finite-difference method.
    This also uses Douglas scheme to improve accuracy.
    This is for 1 factor model.

    For more information refer section 78.6

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value = np.zeros((grid.number_of_asset_step+1, grid.number_of_time_step+1))
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    douglas = 0.5-asset_step_size**2/(12*time_step_size)

    a_left = np.zeros(grid.number_of_asset_step-1)
    b_left = np.zeros(grid.number_of_asset_step-1)
    c_left = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        a_left[i-1] = ((asset_volatility**2*i**2-interest_rate*i) *
                       time_step_size*douglas)/2
        b_left[i-1] = -((asset_volatility**2*i**2+interest_rate) *
                        time_step_size*douglas)
        c_left[i-1] = ((asset_volatility**2*i**2+interest_rate*i) *
                       time_step_size*douglas)/2

    a_right = np.zeros(grid.number_of_asset_step-1)
    b_right = np.zeros(grid.number_of_asset_step-1)
    c_right = np.zeros(grid.number_of_asset_step-1)

    for i in range(1, grid.number_of_asset_step):
        a_right[i-1] = ((asset_volatility**2*i**2-interest_rate*i) *
                        time_step_size*(1-douglas))/2
        b_right[i-1] = -((asset_volatility**2*i**2+interest_rate) *
                         time_step_size*(1-douglas))
        c_right[i-1] = ((asset_volatility**2*i**2+interest_rate*i) *
                        time_step_size*(1-douglas))/2

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -a_left[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        a_left[grid.number_of_asset_step-2]+c_left[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-b_left[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        b_left[grid.number_of_asset_step-2]-2*c_left[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -c_left[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = a_right[i]
        matrix_right[i, i+1] = 1+b_right[i]
        matrix_right[i, i+2] = c_right[i]

    matrix_left = np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) + \
        np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1)

    for i in range(grid.number_of_asset_step+1):
        option_value[i, 0] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value[i, 0]

    for k in range(1, grid.number_of_time_step + 1):
        option_value[0, k] = option_value[0, k-1]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value[0, k] = max(option_value[0, k], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -a_left[0]*option_value[0, k]
        vector_right = np.matmul(matrix_right, option_value[:, k-1])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        option_value[1:grid.number_of_asset_step, k] = np.linalg.solve(matrix_left, vector_right)

        option_value[grid.number_of_asset_step, k] = 2 * \
            option_value[grid.number_of_asset_step-1, k] - \
            option_value[grid.number_of_asset_step-2, k]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value[i, k] = max(option_value[i, k], pay_off[i])

    return option_value


def crank_nicolson_douglas_two_d(grid, asset_volatility, interest_rate, strike_price,
                                 excercise_type="European", option_type="Call"):
    """This function uses fully CRANK–NICOLSON finite-difference method.
    This also uses Douglas scheme to improve accuracy.
    This is for 1 factor model.

    For more information refer section 78.6

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    interest_rate : float
        value of the fixed interest rate
    strike_price : float
        strike price for the option
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns one dimensional numpy array of option value for t=0.0

    """
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    pay_off = np.zeros(grid.number_of_asset_step+1)
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(grid.number_of_asset_step + 1)
    option_value_new = np.zeros(grid.number_of_asset_step + 1)
    multiplier = 1

    if option_type == "Put":
        multiplier = -1

    douglas = 0.5-asset_step_size**2/(12*time_step_size)

    a_left = np.zeros(grid.number_of_asset_step-1)
    b_left = np.zeros(grid.number_of_asset_step-1)
    c_left = np.zeros(grid.number_of_asset_step-1)

    # from page 1230
    for i in range(1, grid.number_of_asset_step):
        a_left[i-1] = ((asset_volatility**2*i**2-interest_rate*i) *
                       time_step_size*douglas)/2
        b_left[i-1] = -((asset_volatility**2*i**2+interest_rate) *
                        time_step_size*douglas)
        c_left[i-1] = ((asset_volatility**2*i**2+interest_rate*i) *
                       time_step_size*douglas)/2

    a_right = np.zeros(grid.number_of_asset_step-1)
    b_right = np.zeros(grid.number_of_asset_step-1)
    c_right = np.zeros(grid.number_of_asset_step-1)

    for i in range(1, grid.number_of_asset_step):
        a_right[i-1] = ((asset_volatility**2*i**2-interest_rate*i) *
                        time_step_size*(1-douglas))/2
        b_right[i-1] = -((asset_volatility**2*i**2+interest_rate) *
                         time_step_size*(1-douglas))
        c_right[i-1] = ((asset_volatility**2*i**2+interest_rate*i) *
                        time_step_size*(1-douglas))/2

    sub_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(1, grid.number_of_asset_step-2):
        sub_diagonal[i] = -a_left[i]
    sub_diagonal[grid.number_of_asset_step-2] = - \
        a_left[grid.number_of_asset_step-2]+c_left[grid.number_of_asset_step-2]
    diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        diagonal[i] = 1-b_left[i]
    diagonal[grid.number_of_asset_step-2] = 1 - \
        b_left[grid.number_of_asset_step-2]-2*c_left[grid.number_of_asset_step-2]
    super_diagonal = np.zeros(grid.number_of_asset_step-1)
    for i in range(grid.number_of_asset_step-2):
        super_diagonal[i] = -c_left[i]

    matrix_right = np.zeros((grid.number_of_asset_step-1, grid.number_of_asset_step+1))
    for i in range(grid.number_of_asset_step-1):
        matrix_right[i, i] = a_right[i]
        matrix_right[i, i+1] = 1+b_right[i]
        matrix_right[i, i+2] = c_right[i]

    matrix_left = np.diagonal(sub_diagonal[1:], k=-1) + np.diagonal(diagonal[:]) + \
        np.diagonal(super_diagonal[0:grid.number_of_asset_step-2], k=1)

    for i in range(grid.number_of_asset_step+1):
        option_value_old[i] = max(multiplier*(asset_value[i]-strike_price), 0)
        pay_off[i] = option_value_old[i]

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        option_value_new[0] = option_value_old[0]*(1-interest_rate*time_step_size)
        if excercise_type == "American":
            option_value_new[0] = max(option_value_old[0], pay_off[0])
        remainder_vector = np.zeros(grid.number_of_asset_step-1)
        remainder_vector[0] = -a_left[0]*option_value_new[0]
        vector_right = np.matmul(matrix_right, option_value_old[:])
        vector_right[0] = vector_right[0]-remainder_vector[0]

        option_value_new[1:grid.number_of_asset_step] = np.linalg.solve(matrix_left, vector_right)

        option_value_new[grid.number_of_asset_step] = 2 * \
            option_value_new[grid.number_of_asset_step-1] - \
            option_value_new[grid.number_of_asset_step-2]
        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                option_value_new[i] = max(option_value_new[i], pay_off[i])
        option_value_old = list(option_value_new)

    return option_value_new


def two_factor_Explicit(grid, asset_volatility, strike_price,
                        interest_rate_volatility, interest_rate_drift,
                        excercise_type="European",
                        option_type="Call"):
    """This function uses fully Explicit finite-difference method.
    This is for 2 factor model.

    For more information refer section 79.3

    Parameters
    ----------
    grid : Grid class object
        Grid class object for grid
    asset_volatility : float
        volatility of asset price
    strike_price : float
        strike price for the option
    interest_rate_volatility : float
        value of the interest rate volatility
    interest_rate_drift : float
        value of the interest rate drift
    excercise_type : {"European","American"}
        specify excercise type
    option_type : {"Call","Put"}
        specify option type

    Returns
    -------
    numpy_array
        returns two dimensional numpy array of option value for t=0.0

    """
    lamda = 0.1
    rho = 0.5
    converet_rate = 0.9
    asset_value = np.linspace(grid.minimum_asset_value,
                              grid.maximum_asset_value, grid.number_of_asset_step+1)
    interest_rate_value = np.linspace(grid.minimum_interest_rate_value,
                                      grid.maximum_interest_rate_value,
                                      grid.number_of_interest_rate_step+1)
    pay_off = np.zeros(
        (grid.number_of_asset_step+1, grid.number_of_interest_rate_step+1))
    asset_step_size = float(
        (grid.maximum_asset_value-grid.minimum_asset_value)/grid.number_of_asset_step)
    interest_rate_step_size = float(
        (grid.maximum_interest_rate_value-grid.minimum_interest_rate_value) /
        grid.number_of_interest_rate_step)
    if((int(grid.expiration_time /
            (asset_volatility**2*grid.number_of_asset_step**2 +
             (interest_rate_volatility/interest_rate_step_size)**2))+1) >
       grid.number_of_time_step):
        print("numer Of Time Steps Changed for stability.")
        grid.number_of_time_step = (int(
            grid.expiration_time /
            (asset_volatility**2*grid.number_of_asset_step**2))+1)

    time_step_size = grid.expiration_time / grid.number_of_time_step
    option_value_old = np.zeros(
        (grid.number_of_asset_step+1, grid.number_of_interest_rate_step+1))
    option_value_new = np.zeros(
        (grid.number_of_asset_step+1, grid.number_of_interest_rate_step+1))
    multiplier = 1
# Dummy = np.zeros((NAS+1,6))

    if option_type == "Put":
        multiplier = -1
    print(asset_value)
    for i in range(grid.number_of_asset_step + 1):
        for j in range(grid.number_of_interest_rate_step + 1):
            option_value_old[i, j] = max(
                multiplier*(asset_value[i]-strike_price), 0)
            pay_off[i, j] = option_value_old[i, j]

    for _ in range(1, grid.number_of_time_step + 1):# time(k) loop
        for i in range(1, grid.number_of_asset_step):
            for j in range(1, grid.number_of_interest_rate_step):
                VS = (option_value_old[i+1, j] -
                      option_value_old[i-1, j])/(2*asset_step_size)
                Vrp = (option_value_old[i, j + 1] -
                       option_value_old[i, j]) / interest_rate_step_size
                Vrm = ((option_value_old[i, j] -
                        option_value_old[i, j - 1]) /
                       interest_rate_step_size)
                Vr = Vrm
                if(interest_rate_drift-lamda*interest_rate_volatility) > 0:
                    Vr = Vrp

                VSS = ((option_value_old[i + 1, j] - 2 * option_value_old[i, j] +
                        option_value_old[i - 1, j]) /
                       (asset_step_size * asset_step_size))
                Vrr = ((option_value_old[i, j + 1] - 2 * option_value_old[i, j] +
                        option_value_old[i, j - 1]) /
                       (interest_rate_step_size * interest_rate_step_size))
                VSr = ((option_value_old[i + 1, j + 1] -
                        option_value_old[i - 1, j + 1] -
                        option_value_old[i + 1, j - 1] +
                        option_value_old[i - 1, j - 1]) /
                       (4 * asset_step_size * interest_rate_step_size))
                option_value_new[i, j] = (option_value_old[i, j] + time_step_size *
                                          (0.5 * asset_value[i] * asset_value[i] *
                                           asset_volatility * asset_volatility *
                                           VSS + 0.5 * interest_rate_volatility *
                                           interest_rate_volatility * Vrr + rho *
                                           asset_volatility *
                                           interest_rate_volatility *
                                           asset_value[i] * VSr +
                                           interest_rate_value[j] *
                                           (asset_value[i] * VS -
                                            option_value_old[i, j]) +
                                           (interest_rate_drift-lamda *
                                            interest_rate_volatility) * Vr))

        for j in range(grid.number_of_interest_rate_step+1):
            option_value_new[0, j] = 0
            option_value_new[grid.number_of_asset_step, j] = 2 * \
                option_value_new[grid.number_of_asset_step-1, j] - \
                option_value_new[grid.number_of_asset_step-2, j]

        for i in range(1, grid.number_of_asset_step):
            option_value_new[i, grid.number_of_interest_rate_step] = asset_value[i]
            j = 0
            VS = (option_value_old[i + 1, j] -
                  option_value_old[i - 1, j])/(2*asset_step_size)
            Vrp = (option_value_old[i, j + 1] -
                   option_value_old[i, j]) / interest_rate_step_size
            Vrm = (option_value_old[i, j] -
                   option_value_old[i, j]) / interest_rate_step_size
            Vr = Vrm
            if(interest_rate_drift-lamda*interest_rate_volatility) > 0:
                Vr = Vrp

            VSS = (option_value_old[i + 1, j] - 2 * option_value_old[i, j] +
                   option_value_old[i - 1, j]) / (asset_step_size * asset_step_size)
            Vrr = ((option_value_old[i, j + 1] - 2 * option_value_old[i, j] +
                    option_value_old[i, j]) / (interest_rate_step_size *
                                               interest_rate_step_size))
            VSr = ((option_value_old[i + 1, j + 1] -
                    option_value_old[i - 1, j + 1] -
                    option_value_old[i + 1, j] + option_value_old[i - 1, j]) /
                   (4 * asset_step_size * interest_rate_step_size))
            option_value_new[i, j] = (option_value_old[i, j] + time_step_size *
                                      (0.5 * asset_value[i] * asset_value[i] *
                                       asset_volatility * asset_volatility *
                                       VSS + 0.5 * interest_rate_volatility *
                                       interest_rate_volatility * Vrr + rho *
                                       asset_volatility *
                                       interest_rate_volatility *
                                       asset_value[i] * VSr +
                                       interest_rate_value[j] *
                                       (asset_value[i] * VS -
                                        option_value_old[i, j]) +
                                       (interest_rate_drift-lamda *
                                        interest_rate_volatility) * Vr))

        for i in range(grid.number_of_asset_step+1):
            for j in range(grid.number_of_interest_rate_step+1):
                option_value_old[i, j] = max(
                    option_value_new[i, j], converet_rate*asset_value[i])

        if excercise_type == "American":
            for i in range(grid.number_of_asset_step+1):
                for j in range(grid.number_of_interest_rate_step+1):
                    option_value_old[i, j] = max(
                        option_value_old[i, j], pay_off[i, j])

# option_value_old = list(option_value_new)

    return option_value_new
