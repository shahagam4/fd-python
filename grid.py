"""This module contains a class named Grid, which can used for creating
a grid.
Grid can be use for running finite difference method on it.

-*- coding: utf-8 -*-
Author: Agam Shah(shahagam4@gmail.com)

"""




class Grid:
    """
    Contains set of parameters which decribes entire grid.

    Parameters
    ----------
    maximum_asset_value : float
        Maximum value of asset(stock price) price in grid
    minimum_asset_value : float, optional
        Minimum value of asset(stock price) price in grid
        (the default is 0)
    expiration_time : float, optional
        Time of expiration in years
        (the default is 1.0)
    number_of_asset_step : int, optional
        Number of steps in asset price
    number_of_time_step : int, optional
        Number of steps in time
    maximum_interest_rate_value : float, optional
        Maximum value of interest rate in grid
    minimum_interest_rate_value : float, optional
        Minimum value of interest rate in grid
    number_of_interest_rate_step : int, optional
        Number of steps in interest rate

    """

    def __init__(self, maximum_asset_value, minimum_asset_value=0,
                 expiration_time=1.0, number_of_asset_step=20,
                 number_of_time_step=20, maximum_interest_rate_value=None,
                 minimum_interest_rate_value=None,
                 number_of_interest_rate_step=None):
        self.minimum_asset_value = minimum_asset_value
        self.maximum_asset_value = maximum_asset_value
        self.expiration_time = expiration_time
        self.number_of_asset_step = number_of_asset_step
        self.number_of_time_step = number_of_time_step
        self.maximum_interest_rate_value = maximum_interest_rate_value
        self.minimum_interest_rate_value = minimum_interest_rate_value
        self.number_of_interest_rate_step = number_of_interest_rate_step
