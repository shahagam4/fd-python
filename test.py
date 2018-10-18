from grid import Grid
from fd_solver import *
import csv
import numpy as np

g = Grid(minimum_asset_value=0, maximum_asset_value=200,
         expiration_time=1.0, number_of_asset_step=20, number_of_time_step=18)

c = solve(g, asset_volatility=0.2, interest_rate=0.05, strike_price=100, excercise_type="European", solving_method="Explicit",
          current_stock_price=100, current_time=0.0, option_type="Call")
print(c)
