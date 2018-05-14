import csv

from euro_explicit_fd import Option_Value_3D

C=Option_Value_3D(0.2,0.05,"C",100,1,20)

with open("table77.1.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)

C=Option_Value_3D(0.2,0.05,"P",100,1,20)

with open("table77.2.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)

from amer_explicit_fd import Option_Value_3D_US

C=Option_Value_3D_US(0.2,0.05,"P",100,1, "Y",20)

with open("table77.3.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)

from option_explicit_fd import Option_Value_2D_US

C=Option_Value_2D_US(0.2,0.05,"C",100,1, "Y",40)

with open("table77.4.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)

from bilinear_interpolation import interpolate_value

C=interpolate_value(0.2,0.05,"P",100,1, "Y",20,117,0.3)
print("The interpolated value is ",C)

from upwind_differencing import Option_Value_2D_US_upwind

C=Option_Value_2D_US_upwind(0.2,0.05,"C",100,1, "Y",40)

with open("upwind.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)