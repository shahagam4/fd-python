import csv

from option_explicit_fd import Option_Value_2D_US


print('BSM value is 13.269')

C=Option_Value_2D_US(0.2,0.1,"C",100,1, "N",20)
dS1 = float(2*100/20)
V1=C[int(100/dS1), 2]
print('Value for 20 asset steps: ',V1)

C=Option_Value_2D_US(0.2,0.1,"C",100,1, "N",30)
dS2 = float(2*100/30)
V2=C[int(100/dS2), 2]
print('Value for 30 asset steps: ',V2)

V_re=(dS2**2*V1-dS1**2*V2)/(dS2**2-dS1**2)
print('Richardson extrapolate value is: ', V_re)