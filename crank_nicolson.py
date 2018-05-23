import numpy as np
import csv
'''
Code for option value using the Crank-Nicolson Method.
PType="C" for call option and PType="P" for put option.
EType="Y" for early excercise.
NAS is number of asset steps.
NTS is number of time steps.
Reference page no. 1229 to 1236 of book "Paul-Wilmott-on-Quantitative-Finance"
Output: The option values and greeks output.
'''

def Option_Value_3D_cn(Vol, Int_Rate, PType, Strike, Expiration, EType, NAS, NTS):
    S = np.zeros((1,NAS+1))
    Payoff = np.zeros((1,NAS+1))
    dS = float(2*Strike/NAS)
    dt = float(Expiration/NTS)
    V = np.zeros((NAS+1,NTS+1))
    q=1
    A=np.zeros((1,NAS-1))
    B=np.zeros((1,NAS-1))
    C=np.zeros((1,NAS-1))
    #from page 1230
    for i in range(1,NAS):
        A[0, i-1]=((Vol**2*i**2-Int_Rate*i)*dt)/4
        B[0, i-1]=-((Vol**2*i**2+Int_Rate)*dt)/2
        C[0, i-1]=((Vol**2*i**2+Int_Rate*i)*dt)/4

    SubDiag=np.zeros((1,NAS-1))
    for i in range(1,NAS-2):
        SubDiag[0, i]=-A[0, i]
    SubDiag[0, NAS-2]=-A[0, NAS-2]+C[0, NAS-2]
    Diag=np.zeros((1,NAS-1))
    for i in range(NAS-2):
        Diag[0, i]=1-B[0, i]
    Diag[0, NAS-2]=1-B[0, NAS-2]-2*C[0, NAS-2]
    SuperDiag=np.zeros((1,NAS-1))
    for i in range(NAS-2):
        SuperDiag[0, i]=-C[0, i]

    MR=np.zeros((NAS-1,NAS+1))
    for i in range(NAS-1):
        MR[i, i]=A[0, i]
        MR[i, i+1]=1+B[0, i]
        MR[i, i+2]=C[0, i]

    ML = np.diag(SubDiag[0,1:], k=-1) + np.diag(Diag[0, :]) + np.diag(SuperDiag[0, 0:NAS-2], k=1)

    if PType=="P":
        q = -1
    
    for  i in range(NAS+1):
        S[0,i] = i*dS
        V[i, 0] = max(q*(S[0,i]-Strike), 0)
        Payoff[0,i] = V[i, 0]

    for k in range(1,NTS+1):
        V[0, k] = V[0, k-1]*(1-Int_Rate*dt)
        if EType=="Y":
            V[0, k] = max(V[0,k],Payoff[0,0])
        r=np.zeros((1,NAS-1))
        r[0, 0]=-A[0, 0]*V[0, k]
        q=np.matmul(MR, V[:,k-1])
        q[0]=q[0]-r[0,0]

        V[1:NAS,k]=np.linalg.solve(ML,q)

        V[NAS, k] = 2*V[NAS-1, k]-V[NAS-2, k]
        if EType=="Y":
            for i in range(NAS+1):
                V[i,k] = max(V[i,k],Payoff[0,i])

    return V

C=Option_Value_3D_cn(0.2,0.05,"P",100,1, "N",20,18)

with open("table_cn.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)