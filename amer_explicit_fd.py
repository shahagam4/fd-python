import numpy as np
import csv
'''
Code for American option using explicit finite difference.
PType="C" for call option and PType="P" for put option.
EType="Y" for early excercise.
NAS is number of asset steps.
Reference code page no. 1219 of book "Paul-Wilmott-on-Quantitative-Finance"
Output: The whole grid of option value for given stock price and time.
'''

def Option_Value_3D_US(Vol, Int_Rate, PType, Strike, Expiration, EType, NAS):
    S = np.zeros((1,NAS+1))
    Payoff = np.zeros((1,NAS+1))
    dS = float(2*Strike/NAS)
    dt = 0.9/(Vol**2*NAS**2)
    NTS = int(Expiration/dt) + 1
    dt = Expiration / NTS
    V = np.zeros((NAS+1,NTS+1))
    q=1

    if PType=="P":
        q = -1
    
    for  i in range(NAS+1):
        S[0,i] = i*dS
        V[i, 0] = max(q*(S[0,i]-Strike), 0)
        Payoff[0,i] = V[i, 0]

    for k in range(1,NTS+1):
        for i in range(1,NAS):
            Delta = (V[i+1, k-1]-V[i-1, k-1])/(2*dS)
            Gamma = (V[i+1, k-1]-2*V[i, k-1]+V[i-1, k-1])/(dS*dS)
            Theta = -0.5*Vol**2*S[0,i]**2*Gamma-Int_Rate*S[0,i]*Delta+Int_Rate*V[i, k-1]
            V[i, k] = V[i, k-1]-dt*Theta
        V[0, k] = V[0, k-1]*(1-Int_Rate*dt)
        V[NAS, k] = 2*V[NAS-1, k]-V[NAS-2, k]
        if EType=="Y":
            for i in range(NAS+1):
                V[i,k] = max(V[i,k],Payoff[0,i])

    return V
		