import numpy as np
import csv
'''
Code for option value using explicit finite difference with discretely paid dividend.
PType="C" for call option and PType="P" for put option.
EType="Y" for early excercise.
NAS is number of asset steps.
Reference 78.10 of book "Paul-Wilmott-on-Quantitative-Finance"
Output: The option values and greeks output.
'''

def dividendgivenasset(Asset_value):
    return 0.05*Asset_value

def Option_Value_2D_US(Vol, Int_Rate, PType, Strike, Expiration, EType, NAS, Dividend_date):
    S = np.zeros((1,NAS+1))
    Payoff = np.zeros((1,NAS+1))
    dS = float(2*Strike/NAS)
    dt = 0.9/(Vol**2*NAS**2)
    NTS = int(Expiration/dt) + 1
    dt = Expiration / NTS
    '''IntermidiateTS = ((Expiration-Dividend_date)/dt) + 1
    dt = (Expiration-Dividend_date)/IntermidiateTS
    NTS = int(Expiration/dt)
    print(NTS, Expiration/dt)'''
    POD = int(Dividend_date/dt)


    VOld = np.zeros((1,NAS+1))
    VNew = np.zeros((1,NAS+1))
    Dummy = np.zeros((NAS+1,6))

    q=1

    if PType=="P":
        q = -1
    
    for  i in range(NAS+1):
        S[0,i] = i*dS
        VOld[0, i] = max(q*(S[0,i]-Strike), 0)
        Payoff[0,i] = VOld[0, i]
        Dummy[i,0] = S[0,i]
        Dummy[i,1] = Payoff[0,i]
        

    for k in range(1,NTS+1):
        for i in range(1,NAS):
            Delta = (VOld[0,i+1]-VOld[0,i-1])/(2*dS)
            Gamma = (VOld[0,i+1]-2*VOld[0,i]+VOld[0,i-1])/(dS*dS)
            Theta = -0.5*Vol**2*S[0,i]**2*Gamma-Int_Rate*S[0,i]*Delta+Int_Rate*VOld[0,i]
            VNew[0,i] = VOld[0,i]-dt*Theta
        VNew[0,0] = VOld[0,0]*(1-Int_Rate*dt)
        VNew[0,NAS] = 2*VNew[0,NAS-1]-VNew[0,NAS-2]

        if(k==POD):
            for i in range(NAS+1):
                VOld[0,i] = VNew[0,i]

            for i in range(NAS+1):
                inew = int(max(S[0, i]-dividendgivenasset(S[0, i]), 0)/dS)
                ratio = (max(S[0, i]-dividendgivenasset(S[0, i]), 0) -inew*dS )/dS
                VNew[0, i] = (1-ratio)*VOld[0, inew]+ratio*VOld[0, inew+1]

        for i in range(NAS+1):
            VOld[0,i] = VNew[0,i]

        if EType=="Y":
            for i in range(NAS+1):
                VOld[0,i] = max(VOld[0,i],Payoff[0,i])

    for i in range(1,NAS):
        Dummy[i,2] = VOld[0,i]
        Dummy[i,3] = (VOld[0,i+1]-VOld[0,i-1])/(2*dS)
        Dummy[i,4] = (VOld[0,i+1]-2*VOld[0,i]+VOld[0,i-1])/(dS*dS)
        Dummy[i,5] = -0.5*Vol**2*S[0,i]**2*Dummy[i,4]-Int_Rate*S[0,i]*Dummy[i,3]+Int_Rate*VOld[0,i]
    
    Dummy[0,2] = VOld[0,0]
    Dummy[NAS,2] = VOld[0,NAS]
    Dummy[0,3] = (VOld[0,1]-VOld[0,0])/dS
    Dummy[NAS,3] = (VOld[0,NAS]-VOld[0,NAS-1])/dS
    Dummy[0,4] = 0
    Dummy[NAS,4] = 0
    Dummy[0,5] = Int_Rate*VOld[0,0]
    Dummy[NAS,5] = -0.5*Vol**2*S[0,NAS]**2*Dummy[NAS,4]-Int_Rate*S[0,NAS]*Dummy[NAS,3]+Int_Rate*VOld[0,NAS]
    return Dummy
		

C=Option_Value_2D_US(0.2,0.05,"C",100,1, "Y",40, 0.5)

with open("table_discrete_dividend.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(C)