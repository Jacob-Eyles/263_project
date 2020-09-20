#######################################################
#   ENGSCI-263  LPM FUNCTIONS   GROUP 5

#   Functions:
#           - pressure_model
#           - conc_model
#           - solve_pressure
#           - solve_copper
#           - plottingpressure
#           - analyt
#           - analytcopper
#           - posteriorcopper
#           - plot_posteriors
#           - plot_predictions
#           - plottingfunc

#######################################################
# main.py
#######################################################
#imports
import numpy as np
import data_plot
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.interpolate import interp1d
########################################################
# global variables:
########################################################
global scenario,afit,bfit,P0fit,P1fit,pmodel_end,cmodel_end
# initialise pmodel_end
scenario = 'normal'
pmodel_end = 2
cmodel_end = 2

# collect data from csv files (copper conc., pressure, extraction):
tcu,cu = np.genfromtxt('ac_cu.csv', delimiter = ',', skip_header=True).T
tp,p = np.genfromtxt('ac_p.csv', delimiter = ',', skip_header=True).T
tq,q = np.genfromtxt('ac_q.csv', delimiter = ',', skip_header=True).T

# set of time values for aquifer.
tv = np.linspace(1980,2018,(2018-1980)*10+1)        # small time steps for extraction [1980-2018]
t_new = np.append(tq,[2019,2020])                   # model time to current [1980-2020]
t_large = np.linspace(2020,2060,(2060-2020)*10+1)   # model prediction time [2020-2060]
tp_ = np.linspace(1980,2016,(2016-1980)*10+1)       # small time steps for pressure [1980-2016]
tcu_ = np.linspace(1980,2015,(2015-1980)*10+1)      # small time steps for copper conc. [1980-2015]

# interpolated values for pressure.
fp = interp1d(tp, p, kind='cubic')
interpp = fp(tp_)

# times for analytical comparions
anatime = np.linspace(0,(2018-1980),(2018-1980)*4+1)
#########################################################
# functions
#########################################################

def pressure_model(t, P, a, b, P0, P1):
    ''' This function returns the derivative dP/dt at time t for given parameters.

        Parameters:
        -----------
        t : float
            Independant variable of time at which the derivative dP/dt is to be solved. (s)
        P : float
            Dependant variable Pressure to be solved. (MPa)
        a : float
            Extraction coefficient 'a'.
        b : float
            Recharge coefficient 'b'.
        P0 : float
            Pressure at 'low pressure boundary'. (MPa)
        P1 : float
            Pressure at 'high pressure boundary'. (MPa)

        Returns:
        --------
        dP/dt : float
            Value of pressure time derivative at time t.
        
        NOTE: The equation produced i.e. the value of dP/dt, is dependant on
        the global variable 'scenario'. Each scenario has a particular set of
        parameters which may differ.
    '''
    # pressure derivative depends on extraction scenario:
    if scenario == 'normal':                
        qi = np.interp(t,tq,q)                  # find the corresponding q value at time t:
        return -a*qi - b*(P-P0) - b*(P-P1)      # return dpdt
    if scenario == 'analyticalp' or scenario == 'zeroq':
        return - b*(P-P0) - b*(P-P1)            # zero extraction
    # the other scenarios have specific constant extraction rates.  
    if scenario == 'analyticalcu':
        return -a*18
    if scenario == 'increase':
        return -a*40 - b*(P-P0) - b*(P-P1)
    if scenario == 'staysame':
        return -a*20 - b*(P-P0) - b*(P-P1)
    if scenario == 'decrease1':
        return -a*15 - b*(P-P0) - b*(P-P1)
    if scenario == 'decrease2':
        return -a*10 - b*(P-P0) - b*(P-P1)
    if scenario == 'decrease3':
        return -a*5 - b*(P-P0) - b*(P-P1)
    if scenario == 'largeex':
        return -a*60 - b*(P-P0) - b*(P-P1)
  
def conc_model(t, C, P, dcsrc, M0, a, b, P1):
    ''' This function returns the derivative dC/dt at time t for given parameters.

        Parameters:
        -----------
        t : float
            Independant variable of time at which the derivative dC/dt is to be solved. (s)
        C : float
            Dependant variable Copper concentration to be solved. (mg/L)
        P : float
            Value of system pressure at time of 't'. (MPa)
        dcsrc : float
            Combined product of leaching coefficient 'd' and leaching copper conc.'Csrc'.
        M0 : float
            Value for total mass of the system ()
        P1 : float
            Pressure at 'high pressure boundary'.

        Returns:
        --------
        dP/dt : float
            Value of pressure time derivative at time t.
        
        NOTE: The equation produced i.e. the value of dP/dt, is dependant on
        the global variable 'scenario'. Each scenario has a particular set of
        parameters which may differ.
    '''
    # boolean switch for direction of flow:
    # NOTE: P0 is set as 0 relative pressure.
    Cprime = 0
    if 0 > P:
        Cprime = 1
    # if water flows in, Cprime term goes to 1, if flow is out Cprime goes to 0.
    dcdt = (-b/a)*(P)*(-C)*Cprime + (b/a)*(P-P1)*C - dcsrc*(P-P1/2)
    return dcdt/M0

def solve_pressure(t,a,b,P1):
    '''
    '''
    if scenario in ['normal']:
        Ps = [P1/2,]                    # initial condition: Pi=(P0+P1)/2 where P0 = 0.
    elif scenario in ['analyticalp','analyticalcu']:
        Ps = [10,]                      # initial condition for pressure analytical model.            
    else:
        Ps = [pmodel_end]               # initial condition for predictions (continues original model).

    for t0,t1 in zip(t[:-1],t[1:]):                          # solve at pressure steps
        dpdt1 = pressure_model(t0, Ps[-1], a, b, 0, P1)      # predictor gradient
        pp = Ps[-1] + dpdt1*(t1-t0)                          # predictor step
        dpdt2 = pressure_model(t1, pp, a, b, 0, P1)          # corrector gradient
        Ps.append(Ps[-1] + 0.5*(t1-t0)*(dpdt2+dpdt1))        # corrector step 
    return Ps   

def solve_copper(t, dcsrc, M0):
    '''
    '''

    if scenario not in ['normal']:
        if scenario in ['analyticalcu']:
            Cus = [2,]                                       # initial value for copper analytical.
            solved = solve_pressure(anatime,afit,bfit,P1fit)
            def inter(t0):                                   # define interpolate function using analytical times.
                return np.interp(t0,anatime,solved)
                
        else:
            Cus = [cmodel_end]                               # initial value for copper predictions.
            solved = solve_pressure(t,afit,bfit,P1fit)
            def inter(t0):                                   # define interpolate function using prediction passed in times.
                return np.interp(t0,t,solved)
    else:
        Cus = [cu[0],]                                       # initial value for copper data (normal scenario).
        def inter(t0):                                       # define interpolate function using interpolated pressure data.
            return np.interp(t0,tp_,interpp)

    # inter(t) is called to find a value for pressure for copper conc. dervative at each time step.
    for t0,t1 in zip(t[:-1],t[1:]):                                               # solve at pressure steps
        dcdt1 = conc_model(t0, Cus[-1], inter(t0), dcsrc, M0, afit, bfit, P1fit)  # predictor gradient
        cc = Cus[-1] + dcdt1*(t1-t0)                                              # predictor step
        dcdt2 = conc_model(t1, cc, inter(t1), dcsrc, M0, afit, bfit, P1fit)       # corrector gradient
        Cus.append(Cus[-1] + 0.5*(t1-t0)*(dcdt2+dcdt1))                           # corrector step
    return Cus

def analyt(time, pars):
    '''
    '''
    a,b,P0,P1 = pars
    # initial value for analytical pressure is deemed as 10 MPa. This is arbitrary.
    return (10 - (P0+P1)/2)*np.exp(-2*b*time) + (P0+P1)/2

def analytcopper(time, pars, initialC = 2.):
    '''
    '''
    a,b,P0,P1,dcsrc,M0 = pars    
    # initial value for analytical copper conc. is deemed as 2 mg/L. This is arbitrary.
    # initial pressure is set to 10. This is the value of intial pressure in pressure analytical.
    return -dcsrc/M0*(-a*18*time/2 + 10 - (P0+P1)/2)*time + initialC

def posteriorcopper(ppars, pcov, initial_guess,sigma):
    '''
    '''
    # sample a number of posterior samples from pressure variables.
    psamples = np.random.multivariate_normal(ppars,pcov,10)
    # initialise arrays for copper and pressure variable samples.
    csamples = []
    pressureparams = []
    # make sure afit,bfit,P1fit are global.
    global afit,bfit,P1fit

    # for each pressure sample variables, calculate covariance for copper posterior.
    for sample in psamples:
        # set solve_pressure variables to posterior samples.
        afit,bfit,P1fit = sample
        cpars, ccov = curve_fit(solve_copper, xdata=tcu, ydata=cu, p0=initial_guess, sigma = sigma, absolute_sigma=True)
        cps = np.random.multivariate_normal(cpars, ccov, 10)
        # copper samples are produced for each pressure sample. 
        for csamp in cps:
            # these are stored into corresponding arrays for parameters and returned.
            csamples.append(csamp)
            pressureparams.append(sample)
    
    return csamples, pressureparams

def plot_posteriors(figure,ps,time1,time2=None,scenariopps=[],pparams=[],pressure=True):
    '''
    '''
    # make sure all variables changed can effect other functions.
    global scenario,pmodel_end,cmodel_end,afit,bfit,P1fit
    # index parameter counter.
    k = 0
    num = 0
    pressurepos = np.zeros((len(scenariopps),len(ps)))
    copperpos = np.zeros((len(scenariopps),len(ps)))
    # for each posterior sample, plot depending on scenario:
    for pi in ps:
        if pressure:
            # plot all normal scenario posteriors up to 2020.
            scenario = 'normal'
            pposteriors1 = solve_pressure(time1, *pi)
            figure.plot(time1, pposteriors1, 'k-', alpha=0.2, lw=0.5)
            # the start of each prediction is the end of the posterior model.
            pmodel_end = pposteriors1[-1]

            cc = 0
            for s in scenariopps:
                # for each scenario plot the corresponding posterior for pressure.
                colours = ['y','g','c','b','m','k']
                scenario = s
                pposteriors2 = solve_pressure(time2, *pi)
                # calculate the posterior value at 2040.
                pressurepos[cc][num] = np.interp(2040,time2,pposteriors2)
                figure.plot(time2, pposteriors2, colours[cc], alpha=0.2, lw=0.5)
                cc += 1
        
        else:
            # plot all normal scenario posteriors up to 2020.
            scenario = 'normal'
            afit,bfit,P1fit = pparams[k]
            cposteriors1 = solve_copper(time1, *pi)
            # if posterior samples end up negative, remove.
            if cposteriors1[-1] < 0:
                break

            figure.plot(time1, cposteriors1, 'k-', alpha=0.2, lw=0.5)
            cmodel_end = cposteriors1[-1]
            # the start of each prediction is the end of the posterior model.

            cc = 0
            for s in scenariopps:
                scenario = s
                cposteriors2 = solve_copper(time2, *pi)
                # calculate the posterior value at 2040.
                copperpos[cc][num] = np.interp(2040,time2,cposteriors2)
                if s in ['zeroq','increase','staysame']:
                    # for each posterior parameters plot with corresponding pressure and copper parameters.
                    colours = ['y','g','c','b','m','k']
                    figure.plot(time2, cposteriors2, colours[cc], alpha=0.2, lw=0.5)
                cc += 1
            k += 1
        num += 1
    # changing scenario is not needed as functions past this are all plotting pre-calculated data.
    # num = 0
    # when plotting predictions, we need to showcase uncertainty:
    # if scenariopps != []:
        # for every scenario, use np.percentile for the posterior samples at 2040 time.
        # for scen in scenariopps:
            # if pressure == True:
                # print(f'95th percentile range for pressure for {scen} is {np.percentile(pressurepos[num][:],5):0.5f} to {np.percentile(pressurepos[num][:],95):0.5f}')
            # else:
                # print(f'95th percentile range for copper for {scen} is {np.percentile(copperpos[num][:],5):0.5f} to {np.percentile(copperpos[num][:],95):0.5f}')
            # num += 1

def plot_predictions(pars,pps,title,typepred = 'pressure',pparams = []):
    '''
    '''
    # take in pressure/copper conc. data
    Pi,Pd1,Pd2,Pd3,Pss,Pzq,P2020 = pars

    # create a figure and produce labels and axis.
    # Lines denoting maximum copper conc. and prediction time span are plotted.
    fig = plt.figure(figsize=[18,9.])
    ax1 = fig.add_subplot(111)
    if typepred == 'pressure':
        ax1.plot(tp,p,'go',label = 'Data of pressure values from 1980 to 2016')
        ax1.set_ylabel('pressure (Mpa)',fontsize = 15)
        ax1.plot([2040,2040],[0.035,-0.09],'k',lw=0.5)

    else:
        ax1.plot(tcu,cu,'go',label = 'Data of copper conc. values from 1980 to 2016')
        ax1.set_ylabel('copper conc. (mg/L)',fontsize = 15)
        ax1.plot([2040,2040],[0.7,2],'k',lw=0.5)
        ax1.plot([2020,2060],[2,2],'k--',lw=0.5)
        ax1.text(2020, 2.05, 'Maximum allowable copper conc (2 mg/L)', fontsize=15)

    # each of the passed in data is plotted (predictions and model).
    ax1.plot(t_new,P2020,'r',label = f'Model of {typepred} values from 1980 to 2020')
    ax1.plot(t_large,Pzq,'y',label = 'Prediction of model (decrease to 0 million L/day extraction)')
    ax1.plot(t_large,Pd3,'k',label = 'Prediction of model (decrease to 5 million L/day extraction)')
    ax1.plot(t_large,Pd2,'m',label = 'Prediction of model (decrease to 10 million L/day extraction)')
    ax1.plot(t_large,Pd1,'b',label = 'Prediction of model (decrease to 15 million L/day extraction)')
    ax1.plot(t_large,Pss,'c',label = 'Prediction of model (stay constant at 20 million L/day extraction)')
    ax1.plot(t_large,Pi,'g',label = 'Prediction of model (increase to 40 million L/day extraction)')

    ax1.set_xlabel('time (year)',fontsize = 15)
    ax1.set_title(label = f'Plot of modelled and data of {typepred} values of Onehunga aquifer (MPa) - 1980 to 2060\n' + title,fontsize = 15)

    # printing values of predictions in 2040.

    # for (data,scen) in [(Pi,'40'),(Pd1,'15'),(Pd2,'10'),(Pd3,'5'),(Pss,'20'),(Pzq,'0')]:
    #     print(f'Scenario of extraction ({scen}) is {np.interp(2040,t_large,data)}') 
    
    # scenarios to plot for posterior samples.
    scenarios =  ['zeroq','increase','staysame','decrease1','decrease2','decrease3']
    # posterior samples.
    if typepred == 'pressure':
        plot_posteriors(ax1,pps,t_new,t_large,scenarios)
        ax1.legend(loc = 'lower left')
    else:
        plot_posteriors(ax1,pps,t_new,t_large,scenarios,pparams,pressure=False)
        ax1.legend(loc = 'upper left')

    # save and show
    if typepred == 'pressure':
        plt.savefig('Figure 7 - pressure data and model and predictions')
    else:
        plt.savefig('Figure 8 - copper data and model and predictions')
    # plt.show()


def plottingfunc(fig1pars,fig2pars,fig4pars,fig5pars,fig6pars,fig7pars,fig8pars):
    '''
    '''
    ######################################
    # figure 1 - analytical of pressure:
    ana,nump,numpars = fig1pars

    fig = plt.figure(figsize=[12.0,7.])
    ax1 = fig.add_subplot(111)
    ax1.plot(anatime,ana,'r',label='Analyitical solution')
    ax1.plot(anatime,nump,'kx',label='Numerical solution')
    ax1.set_xlim([-.5,20])
    ax1.set_ylim([-0.5,11])
    ax1.set_xlabel('years since 1980',fontsize = 15)
    ax1.set_ylabel('Pressure in MPa',fontsize = 15)
    # show analytical equation:
    ax1.text(5, 6, r'Simplified equation: $P=Ae^{-2bt} + \frac{P0+P1}{2}$', fontsize=20)
    ax1.set_title(label = f'Plot of Analytical and Numerical solutions to pressure equation of Onehunga aquifer (MPa) - 1980 to 2018\nParameters: b, P0, P1 = {numpars[1], numpars[2], numpars[3]}.\nNote: extraction = 0',fontsize = 15)
    ax1.legend()
    plt.savefig('Figure 1 - analytical of pressure')
    # plt.show()
    plt.close(fig)
    # ######################################
    # figure 2 - analytical of copper conc.
    anac,numc,numparsc = fig2pars

    fig = plt.figure(figsize=[12.0,7.])
    ax2 = fig.add_subplot(111)
    ax2.plot(anatime,anac,'r',label='Analyitical solution')
    ax2.plot(anatime,numc,'kx',label='Numerical solution')
    ax2.set_xlim([-.5,38])
    ax2.set_xlabel('years since 1980',fontsize = 15)
    ax2.set_ylabel('Conc. in mg/L',fontsize = 15)
    # show analytical equation:
    ax2.text(3, 6, 'Simplified equation:\n'r'$C=\frac{-d*Csrc}{M0}(\frac{-aqot^2}{2} + Pi*t - \frac{(P0+P1)*t}{2}) + Ci$', fontsize=20)
    ax2.set_title(fontsize = 15, label = f'Plot of Analytical and Numerical solutions to copper conc. equation of Onehunga aquifer (mg/L) - 1980 to 2018\nParameters: a, b, P0, P1, d*Csrc, Mass = {numparsc[0],numparsc[1],numparsc[2],numparsc[3], numparsc[4], numparsc[5]}.\nNote: flow at high/low pressure boundaries = 0')
    ax2.legend()
    plt.savefig('Figure 2 - analytical of copper conc.')
    # plt.show()
    plt.close(fig)
    # ######################################
    # figure 3 - analytical and numerical comparsion

    fig = plt.figure(figsize=[12.0,7.])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # plotting the variance between analytical and numerical solutions as a percentage.
    ax2.plot(anatime,abs(anac-numc)*100/anac,'k',label='Analyitical solution')
    ax1.plot(anatime,abs(nump-ana)*100/ana,'k',label='Numerical solution')
    for ax in [ax1,ax2]:
        ax.set_xlabel('years since 1980',fontsize = 15)
        ax.set_ylabel('% variance from analytical',fontsize = 15)
    fig.suptitle('Convergence of Analytical and Numerical solutions to:\n Pressure (left) and Copper conc. (right) equations',fontsize = 15)
    # ax1.set_title(label = f'Convergence of Analytical and Numerical solutions to pressure. equations\nOnehunga aquifer (mg/L) - 1980 to 2018',fontsize = 15)
    plt.legend()
    plt.savefig('Figure 3 - analytical and numerical comparsion')
    # plt.show()
    plt.close(fig)
    # ######################################
    # figure 4 - variable scenarios
    [Pzq,Czq,titlezq,Plx,Clx,titlelx,Plb,Clb,titlelb,Pzb,Czb,titlezb] = fig4pars

    # create figure
    fig = plt.figure(figsize=[22.0,15.])
    # create subplots
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    # set up pressure plots on sublots.
    for plot,pressure,title in [[ax1,Pzq,titlezq],[ax2,Plx,titlelx],[ax3,Plb,titlelb],[ax4,Pzb,titlezb]]:
        plot.plot(tv,pressure,'b',label='Pressure model')
        plot.set_xlabel('years since 1980',fontsize = 15)
        plot.set_ylabel('Pressure (MPa)',fontsize = 15)
        plot.set_title(title,fontsize = 15)
        plot.legend()
    # set up copper plots on same subplots as pressure.
    ax11 = ax1.twinx()
    ax11.plot(tv,Czq,'r',label='Copper model')
    ax11.set_ylabel('Copper conc. (mg/L)',fontsize = 15)
    ax11.legend(loc='center right')
    ax22 = ax2.twinx()
    ax22.plot(tv,Clx,'r',label='Copper model')
    ax22.set_ylabel('Copper conc. (mg/L)',fontsize = 15)
    ax22.legend(loc='center right')
    ax33 = ax3.twinx()
    ax33.plot(tv,Clb,'r',label='Copper model')
    ax33.set_ylabel('Copper conc. (mg/L)',fontsize = 15)
    ax33.legend(loc='center right')
    ax44 = ax4.twinx()
    ax44.plot(tv,Czb,'r',label='Copper model')
    ax44.set_ylabel('Copper conc. (mg/L)',fontsize = 15)
    ax44.legend(loc='center right')
    # adjust plots to fit on figure.
    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig('Figure 4 - variable scenarios')
    # plt.show()
    plt.close(fig)
    ######################################
    # figure 5 - pressure data and model
    pvalues,pps,title1 = fig5pars

    fig = plt.figure(figsize=[18.0,6])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(tp,p,'go',label = 'Data of pressure values from 1980 to 2016')
    ax1.plot(tv,pvalues,'r',label = 'Model of pressure values from 1980 to 2016')
    # plot the misfit between the model and data.
    ax2.plot(tp,(p-np.interp(tp,tv,pvalues)),'rx')
    ax2.plot([1980,2016],[0,0],'k--')
    # posterior samples for pressure
    plot_posteriors(ax1,pps,tv)
    # plotting upkeep
    for ax in [ax1,ax2]:
        ax.set_xlabel('time (year)',fontsize = 15)
        ax.set_ylabel('Pressure (MPa)',fontsize = 15)
        ax.set_title(label = 'Best fit Pressure model\n' + title1,fontsize = 15)
    ax1.legend()
    # save and show
    plt.savefig('Figure 5 - pressure data and model')
    # plt.show()
    plt.close(fig)
    #######################################
    # figure 6 - copper data and model
    cvalues,cps,pparams,title2 = fig6pars

    fig = plt.figure(figsize=[18.,6.])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(tcu,cu,'go',label = 'Data of copper conc. values from 1980 to 2016')
    ax1.plot(tv,cvalues,'r',label = 'Model of copper conc. values from 1980 to 2016')
    # plot the misfit between the model and data.
    ax2.plot(tcu,(cu-np.interp(tcu,tv,cvalues)),'rx')
    ax2.plot([1980,2016],[0,0],'k--')
    # posterior samples for pressure
    plot_posteriors(ax1,cps,tv,pparams=pparams,pressure=False)
    # plotting upkeep
    for ax in [ax1,ax2]:
        ax.set_xlabel('time (year)',fontsize = 15)
        ax.set_ylabel('copper conc. (mg/L)',fontsize = 15)
        ax.set_title(label = 'Best fit Copper model\n' + title2,fontsize = 15)
    ax1.legend()
    # save and show
    plt.savefig('Figure 6 - copper data and model')
    # plt.show()
    plt.close(fig)
    #######################################
    # figure 7 - pressure data and model and predictions
    plot_predictions(fig7pars,pps,title1,typepred = 'pressure')
    #######################################
    # figure 8 - copper data and model and predictions
    plot_predictions(fig8pars,cps,title2,typepred = 'copper',pparams = pparams)
    #######################################

if __name__=="__main__":
    #############################################
    scenario = 'analyticalp'

    # initialise analytical parameters (q=0)
    pars = [1,0.5,0,2]
    # set global variables.
    afit,bfit,P0fit,P1fit = pars
    # calculate numerical and analytical models.
    ana = analyt(anatime,pars)
    nump = solve_pressure(anatime,a=pars[0],b=pars[1],P1=pars[3])
    fig1pars = [ana,nump,pars]
    #############################################
    scenario = 'analyticalcu'

    # initialise parameters (b=0)
    pars = [1,0,0,0.06]
    parsc = np.append(pars,[0.05,100])
    # global variables
    afit,bfit,P0fit,P1fit = pars
    # calculate numerical and analytical solutions.
    anac = analytcopper(anatime,parsc,initialC=2)
    numc = solve_copper(anatime,dcsrc=parsc[-2],M0=parsc[-1])
    fig2pars = [anac,numc,parsc]
    #############################################
    # take various scenarios that produces abnormal parameter values (eg: extraction is 0)...
    scenario = 'zeroq'
    # initial variables
    ppars = [1,.5,10]
    cpars = [5,100]
    # extraction is 0.
    afit,bfit,P1fit = ppars
    # solve pressure and copper for model.
    Pzq = solve_pressure(tv,*ppars)
    Czq = solve_copper(tv,*cpars)
    titlezq =(f'Plot of pressure and copper model values (non calibrated).\nCASE: extraction is 0.')

    # extraction is large.
    scenario = 'largeex'
    ppars = [1,.5,10]
    # solve pressure and copper for model.
    Plx = solve_pressure(tv,*ppars)
    Clx = solve_copper(tv,*cpars)
    titlelx =(f'Plot of pressure and copper model values (non calibrated).\nCASE: extraction is large.')

    # b is large
    scenario = 'staysame'
    ppars = [1,5,10]
    cpars = [5,100]
    afit,bfit,P1fit = ppars
    # solve pressure and copper for model.
    Plb = solve_pressure(tv,*ppars)
    Clb = solve_copper(tv,*cpars)
    titlelb =(f'Plot of pressure and copper model values (non calibrated).\nCASE: recharge term "b" is large.')

    # b is 0.
    ppars = [1,0.,10]
    afit,bfit,P1fit = ppars
    # solve pressure and copper for model.
    Pzb = solve_pressure(tv,*ppars)
    Czb = solve_copper(tv,*cpars)
    titlezb =(f'Plot of pressure and copper model values (non calibrated).\nCASE: recharge term "b" is 0.')

    fig4pars = [Pzq,Czq,titlezq,Plx,Clx,titlelx,Plb,Clb,titlelb,Pzb,Czb,titlezb]
    #############################################
    scenario = 'normal'

    # PRESSURE

    # initial guess for a,b,P1 for the pressure.
    # P0 is set as 0.
    ppars = [0.000000003,0.5,0.055]
    # ad hoc refinement of sigma has produced this value.
    sigma = [0.004]*len(tp)
    # curve fitting parameters for pressure. This produces fit parameters and covariance relations between each parameter. 
    ppars, pcov= curve_fit(solve_pressure, xdata=tp, ydata=p, p0=ppars, sigma = sigma, absolute_sigma=True)
    # samples from posterior
    pps = np.random.multivariate_normal(ppars, pcov, 100)   
    # assigned to global variables
    afit,bfit,P1fit = ppars
    # solve pressure for model.
    Pressure = solve_pressure(tv,*ppars)
    # solve model at data points and compare to find fit.
    datapressure = solve_pressure(tp,*ppars)
    Sp = np.sum((p-datapressure)**2)
    title1 =(f'Parameters: a = {afit:0.5f}, b = {bfit:0.5f}, P0 = {P0fit:0.5f}, P1 = {P1fit:0.5f}.')
    fig5pars = [Pressure,pps,title1]

    # solve for model up to year 2020.
    Pressure2020 = solve_pressure(t_new,*ppars)
    # last point is now start of prediction models.
    pmodel_end = Pressure2020[-1]
    
    # COPPER

    # initial guess for d*Csrc,M0 for copper.
    cpars = [45,2860]
    # ad hoc refinement of sigma has produced this value.
    sigma = [0.017]*len(tcu)
    # curve fitting parameters for copper. This produces fit parameters and covariance relations between each parameter.    
    cpars, ccov= curve_fit(solve_copper, xdata=tcu, ydata=cu, p0=cpars, sigma = sigma, absolute_sigma=True)
    # producing posterior samples for copper.
    cps,pparams = posteriorcopper(ppars, pcov, cpars, sigma)
    afit,bfit,P1fit = ppars
    # solve copper for model data and for fit.
    Copper = solve_copper(tv,*cpars)
    datacopper = solve_copper(tcu,*cpars)
    Scu = np.sum((cu-datacopper)**2)
    title2 =(f'Parameters: d*csrc = {cpars[0]:0.5f}, M0 = {cpars[1]:0.5f}.')
    fig6pars = [Copper,cps,pparams,title2]

    # solve copper up to 2020. Last point is used for predictions.
    Copper2020 = solve_copper(t_new,*cpars)
    cmodel_end = Copper2020[-1]
    #############################################
    # reset the pressure parameters to best fit. (function 'posteriorcopper' would have changed this).
    afit,bfit,P1fit = ppars
    # solve pressure, copper at each scenario. (predictions so use time scale from 2020 to 2060).
    #############################################
    scenario = 'increase'

    # PRESSURE
    Pressurei = solve_pressure(t_large,*ppars)
    # COPPER
    Copperi = solve_copper(t_large,*cpars)
    #############################################
    scenario = 'decrease1'

    # PRESSURE
    Pressured1 = solve_pressure(t_large,*ppars)
    # COPPER
    Copperd1 = solve_copper(t_large,*cpars)
    #############################################
    scenario = 'decrease2'

    # PRESSURE
    Pressured2 = solve_pressure(t_large,*ppars)
    # COPPER
    Copperd2 = solve_copper(t_large,*cpars)
    #############################################
    scenario = 'decrease3'

    # PRESSURE
    Pressured3 = solve_pressure(t_large,*ppars)
    # COPPER
    Copperd3 = solve_copper(t_large,*cpars)
    #############################################
    scenario = 'staysame'

    # PRESSURE
    Pressuress = solve_pressure(t_large,*ppars)
    # COPPER
    Copperss = solve_copper(t_large,*cpars)
    #############################################
    scenario = 'zeroq'

    # PRESSURE
    Pressurezq = solve_pressure(t_large,*ppars)  
    # COPPER
    Copperzq = solve_copper(t_large,*cpars)
    #############################################
    # collect predictions for seperate plots.
    fig7pars = [Pressurei,Pressured1,Pressured2,Pressured3,Pressuress,Pressurezq,Pressure2020]
    fig8pars = [Copperi,Copperd1,Copperd2,Copperd3,Copperss,Copperzq,Copper2020]
    ############################################# 
    # plottingfunc(fig1pars,fig2pars,fig4pars,fig5pars,fig6pars,fig7pars,fig8pars)

    

        



    
