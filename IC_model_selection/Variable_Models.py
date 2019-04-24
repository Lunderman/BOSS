'''
There are two functions here:
1) param_estimation:
        This function performs one single paramter estimation by the MCMC Hammer given the input parameters

2)Model_selection:
        This function performs the parameter estimation problem for the various possible models and then makes the model selection based on the information criterion (IC) of each model. If the IC do not agree on the ``best'' model, then the number of MCMC_steps is increased by a factor of 10 and everything is rerun. If param_estimation could not accurately estimated the IACT, then a warning is returned printed that the user should increase MCMC_steps.

Spence Lunderman
Lunderman@math.arizona.edu
2018-10-31

'''

import numpy as np
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import corner
import time
from scipy.stats import multivariate_normal

plt.rcParams['figure.figsize'] = [16,8]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100

plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'large'



def param_estimation(nTruth,nModel,nMoment,param_Truth,MCMC_Steps = int(1E5),param_lb = [-10.,-10.],
                        param_ub = [ 10., 10.],model_eval = [0.1 , 1., 10],obs_var = 0.01,
                       Plot_figs = False,run_MCMC=True):
    '''
    Inputs:
        nTruth: 
            Number of terms in the truth 
        nModel: 
            Number of terms in the model trying to estimate the truth
        nMoment: 
            Number of moments in each model term, either 1 or 2
        param_Truth: 
            True parameters for model with nTruth terms
            The format must be:  [ a_1, a_2, ... , a_nTruth, b_1, b_2, ..., b_nTruth]
            with b_1 > b_2 >  ... > b_nTruth
        MCMC_Steps = int(5E4):
            Number of steps per MCMC Hammer walker
        param_lb = [-10.,-10.]: 
        param_ub = [ 10., 10.]: 
            Define lower and upper bounds for parameters [ a's , b's]
            We are assuming the bounds are the same for all a's and for all b's    
        model_eval = [0.1 , 1., 10]: 
            [ lb, ub, nEval]
            Define where the true model and approximate model will be evaluate
        obs_var = 0.1: 
            Observation error for likelihood distribution
        Plot_figs = False:
            Option to create figures
        run_MCMC = True:
            This allows you to turn off the parameter estimation if you only want a figure of the true model
            If run_MCMC = False, then nModel is irrelevant
    
    Outputs:
        Samples:
            All MCMC samples AFTER the burn-in samples have been removed
        IACT:
            Estimated integrated autocorrelation time for the MCMC chains as calculated by emcee.autocorr.integrated_time.
            If emcee.autocorr.integrated_time cannot accurately approximate then IACT = np.inf and there will be no burn in samples removed.
        IC:
            A list of the information criterions for the model. (this will be used for model selection)
            AIC,BIC = Akaike, Bayesian
        sampler:
            This is the unaltered emcee sampler results.
    '''
    
    # Create DataFrame for true parameters
    df_params = pd.DataFrame(columns = ['label','lb','ub','true'])
    df_params['label'] = ['$a_%s$'%(kk+1) for kk in range(nTruth)] +  ['$b_%s$'%(kk+1) for kk in range(nTruth)]
    df_params['lb'] = [param_lb[0] for kk in range(nTruth)] +  [param_lb[0]for kk in range(nTruth)]
    df_params['ub'] = [param_ub[0] for kk in range(nTruth)] +  [param_ub[0]for kk in range(nTruth)]
    df_params['true'] = param_Truth

    # ## Define the nModel term model
    def one_moment_model(a,b,M0):
        return(a*M0**(b))
    def two_moment_model(a,b,M0,M3):
        return(a*M0**(b)*M3**(1-b))

    def True_model(theta):
        if nMoment == 1:
            M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            P_val = [one_moment_model(theta[kk],theta[kk+nTruth],M0) for kk in range(nTruth)]
            return(np.sum(np.array(P_val),axis=0))
        if nMoment == 2:
            M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            M3 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            M0,M3 = np.meshgrid(M0,M3)
            P_val = [two_moment_model(theta[kk],theta[kk+nTruth],M0,M3) for kk in range(nTruth)]
            return(np.sum(np.array(P_val),axis=0))

    def MCMC_model(theta):
        if nMoment == 1:
            M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            P_val = [one_moment_model(theta[kk],theta[kk+nModel],M0) for kk in range(nModel)]
            return(np.sum(np.array(P_val),axis=0))
        if nMoment == 2:
            M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            M3 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
            M0,M3 = np.meshgrid(M0,M3)
            P_val = [two_moment_model(theta[kk],theta[kk+nModel],M0,M3) for kk in range(nModel)]
            return(np.sum(np.array(P_val),axis=0))
    
    # ### Define and plot the true observations

    # Calculate the "true" observations
    Y_true= True_model(df_params['true'].values)


    # Plot figs of truth    
    if nMoment == 1 and Plot_figs:
        Plot_Truth = False
        # Plot the first term, second term, and their sum
        M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
        for kk in range(nTruth):
            cancel_out = [mm == kk for mm in range(nTruth)]+[1 for mm in range(nTruth)]
            plt.plot(M0,True_model(df_params['true'].values*cancel_out),
                     label='$%.2fM^{%.2f}$'%(df_params['true'].values[kk],df_params['true'].values[kk+nTruth]))
        plt.plot(M0,Y_true,label='True Observations',linewidth=5)
        plt.xlabel('$M$')
        plt.title('Full %s term model'%(nTruth))
        plt.legend()
        plt.show()
    
    if nMoment == 2 and Plot_figs:
        Plot_Truth = False
        M0 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
        M3 = np.linspace(model_eval[0],model_eval[1],model_eval[2])
        M0,M3 = np.meshgrid(M0,M3)
        for kk in range(nTruth):
            cancel_out = [mm == kk for mm in range(nTruth)]+[1 for mm in range(nTruth)]
            plt.imshow(True_model(df_params['true'].values*cancel_out)[::-1])
            plt.xticks(np.arange(model_eval[2]),np.linspace(model_eval[0],model_eval[1],model_eval[2]))
            plt.yticks(np.arange(model_eval[2]),np.linspace(model_eval[0],model_eval[1],model_eval[2])[::-1])
            plt.xlabel('$M_0$')
            plt.ylabel('$M_3$')
            plt.colorbar()
            plt.title('Term %s'%(kk+1))
            plt.show()

        plt.imshow(Y_true[::-1])
        plt.xticks(np.arange(model_eval[2]),np.linspace(model_eval[0],model_eval[1],model_eval[2]))
        plt.yticks(np.arange(model_eval[2]),np.linspace(model_eval[0],model_eval[1],model_eval[2])[::-1])
        plt.xlabel('$M_0$')
        plt.ylabel('$M_3$')
        plt.colorbar()
        plt.title('Full %s term model'%(nTruth))
        plt.show()
    
    if run_MCMC:
        Y_true = np.ravel(Y_true).reshape((model_eval[2]**nMoment),1)
        # Define the covariance for the likelihood distribution.
        Cov = obs_var*np.identity(model_eval[2]**nMoment)

        # ### Define the posterior distribution for the MCMC Hammer

        # Define the log prior. 
        # Uniform over the lower bound to upper bound cube with the added condition that b_{k}>b_{k+1}
        def lnprior(theta):
            lb_check = [param_lb[0]<theta[kk] for kk in range(nModel)]+[param_lb[1]<theta[kk+nModel] for kk in range(nModel)]
            ub_check = [theta[kk]<param_ub[0] for kk in range(nModel)]+[theta[kk+nModel]<param_ub[1] for kk in range(nModel)]
            b_decending_check = [theta[kk+nModel]>theta[kk+nModel+1] for kk in range(nModel-1)]
            check = lb_check+ub_check+b_decending_check
            if np.all(check)==True:
                return(0.0)
            else:
                return(-np.inf)

        # Define the log likelihood.
        # p( Obs | theta ) ~ Normal with mean = Obs and Cov in the function
        def lnlike(theta):
            M_theta = MCMC_model(theta).reshape((len(Y_true),1))
            expo = -0.5*(M_theta-Y_true).T@np.linalg.solve(Cov,(M_theta-Y_true))
            return(expo[0,0])

        def lnprob(theta):
            if lnprior(theta)==0.0:
                return(lnlike(theta))
            else:
                return(-np.inf)

        # ### Find initial walkers
        # * Uniformly sample the prior distribution
        # * Choose the most likely samples to initialize the hammer

        #  Set up for using MCMC Hammer:
        ndim , nwalkers = 2*nModel , 4*nModel

        post_samps = []
        post_val = []
        nSamps = int(1E5)

        lb = [param_lb[0] for kk in range(nModel)]+[param_lb[1] for kk in range(nModel)]
        ub = [param_ub[0] for kk in range(nModel)]+[param_ub[1] for kk in range(nModel)]

        for kk in range(nSamps):
            theta = np.random.uniform(lb,ub)
            index = np.argsort(theta[nModel:])+nModel
            theta[nModel:]=theta[index[::-1]]
            post_samps.append(theta)
            post_val.append(lnlike(theta))

        Index = np.argsort(post_val)[-nwalkers:]
        pos = np.array(post_samps)[Index,:]


        # ### Hammer time

        # Create sampler object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        # Run MCMC: (initial walkers, number of MCMC iterations)
        sampler.run_mcmc(pos,MCMC_Steps)


        # ### Remove burn in samples: burn_in period = 5 $\times$ max IACT
        # 
        # * NOTE: If the true IACT time is too large or the MCMC sampler did not converge to the posterior, then the IACT estimater will fail and through an error.
        # * If an error is thrown, I dismiss it and set the burn-in period to be 0.

        tmp = sampler.chain.reshape((MCMC_Steps*nwalkers,ndim))
        try:
            IACT = emcee.autocorr.integrated_time(tmp)
    #        print('IACT: ',IACT)
            Burn_In = int(5* np.max(IACT))
        except:
            if Plot_figs:
                print('The chain is too short to reliably estimate the autocorrelation time, Burn_In set to zero')
            IACT = np.inf
            Burn_In=0

        tmp = sampler.chain[:,Burn_In:,:]
        d1,d2,d3 = tmp.shape
        Samples = tmp.reshape(d1*d2,d3)


        # ### Triangle plot of posterior

        labels = ['$a_%s$'%(kk+1) for kk in range(nModel)] +  ['$b_%s$'%(kk+1) for kk in range(nModel)]
        if Plot_figs:
            corner.corner(Samples,labels=labels)
            plt.show()


    # ### Trajectories of posterior samples (blue) plotted against the true observation (orange)
    # * Note: What is plotted is the column stack of the model outputs and true observations
        RMSE = []
        if Plot_figs:
            nSamps = 100
            for kk in range(nSamps):
                theta = Samples[np.random.randint(len(Samples)),:]
                MCMC_theta = MCMC_model(theta).ravel()
                plt.plot(MCMC_theta,'C0')
                RMSE.append(np.sqrt(np.mean((MCMC_theta-np.ravel(Y_true))**2)))
            plt.plot(MCMC_theta,'C0',label='Trajectories of %s posterior samples'%(nSamps))
            plt.plot(Y_true,'C1',label='True observation',linewidth=2.5)
            if nMoment==2:
                plt.xlabel('Column stack of model outputs / observations')
            plt.title('%s Term Model Fitting a %s Term Truth'%(nModel,nTruth))
            plt.legend()
            plt.show()
            #print('\n RMSE: ',np.mean(RMSE),np.std(RMSE))
            #plt.hist(RMSE)
            #plt.show()

        
        # Calculate the maximized value of the likelihood function for AIC and BIC
        index = np.array(np.where(sampler.lnprobability == sampler.lnprobability.max()))[:,-1]
        theta = sampler.chain[index[0],index[1],:]
        X = np.ravel(MCMC_model(theta))
        rv = multivariate_normal(mean=np.ravel(Y_true),cov=Cov)
        lnL= np.log(rv.pdf(X))

        # Calculate BIC 
        # BIC = 2*ln( lnL )- k*ln(n) 
        #### lnL = log of the maximized value of the likelihood function
        #### k = number of model parameters
        #### n = number of data points
        BIC = ndim*np.log(model_eval[2]**nMoment)-2.0*lnL

        # Calculate AIC 
        # AIC = 2*ln( L )- 2*k
        AIC = 2*ndim-2.0*lnL

        IC = [AIC,BIC]
        return(Samples,IACT,IC,sampler)


def Model_selection(nTruth,nMoment,param_Truth,MCMC_Steps = int(1E5),param_lb = [-10.,-10.],
                        param_ub = [ 10., 10.],model_eval = [0.1 , 1., 10],obs_var = 0.01,
                       Plot_figs = False,N_Models = [1,2,3]):
    
    '''
    Inputs:
        nTruth: 
            Number of terms in the truth 
        nModel: 
            Number of terms in the model trying to estimate the truth
        param_Truth: 
            True parameters for model with nTruth terms
            The format must be:  [ a_1, a_2, ... , a_nTruth, b_1, b_2, ..., b_nTruth]
            with b_1 > b_2 >  ... > b_nTruth
        MCMC_Steps = int(5E4):
            Number of steps per MCMC Hammer walker
        param_lb = [-10.,-10.]: 
        param_ub = [ 10., 10.]: 
            Define lower and upper bounds for parameters [ a's , b's]
            We are assuming the bounds are the same for all a's and for all b's    
        model_eval = [0.1 , 1., 10]: 
            [ lb, ub, nEval]
            Define where the true model and approximate model will be evaluate
        obs_var = 0.1: 
            Observation error for likelihood distribution
        Plot_figs = False:
            Option to create figures
        N_Models = [1,2,3]:
            The number of terms in each model being considered for model selection, i.e., [1,2,3] implies we will consider models with one term, two terms, and three terms and select the ``best'' model from these three options.
    
    Outputs:
        Final_Results:
            This is a class which has the following:
                Final_Results.input_params: the parameters that were used to select this model
                Final_Results.selected_model: string stating the selected model
                Final_Results.Samples: Samples from param_estimation
                Final_Results.IACT: IACT from param_estimation
                Final_Results.IC: IC from param_estimation
                Final_Results.sampler: sampler from param_estimation

        All_Results:
            A list of Results classes, each with the param_estimation results. There is one class for each model that was considered in the model selection.
    '''    
    

    class Results:
        def __init__(self,selected_model,Samples,IACT,IC,sampler,input_params):
            self.selected_model = selected_model
            self.Samples = Samples
            self.IACT = IACT
            self.IC = IC
            self.sampler = sampler
            self.input_params = input_params

    All_Results = [Results([],[],[],[],[],[]) for kk in range(len(N_Models))]
    
    IC_check = False
    iter_count = 0

    while not(IC_check):
        iter_count+=1
        if iter_count > 4: 
            print('Model selection failed: consider changing input parameters.\nReturng last iteration.')
            return(All_Res)
        All_IC = []

        for kk,nModel in enumerate(N_Models):
            if Plot_figs:
                print('\n%s Term Model Fitting a %s Term Truth'%(nModel,nTruth))
            Samples,IACT,IC,sampler = param_estimation(nTruth,nModel,nMoment,param_Truth,MCMC_Steps,param_lb,
                        param_ub,model_eval,obs_var,Plot_figs)
            
            Results = All_Results[kk]
            
            Results.Samples = Samples
            Results.IACT = IACT
            Results.IC = IC
            Results.sampler = sampler
            
            All_IC.append(IC)
            
            if Plot_figs:
                print('AIC = ', IC[0])
                print('BIC = ', IC[1])
                print('--------------------------------------------------------')
                print('--------------------------------------------------------')

        model_check = [np.argmin(np.array(All_IC)[:,kk]) for kk in range(2)]
        # Turn IACT and IC check into one check
        if np.all(model_check == model_check[0]): 
            IC_check = True
        else:
            MCMC_Steps = int(10*MCMC_Steps)
            print('-------------------------------------------------------------------------------------------\n')
            print('-------------------------------------------------------------------------------------------\n')
            print('-------------------------------------------------------------------------------------------\n')
            print('Need to rerun model selection.\nIncreasing chain length to %s and rerunning MCMC'%(MCMC_Steps))
            print('-------------------------------------------------------------------------------------------\n')
            print('-------------------------------------------------------------------------------------------\n')
            print('-------------------------------------------------------------------------------------------\n')

    Final_Results = All_Results[model_check[0]]
    
    Final_Results.input_params = [['nTruth',nTruth],['N_Models',N_Models],['nMoment',nMoment],
                         ['param_Truth',param_Truth],['MCMC_Steps',MCMC_Steps],
                         ['param_lb',param_lb],['param_ub',param_ub],
                         ['model_eval',model_eval],['obs_var',obs_var]]
    
    Final_Results.selected_model = '%s term model selected'%(int(Final_Results.Samples.shape[1]/2))
    
    print(Final_Results.selected_model)
    if np.any(Final_Results.IACT== np.inf):
        print('**********************************************************************'+
              '**********************************************************************'+
              '** WARNING: Unable to estimate IACT. Consider increasing MCMC_Steps **'+
              '**********************************************************************'+
              '**********************************************************************')
    #return(Results)
    return(Final_Results,All_Results)



