import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

class Evidence_model_selection:
    '''
    nTruth: 
        Number of terms in the truth 
    nMoment: 
        Number of moments in each model term, either 1 or 2
    param_Truth: 
        True parameters for model with nTruth terms
        The format must be:  [ a_1, a_2, ... , a_nTruth, b_1, b_2, ..., b_nTruth]
        with b_1 > b_2 >  ... > b_nTruth
    param_lb = [-10.,-10.]: 
    param_ub = [ 10., 10.]: 
        Define lower and upper bounds for parameters [ a's , b's]
        We are assuming the bounds are the same for all a's and for all b's    
    model_eval = [0.1 , 1., 10]: 
        [ lb, ub, nEval]
        Define where the true model and approximate model will be evaluate
    obs_var = 0.1: 
        Observation error for likelihood distribution
    ntemps = 75:
        Number of temperatures for the Parallel transport ensemble sampler.
        Also the number of temperatures used estimate the model evidence
    '''
    
    def __init__(self,nTruth,nMoment,param_Truth,param_lb = [-10.,-10.],param_ub = [ 10., 10.],
                model_eval = [0.1 , 1., 10],obs_var = 0.1,ntemps = 75):
        self.nTruth = nTruth
        self.nMoment = nMoment
        assert nMoment in [1,2], \
            'nMoment must equal one or two'
        if nMoment == 1:
            self.Model = self.one_moment_model
        else:
            self.Model = self.two_moment_model
        self.param_Truth = param_Truth
        self.param_lb = param_lb
        self.param_ub = param_ub
        self.model_eval = model_eval
        if self.nMoment == 1:
            self.M0_eval = np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2])
        if self.nMoment == 2:
            M0 = np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2])
            M3 = np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2])
            self.M0_eval,self.M3_eval = np.meshgrid(M0,M3)
        self.obs_var = obs_var
        assert ntemps > 9, \
            'There should be at least 10 temperatures to estimate the model evidence.'
        self.ntemps = ntemps
        
        
        
    def one_moment_model(self,a,b,M0):
        return(a*M0**(b))
    def two_moment_model(self,a,b,M0,M3):
        return(a*M0**(b)*M3**(1-b))
    
    
    def MCMC_model(self,theta,nTerms):
        if self.nMoment == 1:
            P_val = [self.one_moment_model(theta[kk],theta[kk+nTerms],self.M0_eval) for kk in range(nTerms)]
            return(np.sum(np.array(P_val),axis=0))
        if self.nMoment == 2:
            P_val = [self.two_moment_model(theta[kk],theta[kk+nTerms],self.M0_eval,self.M3_eval) for kk in range(nTerms)]
            return(np.sum(np.array(P_val),axis=0))
        
    def get_data(self):
        self.data = self.MCMC_model(self.param_Truth,self.nTruth)
    
    def plot_truth(self):
        # Plot figs of truth    
        if self.nMoment == 1:
            # Plot the first term, second term, and their sum
            for kk in range(self.nTruth):
                cancel_out = [mm == kk for mm in range(self.nTruth)]+[1 for mm in range(self.nTruth)]
                plt.plot(self.M0_eval,self.MCMC_model(np.array(self.param_Truth)*cancel_out,self.nTruth),
                         label='$%.2fM^{%.2f}$'%(self.param_Truth[kk],self.param_Truth[kk+self.nTruth]))
            plt.plot(self.M0_eval,self.data,label='True Observations',linewidth=5)
            plt.xlabel('$M$')
            plt.title('Full %s term model'%(self.nTruth))
            plt.legend()
            plt.show()

        if self.nMoment == 2:
            for kk in range(self.nTruth):
                cancel_out = [mm == kk for mm in range(self.nTruth)]+[1 for mm in range(self.nTruth)]
                plt.imshow(self.MCMC_model(np.array(self.param_Truth)*cancel_out,self.nTruth)[::-1])
                #plt.xticks(np.arange(self.model_eval[2]),np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2]))
                #plt.yticks(np.arange(self.model_eval[2]),np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2])[::-1])
                plt.xlabel('$M_0$')
                plt.ylabel('$M_3$')
                plt.colorbar()
                plt.title('Term %s'%(kk+1))
                plt.show()

            plt.imshow(self.data[::-1])
            #plt.xticks(np.arange(self.model_eval[2]),np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2]))
            #plt.yticks(np.arange(self.model_eval[2]),np.linspace(self.model_eval[0],self.model_eval[1],self.model_eval[2])[::-1])
            plt.xlabel('$M_0$')
            plt.ylabel('$M_3$')
            plt.colorbar()
            plt.title('Full %s term model'%(self.nTruth))
            plt.show()    
    
    def run_MCMC(self,nTerms,MCMC_Steps = int(2E4)):
        Y_true = np.ravel(self.data).reshape((self.model_eval[2]**self.nMoment),1)
        # Define the covariance for the likelihood distribution.
        Cov = self.obs_var*np.identity(self.model_eval[2]**self.nMoment)

        def lnprior(theta):
            # Define the log prior. 
            # Uniform over the lower bound to upper bound cube with the added condition that b_{k}>b_{k+1}
            lb_check = [self.param_lb[0]<theta[kk] for kk in range(nTerms)]+[self.param_lb[1]<theta[kk+nTerms] for kk in range(nTerms)]
            ub_check = [theta[kk]<self.param_ub[0] for kk in range(nTerms)]+[theta[kk+nTerms]<self.param_ub[1] for kk in range(nTerms)]
            b_decending_check = [theta[kk+nTerms]>theta[kk+nTerms+1] for kk in range(nTerms-1)]
            check = lb_check+ub_check+b_decending_check
            if np.all(check)==True:
                nb_params = len(theta)/2
                factorial = np.prod(np.arange(1,nb_params+1))
                normalizing_factor = (self.param_ub[0]-self.param_lb[0])**(nb_params)*(self.param_ub[1]-self.param_lb[1])**(nb_params)/factorial
                return(-np.log(normalizing_factor))
            else:
                return(-np.inf)

        def lnlike(theta):
            # Define the log likelihood.
            # p( Obs | theta ) ~ Normal with mean = Obs and Cov in the function
            dim = len(Y_true) 
            M_theta = self.MCMC_model(theta,nTerms).reshape((dim,1))
            expo = -0.5*(M_theta-Y_true).T@np.linalg.solve(Cov,(M_theta-Y_true))
            normalizing_factor = (2*np.pi)**(dim/2)*(np.prod(np.diag(Cov)))**(0.5)
            return(expo[0,0]-np.log(normalizing_factor))

        ndim , nwalkers = 2*nTerms , 4*nTerms
                
        try: 
            pos0=np.zeros((self.ntemps,nwalkers,ndim))
            for kk in range(self.ntemps):
                pos0[kk,:,:] = self.MCMC_pos
        except AttributeError:
            
            post_samps = []
            post_val = []
            nSamps = int(1E5)

            lb = [self.param_lb[0] for kk in range(nTerms)]+[self.param_lb[1] for kk in range(nTerms)]
            ub = [self.param_ub[0] for kk in range(nTerms)]+[self.param_ub[1] for kk in range(nTerms)]

            for kk in range(nSamps):
                theta = np.random.uniform(lb,ub)
                index = np.argsort(theta[nTerms:])+nTerms
                theta[nTerms:]=theta[index[::-1]]
                post_samps.append(theta)
                post_val.append(lnlike(theta))

            Index = np.argsort(post_val)[-nwalkers:]
            pos = np.array(post_samps)[Index,:]

            pos0=np.zeros((self.ntemps,nwalkers,ndim))
            for kk in range(self.ntemps):
                pos0[kk,:,:] = pos
            self.MCMC_pos = pos

        # Create sampler object
        betas = np.logspace(0, -6, self.ntemps, base=10)
        sampler = emcee.PTSampler(ntemps=self.ntemps,nwalkers=nwalkers,dim = ndim,
                                  logl=lnlike,logp=lnprior,betas=betas)
        
        # Run MCMC: (initial walkers, number of MCMC iterations)
        sampler.run_mcmc(pos0=pos0,N=MCMC_Steps)
        return(sampler)
        
    def plot_sampler(self,sampler,nTerms):
        labels = ['$a_%s$'%(kk+1) for kk in range(nTerms)] +  ['$b_%s$'%(kk+1) for kk in range(nTerms)]

        _,d1,d2,d3 = sampler.chain.shape
        Samples = sampler.chain[0,:,:,:].reshape((d1*d2,d3))
        corner.corner(Samples,labels=labels)
        plt.show()

        nSamps = 100
        for kk in range(nSamps):
            theta = Samples[np.random.randint(len(Samples)),:]
            MCMC_theta = self.MCMC_model(theta,nTerms).ravel()
            plt.plot(MCMC_theta,'C0')

        plt.plot(MCMC_theta,'C0',label='Trajectories of %s posterior samples'%(nSamps))
        plt.plot(np.ravel(self.data),'C1',label='True observation',linewidth=2.5)
        if self.nMoment==2:
            plt.xlabel('Column stack of model outputs / observations')
        plt.title('%s Term Model Fitting a %s Term Truth'%(nTerms,self.nTruth))
        plt.legend()
        plt.show()
    
    def model_selection(self,Models=[1,2,3],dlnZ_tol = 1,MCMC_Steps = int(2E3)):
        Results = []
        Evidence = np.zeros((2,len(Models)))
        for kk,nTerms in enumerate(Models):
            dlnZ = np.inf
            iter_count = 0
            while dlnZ >  dlnZ_tol:
                iter_count += 1
                sampler = self.run_MCMC(nTerms,MCMC_Steps = MCMC_Steps)
                lnZ , dlnZ = sampler.thermodynamic_integration_log_evidence()
                if iter_count > 5:
                    print('\nThermodynamic integration is not converging within the set tolerance.')
                    print('Consider increasing MCMC_Steps.')
                    print('Current log evidence and estimation error: lnZ = ',lnZ, ', dlnZ = ',dlnZ)
                    return(None)
                if dlnZ > dlnZ_tol:
                    self.ntemps += 50
                    print('Model',nTerms,' has error > tolerance, increasing the number of temperatures to ',self.ntemps)
                    print('      Current log evidence and estimation error: lnZ = ',lnZ, ', dlnZ = ',dlnZ)
            Results.append(sampler)
            Evidence[:,kk] = lnZ,dlnZ
            delattr(self,'MCMC_pos')
           
        print('\nResults:')
        for kk,nTerms in enumerate(Models):
            print('Model',Models[kk],'with  lnZ = ',Evidence[0,kk], ', dlnZ = ',Evidence[1,kk])
        
        self.model_selection_results = Results
        best_model = np.argmax(Evidence[0,:])

        tolerance_check = np.array([False,False,False])
        
        for kk,nTerms in enumerate([1,2,3]):
            if kk == best_model:
                tolerance_check[kk] = True
            else:
                if Evidence[0,kk]+ Evidence[1,kk]< Evidence[0,best_model]-Evidence[1,best_model]:
                    tolerance_check[kk] = True


        if np.any(tolerance_check == False):
            print('\n====================')
            print(  '===== WARNING ======')
            print(  '====================\n')
            print('The model selection is inconclusive because the estimated')
            print('evidence of multiple models are within the errors of each other.\n')
            
        else:
            print('The selected model is Model',Models[best_model])
        self.selected_model = Models[best_model]
            
            
            
            
            