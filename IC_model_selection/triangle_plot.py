import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def triplt(samples,
           cmap_hist=cm.gray_r,
           linewidth= 2,
           fill_contours = True, 
           nLevels = 6, 
           bounds = None, 
           labels = None,
           truth = None, 
           truth_color='b',
           plot_mean = False,
           mean_color='r'):
    '''
    Input:
    samples: samples of the distribution with shape (nSamps,nDim)
    cmap_hit: matplotlib colormap for the histogram
    linewidth: The width of the 1D histogram line as well as the true/mean values
    fill_contours: If False, only plots contour lines.
    nLevels: The number of contours. If None, then matplotlib chooses.
    bounds: The bounds for each dimension as a list of lists, i.e., [[lb_0,ub_0],[lb_1,ub_1],...,[lb_nDim,ub_nDim]]
    labels: The labels for each dimension as a list of strings.
    truth: The true values for each dimension in a list.
    truth_color: The color of the truth line.
    plot_mean: Do you want the sample mean plotted?
    mean_color: The color of the sample mean line.
    '''
    
    nSamps,nDim = samples.shape
    if bounds == None:
        bounds = [[np.min(samples[:,kk]),np.max(samples[:,kk])] for kk in range(nDim)]
        
    if labels == None:
        labels = ['' for kk in range(nDim)]
    fig, ax = plt.subplots(nrows=nDim, ncols=nDim, figsize=(nDim**2, nDim**2))

    for kk in range(0,nDim):
        for jj in range(kk+1,nDim):
            ax[kk,jj].axis('off')
        ax[kk,kk].set_xlim(bounds[kk])
        ax[kk,kk].hist(samples[:,kk],histtype=u'step', normed=True,linewidth = linewidth,color='k')
        
        ylims = ax[kk,kk].get_ylim()
        if truth!=None:
            ax[kk,kk].plot([truth[kk] for ii in range(2)],ylims,linewidth=linewidth, color = truth_color)
        if plot_mean==True:
            ax[kk,kk].plot([np.mean(samples[:,kk]) for ii in range(2)],ylims,linewidth=linewidth, color = mean_color)
        
        ax[kk,kk].set_yticks([])
        if kk != nDim-1:
            ax[kk,kk].set_xticks([])

        for jj in range(kk):
            counts,xbins,ybins = np.histogram2d(samples[:,jj],samples[:,kk],bins=20,range=[bounds[jj],bounds[kk]],normed = True)
            Levels = np.linspace(0,np.max(counts),nLevels+1)
            if fill_contours:
                ax[kk,jj].contourf(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap=cmap_hist,levels =Levels )
            else:
                ax[kk,jj].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=linewidth,cmap=cmap_hist,levels =Levels)
            ax[kk,jj].set_xlim(bounds[jj])
            ax[kk,jj].set_ylim(bounds[kk])
            
            if truth!=None:
                ax[kk,jj].plot(bounds[jj],[truth[kk] for ii in range(2)],linewidth=linewidth, color = truth_color)
                ax[kk,jj].plot([truth[jj] for ii in range(2)],bounds[kk],linewidth=linewidth, color = truth_color)
            if plot_mean==True:
                ax[kk,jj].plot(bounds[jj],[np.mean(samples[:,kk]) for ii in range(2)],linewidth=linewidth, color = mean_color)
                ax[kk,jj].plot([np.mean(samples[:,jj]) for ii in range(2)],bounds[kk],linewidth=linewidth, color = mean_color)


            ax[kk,kk].set_yticks([])
            if kk != nDim-1:
                ax[kk,kk].set_xticks([])
            
            if jj != 0:
                ax[kk,jj].set_yticks([])
            if kk != nDim-1:
                ax[kk,jj].set_xticks([])
    for jj in range(nDim):
        ax[nDim-1,jj].set_xlabel(labels[jj])
    for jj in range(1,nDim):
        ax[jj,0].set_ylabel(labels[jj])