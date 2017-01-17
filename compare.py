#!/usr/bin/env python
"""
------------------------------------------------------------------------
  class for running a set of statistical comaparative tests and creating 
  plots on two groups of data
  
  Tries to "intelligently" handle regions of nan.
------------
Borrows code snippets from numerous sources on Internet
------------------------------------------------------------------------
"""
import glob
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from enum import Enum    
import unittest
import logging
import scipy.stats as stats


logging.basicConfig(level=logging.DEBUG)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''

#------------------
class Compare:
    """   class for running a set of statistical comaparative tests and creating 
    plots on two groups of data
    Tries to "intelligently" handle regions of nan.    
    """
    
    #-------------------------------
    def __init__(s, X_, Y_, label_='data', x_label_='X', y_label_='Y'):
        s.X = np.array(X_)
        s.Y = np.array(Y_)
        s.x_label = x_label_
        s.y_label = y_label_
        s.label = label_
        
    #-------------------------------
    def plot_hists(s):
        """ plot histograms of data
        """
        # VIEW DATA
        #  generate two groups of data and display
        from scipy import stats
        import numpy as np
        import matplotlib.pyplot as plt
        #%matplotlib inline
        
        np.random.seed(7)
        A = s.X
        B = s.Y
        C = np.concatenate((A,B)) # Combined = groups A and B
        
        # fit a normal curve to each set
        (mu_A, sigma_A) = stats.norm.fit(A)
        (mu_B, sigma_B) = stats.norm.fit(B)
        (mu_C, sigma_C) = stats.norm.fit(C)
        
        #-----------------
        plt.figure(figsize=(5,10))
        
        #---------------------
        
        ax1 = plt.subplot(511)
        
        #histogram - pooled
        n_C, bins_C, patches_C = plt.hist(C, bins=15, normed=1, facecolor='gray', alpha=0.75)
        
        # 'best fit' line
        y = stats.norm.pdf(bins_C, mu_C, sigma_C)
        l = plt.plot(bins_C, y, 'r--', linewidth=2, label='norm fit')
        plt.title('pooled ' + s.label + ' Histogram')
        plt.ylabel('frequency')
        plt.xlabel(s.label + ' value')
        plt.legend()
        
        # add a kde line
        '''
        gkde_C = gaussian_kde(C)
        plt.plot(bins_C,gkde_C(bins_C), 'b--', linewidth=2, label='Combined KDE')
        '''
        
        #---------------------
        #sample A
        plt.subplot(512, sharex=ax1, sharey=ax1)
        # histogram
        n, bins, patches = plt.hist(A, bins=15, normed=1, facecolor='blue', alpha=0.75)
        
        # 'best fit' line
        y = stats.norm.pdf(bins_C, mu_A, sigma_A)
        l = plt.plot(bins_C, y, 'r--', linewidth=2, label='norm fit')
        
        plt.title(s.x_label + ' ' + s.label + ' Histogram')
        plt.ylabel('frequency')
        plt.xlabel(s.label + ' value')
        plt.legend()
        
        #---------------------
        #sample B
        plt.subplot(513, sharex=ax1, sharey=ax1)
        # histogram
        n, bins, patches = plt.hist(B, bins=15, normed=1, facecolor='green', alpha=0.75)
        
        # 'best fit' line
        y = stats.norm.pdf(bins_C, mu_B, sigma_B)
        l = plt.plot(bins_C, y, 'r--', linewidth=2, label='norm fit')
        
        plt.title(s.y_label + ' ' + s.label + ' Histogram')
        plt.ylabel('frequency')
        plt.xlabel(s.label + ' value')
        plt.legend()
        
        #---------------------
        # boxplots
        plt.subplot(5,1,4)
        plt.boxplot([B,A], 0, 'rs', 0)
        plt.yticks([1, 2], [s.y_label, s.x_label])
        plt.title('Boxplot of ' + s.label )
        plt.xlabel(s.label + ' value')
        plt.grid(b=True)
        
        #---------------------
        #1-D scatter
        plt.subplot(5,1,5)
        plt.scatter(A,np.ones_like(A), marker='.', color='b')
        plt.grid(1)
        
        plt.scatter(B, np.zeros_like(B), marker='.', color='g')
        plt.yticks([0, 1], [s.y_label, s.x_label])
        plt.title('1-D Scatter of ' + s.label)
        plt.xlabel(s.label + ' value')
        
        plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        #plt.show()
       
    #-------------------------------
    def plot_t_likelihood(s):
        """ Using analytical formulas for Gaussian assumptions (Welch's
        approximate t), plot likelihood of sample means.
        
        High Density Intervals (HDI) are also shown for 95% region.
        """
        
        A = s.X
        B = s.Y
        
        # we only consider mus in the middle histogram range
        mu_min = np.amin(np.concatenate((s.X, s.Y)))
        mu_max = np.amax(np.concatenate((s.X, s.Y)))
        mus = np.linspace(mu_min, mu_max, 200)
        
        sigma_mus_A = np.sqrt(A.var(ddof=1))/len(A)
        p_mus_A = stats.t.pdf(mus, loc=A.mean(), scale=np.sqrt(sigma_mus_A), df = len(A)-1)
        
        fig = plt.figure(figsize=(10,10))
        st = fig.suptitle(s.label + ' - Welch\'s approximate t likelihood', fontsize="x-large")        

        plt.subplot(311)
        plt.plot(mus, p_mus_A, color='b')
        plt.title('P(mu_A| sampleA)')
        plt.xlabel('mu')
        plt.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=A.mean(), scale=np.sqrt(sigma_mus_A), df=len(A)-1)
        HDI_high = stats.t.ppf(.975, loc=A.mean(), scale=np.sqrt(sigma_mus_A), df=len(A)-1)
        HDI = [HDI_low, HDI_high]
        plt.plot( HDI, [0,0],lw=5.0, color='k')
        
        plt.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( sum(HDI)/2, 0.3, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )

        #-------
        
        plt.subplot(312)
        sigma_mus_B = np.sqrt(B.var(ddof=1))/len(B)
        p_mus_B = stats.t.pdf(mus, loc=B.mean(), scale=np.sqrt(sigma_mus_B), df=len(B)-1)
        plt.plot(mus, p_mus_B, color='g')
        plt.title('P(mu_B|sampleB)')
        plt.xlabel('mu')
        plt.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=B.mean(), scale=np.sqrt(sigma_mus_B),df=len(B)-1)
        HDI_high = stats.t.ppf(.975, loc=B.mean(), scale=np.sqrt(sigma_mus_B),df=len(B)-1)
        HDI = [HDI_low, HDI_high]
        plt.plot( HDI, [0,0],lw=5.0, color='k')
        
        plt.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( sum(HDI)/2, 0.3, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )

        #---------------
        plt.subplot(313)
        mu_diff = A.mean()-B.mean()
        
        df_diff_num = ( A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B) )**2
        df_diff_den = ( (A.var(ddof=1)/len(A))**2/(len(A)-1) + 
                        (B.var(ddof=1)/len(B))**2/(len(B)-1) )
        df_diff = np.floor(df_diff_num/df_diff_den)
        
        sigma_diff = A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B)
        
        p_mus_diff = stats.t.pdf(mus, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        plt.plot(mus, p_mus_diff, color='r')
        plt.title('P(mu_A - mu_B| samples A and B)')
        plt.xlabel('mu')
        plt.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        HDI_high = stats.t.ppf(.975, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        HDI = [HDI_low, HDI_high]
        plt.plot( HDI, [0,0],lw=5.0, color='k')
        
        plt.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        plt.text( sum(HDI)/2, 0.2, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        
        plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)        
        #plt.show()

    #-------------------------------
    def plot_qq(s):
        """ the sizes of each group should be about the same. Does not use 
        true quantiles, instead uses sorting and plots on a one to one basis. 
        Should probably replace one to one with binning of some sort when 
        sample sizes are different."""

        plt.figure()
        sort_A = s.X.copy()
        sort_A.sort()
        sort_B = s.X.copy()
        sort_B.sort()
        # TODO rethink n_min
        n_min = np.minimum(len(sort_A),len(sort_B))
        plt.scatter(sort_A[:n_min], sort_B[:n_min], color='g')
        plt.title('QQ plot - ' + s.x_label + ' to ' + s.y_label)
        plt.xlabel(s.x_label + ' quantiles')
        plt.ylabel(s.y_label + ' quantiles')
        plt.grid()
        
        
        A_mat = np.vstack([sort_A[:n_min], np.ones(n_min)]).T
        sort_A[:n_min], sort_B[:n_min]
        slope, intercept = np.linalg.lstsq(A_mat, sort_B[:n_min])[0]
        plt.plot(sort_A,sort_A*slope + intercept, 'r', linewidth=1)
        #plt.show()        
        
    #-------------------------------
    def calc_stats(s, print_=False):
        """ returns averages, t-test_p, Mann-Whitney test_p, Cohens_d """
        
        t,t_test_p = stats.ttest_ind(s.X, s.Y, axis=0, equal_var=False)
        mw,mw_p = stats.mannwhitneyu(s.X, s.Y)
        
        n_tot = len(s.X) + len(s.Y)
        x_var = s.X.var(ddof=0)
        y_var = s.Y.var(ddof=0)
        pooled_var = (len(s.X)*x_var + len(s.Y)*y_var) / n_tot
        cohens_d = (s.X.mean() - s.Y.mean()) / np.sqrt(pooled_var)

        if(print_):
            print(s.x_label + ' ave: ', s.X.mean())
            print(s.y_label + ' ave: ', s.Y.mean())
            print('t-test p_value: ',t_test_p)
            print('mw-test p_value: ',mw_p)
            print('Cohens d: ', cohens_d)        

        return s.X.mean(), s.Y.mean(), t_test_p, mw_p, cohens_d

#------------------------------------------------------------------------
def example():
    """
    
    """
    # generate two different normal samples
    A = np.random.normal(loc=0.75, scale=1.5, size=50)
    B = np.random.normal(loc=0.0, scale=1.0, size=50)    
    compare = Compare(A,B, label_='smile', x_label_='TRUTH', y_label_='BLUFF')
    compare.plot_hists()
    compare.plot_qq()
    compare.plot_t_likelihood()
    compare.calc_stats(print_=True)
    plt.show()

#=============================================================================
if __name__ == '__main__':
    print('running main')
    example()