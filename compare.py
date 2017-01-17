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
    def plot_all(s,fig_=None, fname_=None):
        """ plot histograms of data
        """


        A = s.X
        B = s.Y
        C = np.concatenate((A,B)) # Combined = groups A and B
        
        # fit a normal curve to each set
        (mu_A, sigma_A) = stats.norm.fit(A)
        (mu_B, sigma_B) = stats.norm.fit(B)
        (mu_C, sigma_C) = stats.norm.fit(C)
        
        #-----------------
        if(fig_):
            fig = fig_
        else:
            fig = plt.figure(figsize=(8,12.5))
        st = fig.suptitle(s.label + ' data', fontsize="x-large")     
        
        #---------------------
        
        ax1 = fig.add_subplot(621)
        
        #histogram - pooled
        n_C, bins_C, patches_C = plt.hist(C, bins=15, normed=1, facecolor='gray', alpha=0.75)
        
        # 'best fit' line
        y = stats.norm.pdf(bins_C, mu_C, sigma_C)
        l = ax1.plot(bins_C, y, 'r--', linewidth=2, label='norm fit')
        ax1.set_title('pooled ' + s.label + ' Histogram')
        ax1.set_ylabel('frequency')
        ax1.set_xlabel(s.label + ' value')
        ax1.legend()
        
        # add a kde line
        '''
        gkde_C = gaussian_kde(C)
        plt.plot(bins_C,gkde_C(bins_C), 'b--', linewidth=2, label='Combined KDE')
        '''
        
        #---------------------
        #sample A
        plt.subplot(623, sharex=ax1, sharey=ax1)
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
        plt.subplot(625, sharex=ax1, sharey=ax1)
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
        plt.subplot(6,2,7)
        plt.boxplot([B,A], 0, 'rs', 0)
        plt.yticks([1, 2], [s.y_label, s.x_label])
        plt.title('Boxplot of ' + s.label )
        plt.xlabel(s.label + ' value')
        plt.grid(b=True)
        
        #---------------------
        #1-D scatter
        plt.subplot(6,2,9)
        plt.scatter(A,np.ones_like(A), marker='.', color='b')
        plt.grid(1)
        
        plt.scatter(B, np.zeros_like(B), marker='.', color='g')
        plt.yticks([0, 1], [s.y_label, s.x_label])
        plt.title('1-D Scatter of ' + s.label)
        plt.xlabel(s.label + ' value')
        
        ax11 = fig.add_subplot(6,2,11)
        s.plot_qq(ax11)

        #---------------------
        ax2 = fig.add_subplot(6,2,2)
        ax4 = fig.add_subplot(6,2,4)
        ax6 = fig.add_subplot(6,2,6)
        s.plot_t_likelihood(ax2, ax4, ax6)
        
        s.calc_stats('fname',fig)
        
        plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)   

        return fig
       
    #-------------------------------
    def plot_t_likelihood(s,ax1_,ax2_,ax3_):
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
        
        #fig = plt.figure(figsize=(10,10))
        #st = fig.suptitle(s.label + ' - Welch\'s approximate t likelihood', fontsize="x-large")        

        ax1_.plot(mus, p_mus_A, color='b')
        ax1_.set_title('P(' + s.x_label + '$\mu$ - | data)')
        ax1_.set_xlabel('mu')
        ax1_.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=A.mean(), scale=np.sqrt(sigma_mus_A), df=len(A)-1)
        HDI_high = stats.t.ppf(.975, loc=A.mean(), scale=np.sqrt(sigma_mus_A), df=len(A)-1)
        HDI = [HDI_low, HDI_high]
        ax1_.plot( HDI, [0,0],lw=5.0, color='k')
        
        ax1_.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax1_.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax1_.text( sum(HDI)/2, 0.4, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )

        #-------
        
        #plt.subplot(312)
        sigma_mus_B = np.sqrt(B.var(ddof=1))/len(B)
        p_mus_B = stats.t.pdf(mus, loc=B.mean(), scale=np.sqrt(sigma_mus_B), df=len(B)-1)
        ax2_.plot(mus, p_mus_B, color='g')
        ax2_.set_title('P(' + s.y_label + '$\mu$ - | data)')
        ax2_.set_xlabel('mu')
        ax2_.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=B.mean(), scale=np.sqrt(sigma_mus_B),df=len(B)-1)
        HDI_high = stats.t.ppf(.975, loc=B.mean(), scale=np.sqrt(sigma_mus_B),df=len(B)-1)
        HDI = [HDI_low, HDI_high]
        ax2_.plot( HDI, [0,0],lw=5.0, color='k')
        
        ax2_.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax2_.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax2_.text( sum(HDI)/2, 0.4, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )

        #---------------
        #plt.subplot(313)
        mu_diff = A.mean()-B.mean()
        
        df_diff_num = ( A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B) )**2
        df_diff_den = ( (A.var(ddof=1)/len(A))**2/(len(A)-1) + 
                        (B.var(ddof=1)/len(B))**2/(len(B)-1) )
        df_diff = np.floor(df_diff_num/df_diff_den)
        
        sigma_diff = A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B)
        
        p_mus_diff = stats.t.pdf(mus, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        ax3_.plot(mus, p_mus_diff, color='r')
        ax3_.set_title('P(' + s.x_label + '$\mu$ - ' + s.y_label + '$\mu$| all data)')
        ax3_.set_xlabel('mu')
        ax3_.grid(1)
        
        # add the HDI
        HDI_low = stats.t.ppf(.025, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        HDI_high = stats.t.ppf(.975, loc=(mu_diff), scale=np.sqrt(sigma_diff), df=df_diff)
        HDI = [HDI_low, HDI_high]
        ax3_.plot( HDI, [0,0],lw=5.0, color='k')
        
        ax3_.text( HDI[0], 0.05, '%.3g'%HDI[0],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax3_.text( HDI[1], 0.05, '%.3g'%HDI[1],
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        ax3_.text( sum(HDI)/2, 0.4, '95% HDI',
                  horizontalalignment='center',
                  verticalalignment='bottom',
                  )
        
        #plt.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        #st.set_y(0.95)
        #fig.subplots_adjust(top=0.85)        
        #plt.show()

    #-------------------------------
    def plot_qq(s,ax_=None):
        """ the sizes of each group should be about the same. Does not use 
        true quantiles, instead uses sorting and plots on a one to one basis. 
        Should probably replace one to one with binning of some sort when 
        sample sizes are different."""
        
        if(ax_):
            ax = ax_
        else:
            f = plt.figure()
            ax = plt.subplot(111)

        sort_A = s.X.copy()
        sort_A.sort()
        sort_B = s.X.copy()
        sort_B.sort()
        # TODO rethink n_min
        n_min = np.minimum(len(sort_A),len(sort_B))
        ax.scatter(sort_A[:n_min], sort_B[:n_min], color='g')
        ax.set_title('QQ plot - ' + s.x_label + ' to ' + s.y_label)
        ax.set_xlabel(s.x_label + ' quantiles')
        ax.set_ylabel(s.y_label + ' quantiles')
        ax.grid()
        
        # overlay linear regression fit
        A_mat = np.vstack([sort_A[:n_min], np.ones(n_min)]).T
        sort_A[:n_min], sort_B[:n_min]
        slope, intercept = np.linalg.lstsq(A_mat, sort_B[:n_min])[0]
        ax.plot(sort_A,sort_A*slope + intercept, 'r', linewidth=1)
  
    
    #-------------------------------
    def calc_stats(s, print_=False, fig_=None):
        """ returns averages, t-test_p, Mann-Whitney test_p, Cohens_d """
        
        t,t_test_p = stats.ttest_ind(s.X, s.Y, axis=0, equal_var=False)
        mw,mw_p = stats.mannwhitneyu(s.X, s.Y)
        
        n_tot = len(s.X) + len(s.Y)
        x_var = s.X.var(ddof=0)
        y_var = s.Y.var(ddof=0)
        pooled_var = (len(s.X)*x_var + len(s.Y)*y_var) / n_tot
        cohens_d = (s.X.mean() - s.Y.mean()) / np.sqrt(pooled_var)

        if print_:
            print(s.x_label + ' ave: ', s.X.mean())
            print(s.y_label + ' ave: ', s.Y.mean())
            print('t-test p_value: ',t_test_p)
            print('mw-test p_value: ',mw_p)
            print('Cohens d: ', cohens_d)        
        
        if fig_:
            fig_.text( .6, .4, s.x_label +' mean: ' + '%.3g'%s.X.mean(),
                       horizontalalignment='left',
                       verticalalignment='bottom',
                       )
            fig_.text( .6, .375, s.y_label + ' mean: ' + '%.3g'%s.Y.mean(),
                       horizontalalignment='left',
                       verticalalignment='bottom',
                       )
            fig_.text( .6, .35, 't-test p: ' + '%.3g'%t_test_p,
                       horizontalalignment='left',
                       verticalalignment='bottom',
                       )
            fig_.text( .6, .325, 'mw-test p: ' + '%.3g'%mw_p,
                       horizontalalignment='left',
                       verticalalignment='bottom',
                       )
            fig_.text( .6, .3, 'Cohens d: ' + '%.3g'%cohens_d,
                       horizontalalignment='left',
                       verticalalignment='bottom',
                       )
        

        return s.X.mean(), s.Y.mean(), t_test_p, mw_p, cohens_d

    #-------------------------------
    def page_plot(s):
        f = plt.figure(figsize=(8,10.5),facecolor='white')
        ax1 = f.add_subplot(6,2,1,axisbg='none')
        
    
#------------------------------------------------------------------------
def example():
    """
    
    """
    np.random.seed(7)
    # generate two different normal samples
    A = np.random.normal(loc=0.75, scale=1.5, size=50)
    B = np.random.normal(loc=0.0, scale=1.0, size=50)   
    compare = Compare(A,B, label_='smile', x_label_='TRUTH', y_label_='BLUFF')
    '''
    compare.plot_qq()
    compare.plot_t_likelihood()
    '''
    stats = compare.calc_stats(print_=True)
    f = compare.plot_all()
    #plt.savefig(compare.label + '_plot.png',dpi=70)
    f.savefig(compare.label + '_plot.png',dpi=70)
    plt.show()

#=============================================================================
if __name__ == '__main__':
    print('running main')
    example()