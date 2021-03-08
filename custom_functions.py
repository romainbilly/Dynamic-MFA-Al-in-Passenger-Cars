# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:04:02 2020

@author: romainb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
from matplotlib.figure import Figure






def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def export_to_csv(array: np.array, array_name: str, IndexTable):
    """
    Function used to export a multidimensional array to a flat csv file

    :param array: The array that will be exported
    :param array_name: The name of the array that will be exported
    this will be the name given to the csv file
    IMPORTANT: the name needs to end with "_dims", 
    where dims is a string composed of its dimensions indexes
    according to the IndexTable
    :paran IndexTable: The IndexTable used in the ODYM model,
    used to give the correct index to the exported file
    
    Exporting arrays containing many dimensions will result in slow execution 
    and very large csv files, use with caution!
    """

    array_dims = array_name.split('_')[-1]
    iterables = []
    names = []
    for dim in array_dims:
        iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
        names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  
    
    index = pd.MultiIndex.from_product(iterables, names=names)
    df = pd.DataFrame(array.flatten(),index=index, columns = ['Values'])
    file_name = 'results/' + array_name + '.csv'
    df.to_csv(file_name)
    print(array_name + ' of dimensions: ' + array_dims + ' has been saved as: ' + file_name)
    

def export_to_csv_scenario(array: np.array, array_name: str, IndexTable):
    """
    Function used to export a multidimensional array to a flat csv file
    results are stored depending on the scenario which is the last index of the array

    :param array: The array that will be exported
    :param array_name: The name of the array that will be exported
    this will be the name given to the csv file
    IMPORTANT: the name needs to end with "_dims", 
    where dims is a string composed of its dimensions indexes
    according to the IndexTable
    :paran IndexTable: The IndexTable used in the ODYM model,
    used to give the correct index to the exported file
    
    Exporting arrays containing many dimensions will result in slow execution 
    and very large csv files, use with caution!
    """

    array_dims = array_name.split('_')[-1]
    iterables = []
    names = []
    for dim in array_dims:
        iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
        names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  
    
    index = pd.MultiIndex.from_product(iterables, names=names)
    df = pd.DataFrame(array.flatten(),index=index, columns = ['Values'])
    file_name = 'results/' + array_name + '.csv'
    df.to_csv(file_name)
    print(array_name + ' of dimensions: ' + array_dims + ' has been saved as: ' + file_name)




def plot_result_time(array, y_dict, IndexTable, t_min, t_max, plot_dir, 
                     width=35, height=25, show='no', stack='no'):
    """
    Function used to draw and save standard plots from the model results
    x-axis is always time in years
    
    :param array:  2D numpy array that will be plotted
    :param y_dict: dict, defines the properties of the y axis,
    with the following template:
        y_dict = {
            'name': 'name', #name of graph
            'aspect': aspect', #aspect used for splitting the data in categories, 2nd dim of the array
            'unit': 'unit'  #unit of the data (will show on the y axis)
        }
        the plot will be saved at results/plots/'name' by 'aspect'.png
    :param t_min and t_max: define the years of x-axis 
    :param width and height: define the size of the plot
    :param show: if 'yes', the graph is shown on the console, 
                otherwise it is just saved under results/plot
    :param stack: if 'yes', uses a stackplot
    """    
    fig, ax = plt.subplots()
    m = 0
    category = IndexTable.Classification[y_dict['aspect']].Items
    N_cat = len(category)
    MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/N_cat)) # select 10 colors from the 'Paired' color map.
    if stack == 'yes':
        ax.stackplot(np.array(IndexTable['Classification']['Time'].Items[t_min:t_max]),
                np.transpose(array[t_min:t_max,:]),
                colors = MyColorCycle[:,:])
    else:
        for m in range(N_cat):
            ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
                    array[t_min:t_max,m],
                    color = MyColorCycle[m,:], linewidth = 2)
            m += 1
    ax.set_ylabel(y_dict['name'] +', ' + y_dict['unit'],fontsize =16)
    fig.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
    ax.legend(category, loc='upper left',prop={'size':8})
    mkdir_p(plot_dir)
    plot_path = plot_dir + '/' + y_dict['name'] +' by ' + y_dict['aspect']
    fig.savefig(plot_path, dpi = 400)    
    if show == 'yes':
        plt.show()
    plt.cla()
    plt.clf()
    plt.close(fig)
    print("Saved to: " + plot_path)


def plot_result_time_scenario(array, y_dict, IndexTable, t_min, t_max, scenario, 
                              plot_dir, width=35, height=25, show='no', stack='no'):
    """
    Function used to draw and save standard plots from the model results
    x-axis is always time in years
    
    :param array:  2D numpy array that will be plotted
    :param y_dict: dict, defines the properties of the y axis,
    with the following template:
        y_dict = {
            'name': 'name', #name of graph
            'aspect': aspect', #aspect used for splitting the data in categories, 2nd dim of the array
            'unit': 'unit'  #unit of the data (will show on the y axis)
        }
        the plot will be saved at results/plots/'name' by 'aspect'.png
    :param t_min and t_max: define the years of x-axis 
    :param width and height: define the size of the plot
    :param show: if 'yes', the graph is shown on the console, 
                otherwise it is just saved under results/plot
    :param stack: if 'yes', uses a stackplot
    """    
    fig, ax = plt.subplots()
    # plt.figure(figsize=(width, height))
    m = 0
    scenario_name = IndexTable.Classification[IndexTable.set_index('IndexLetter').\
                                              index.get_loc('S')].Items[scenario]
    category = IndexTable.Classification[y_dict['aspect']].Items
    N_cat = len(category)
    MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/N_cat)) # select 10 colors from the 'Paired' color map.
    if stack == 'yes':
        ax.stackplot(np.array(IndexTable['Classification']['Time'].Items[t_min:t_max]),
                np.transpose(array[t_min:t_max,:,scenario]),
                colors = MyColorCycle[:,:])
    else:
        for m in range(N_cat):
            ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
                    array[t_min:t_max,m,scenario],
                    color = MyColorCycle[m,:], linewidth = 2)
            m += 1
    ax.set_ylabel(y_dict['name'] +', ' + y_dict['unit'],fontsize =16)
    fig.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
    ax.legend(category, loc='upper left',prop={'size':8})
    plot_dir = os.path.join(plot_dir, scenario_name)
    mkdir_p(plot_dir)
    plot_path = (plot_dir + '/' + y_dict['name'] +' by ' + y_dict['aspect'])
    fig.savefig(plot_path, dpi = 400)    
    if show == 'yes':
        plt.show()
    plt.cla()
    plt.clf()
    plt.close(fig)
    print("Saved to: " + plot_path)


    
    
class ExportFigure(Figure):
    """Figure class used to manage the export of plots from the model"""

    def __init__(self, IndexTable, width=35, height=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.IndexTable = IndexTable
        self.ax = self.add_subplot(1,1,1)
     
    
    def plot_result_time(self, array, y_dict, t_min, t_max, show='no', stack='no'):
        """
        Function used to draw and save standard plots from the model results
        x-axis is always time in years
        
        :param array:  2D numpy array that will be plotted
        :param y_dict: dict, defines the properties of the y axis,
        with the following template:
            y_dict = {
                'name': 'name', #name of graph
                'aspect': aspect', #aspect used for splitting the data in categories, 2nd dim of the array
                'unit': 'unit'  #unit of the data (will show on the y axis)
            }
            the plot will be saved at results/plots/'name' by 'aspect'.png
        :param t_min and t_max: define the years of x-axis 
        :param width and height: define the size of the plot
        :param show: if 'yes', the graph is shown on the console, 
                    otherwise it is just saved under results/plot
        :param stack: if 'yes', uses a stackplot
        """    
        
        m = 0
        category = self.IndexTable.Classification[y_dict['aspect']].Items
        N_cat = len(category)
        MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/N_cat)) # select 10 colors from the 'Paired' color map.
        if stack == 'yes':
            self.ax.stackplot(np.array(self.IndexTable['Classification']['Time'].Items[t_min:t_max]),
                    np.transpose(array[t_min:t_max,:]),
                    colors = MyColorCycle[:,:])
        else:
            for m in range(N_cat):
                self.ax.plot(self.IndexTable['Classification']['Time'].Items[t_min:t_max],
                        array[t_min:t_max,m],
                        color = MyColorCycle[m,:], linewidth = 2)
                m += 1
        self.ax.set_ylabel(y_dict['name'] +', ' + y_dict['unit'],fontsize =16)
        self.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
        self.ax.legend(category, loc='upper left',prop={'size':8})
        self.savefig('results/plots/' + y_dict['name'] +' by ' + y_dict['aspect'], dpi = 400)    
        if show == 'yes':
            plt.show()
        plt.cla()
        plt.clf()

    
    
    def plot_result_time_scenario(self, array, y_dict, t_min, t_max, scenario, 
                                  width=35, height=25, show='no', stack='no'):
        """
        Function used to draw and save standard plots from the model results
        x-axis is always time in years
        
        :param array:  2D numpy array that will be plotted
        :param y_dict: dict, defines the properties of the y axis,
        with the following template:
            y_dict = {
                'name': 'name', #name of graph
                'aspect': aspect', #aspect used for splitting the data in categories, 2nd dim of the array
                'unit': 'unit'  #unit of the data (will show on the y axis)
            }
            the plot will be saved at results/plots/'name' by 'aspect'.png
        :param t_min and t_max: define the years of x-axis 
        :param width and height: define the size of the plot
        :param show: if 'yes', the graph is shown on the console, 
                    otherwise it is just saved under results/plot
        :param stack: if 'yes', uses a stackplot
        """    

        m = 0
        scenario_name = self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').\
                                                  index.get_loc('S')].Items[scenario]
        category = self.IndexTable.Classification[y_dict['aspect']].Items
        N_cat = len(category)
        MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/N_cat)) # select 10 colors from the 'Paired' color map.
        if stack == 'yes':
            self.ax.stackplot(np.array(self.IndexTable['Classification']['Time'].Items[t_min:t_max]),
                    np.transpose(array[t_min:t_max,:,scenario]),
                    colors = MyColorCycle[:,:])
        else:
            for m in range(N_cat):
                self.ax.plot(self.IndexTable['Classification']['Time'].Items[t_min:t_max],
                        array[t_min:t_max,m,scenario],
                        color = MyColorCycle[m,:], linewidth = 2)
                m += 1
        self.ax.set_ylabel(y_dict['name'] +', ' + y_dict['unit'],fontsize =16)
        self.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
        self.ax.legend(category, loc='upper left',prop={'size':8})
        self.savefig('results/plots/' + scenario_name + '/' + \
                    y_dict['name'] +' by ' + y_dict['aspect'], dpi = 400)    
        if show == 'yes':
            plt.show()
        # plt.cla()
  
    
  
    
