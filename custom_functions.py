# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:04:02 2020

@author: romainb
"""
import pandas as pd
from varname import nameof
import numpy as np
import matplotlib.pyplot as plt
import pylab
plt.ioff()



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
    

y_dict = {
        'name': 'Stock change',
        'aspect': 'Region',
        'unit': 'cars'
        }
x_dict = {
        'name': 'Time',
        'aspect': 't'
        }
    
def plot_result_time(array, y_dict, IndexTable, t_min, t_max, width=35, height=25, show='no', stack='no'):
    
    # Car Stock per region
    fig, ax = plt.subplots()
    plt.figure(figsize=(width, height))
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
    fig.savefig('results/plots/' + y_dict['name'] +' by ' + y_dict['aspect'], dpi = 400)    
    if show == 'yes':
        print('pouet')
        plt.show()
    plt.close(fig)
