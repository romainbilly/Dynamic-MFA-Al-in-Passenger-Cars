# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:54:11 2020

@author: romainb

This file creates a Dash app that display a dynamic Sankey diagram
of Aluminium use in passenger cars

It uses the excel file flows_per_year.xlsx as a data source

It needs to be run from the Anaconda prompt:
$ cd *current_directory*
$ python sankey_app.py

After the app is launched, it shoud be available on the local server at:
    http://127.0.0.1:8050/

dependencies:
    dash 
    
    
"""
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# import Bootstrap css layout for the Dash app
sankey_app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

sankey_app.layout = html.Div(
    [
         dbc.Row(dbc.Col(html.H1("Mass Flows of Aluminium in Passenger cars (Mt/yr)"),
            style={'textAlign': 'center'},
            width={"size": 10, "offset": 1})),
                 
         dbc.Row(html.Div(dcc.Graph(id="graph",style={'width': '95vw', 'height': '75vh'}))),
         dbc.Row(
            [
                (dbc.Col(html.Div([
                    html.Div("Scenario"),
                    dcc.Dropdown(
                        id='scenario',
                        options=[
                            {'label': 'Baseline', 'value': 'Baseline'},
                            {'label': 'Historic', 'value': 'Historic'},
                            {'label': 'Electric Europe', 'value': 'Electric Europe'}
                        ],
                        value='Baseline'
                    )
                ]), width={"size": 3, "offset": 1})),
                dbc.Col(html.Div("Year"), width={"size": 0.5, "offset": 1}),
                dbc.Col(html.Div(
                        dcc.Slider(id='year', min=2000, max=2050,
                                  value=2020, step=1,
                                  marks={2000: '2000',
                                  2010: '2010',
                                  2020: '2020',
                                  2030: '2030',
                                  2040: '2040',
                                  2050: '2050'},
                                  tooltip={
                                          'always_visible': True}
                                  )), width=5)
            ],
            align="end",
        ),
    ]
)


df = pd.read_excel('results/flows_per_year.xlsx')

# max_value is used so that the size of flow is scaled to the biggest one:
# what really matter is the size of the nodes, so it could be improved
max_value = df.loc[:, df.columns != 'Time'].max().max()

@sankey_app.callback(
    Output("graph", "figure"), 
    [Input("year", "value")],
    [Input("scenario", "value")])

def display_sankey(year, scenario):
    year = year - 1900

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "white", width = 0.5),
          label = ["0. Environment"," 1. Production", "2. Use", "3. Collection",
                   "4. Dismantling", "5. Shredding", "6. Mixed Shredding", "7. Scrap Surplus", ""],
          x = [0.05, 0.2, 0.3, 0.4, 0.5, 0.8, 0.6, 0.7, 1.1],
          y = [0.18, 0.4, 0.4, 0.4, 0.16, 0.16, 0.7, 0.5, 1.1],
          color = ["#594F4F", "#594F4F", "#594F4F", "#594F4F", "#594F4F",
                   "#594F4F", "#594F4F", "#FE4365","white"]
        ),
        link = dict(
          source = [0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = [1, 2, 3, 0, 4, 6, 5, 6, 0, 1, 0, 1, 7, 7, 8],
          color = ["lightsteelblue", "lightsteelblue", "lightsteelblue", "#FE4365", 
                   "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue",
                   "#FE4365", "#83AF9B", "#FE4365","#83AF9B","#FE4365", "white", "white"],
          value = [df['F_0_1_t'][year], df['F_1_2_t'][year], df['F_2_3_t'][year], df['F_3_0_t'][year],
                   df['F_3_4_t'][year], df['F_3_6_t'][year], df['F_4_5_t'][year], df['F_4_6_t'][year],
                   df['F_5_0_t'][year], df['F_5_1_t'][year], df['F_6_0_t'][year], df['F_6_1_t'][year],
                   df['scrap_surplus_t'][year], 0.001, max_value/2], 
                   ), 
        textfont=dict(color="black", size=15))]
        )
    
    fig.update_layout(
            title_text= "Global flows for " + str(year + 1900) + " according to the " + scenario + " scenario (Mt/yr)", font=dict(size = 15, color = 'black'),
            paper_bgcolor='white'
            )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig



if __name__ == '__main__':
    sankey_app.run_server(debug=True)
 