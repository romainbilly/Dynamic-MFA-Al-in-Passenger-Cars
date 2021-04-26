# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:54:11 2020

@author: romainb

This file creates a Dash app that display a dynamic Sankey diagram
of Aluminium use in passenger cars

It uses the excel file flows_per_year.xlsx as a data source

It needs to be run from the Anaconda prompt:
$ cd *current_directory*
$ python sankey_app_parameters.py

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
sankey_app_parameters = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

sankey_app_parameters.layout = html.Div(
    [
         dbc.Row(dbc.Col(html.H1("Mass Flows of Aluminium in Passenger cars (Mt/yr)"),
            style={'textAlign': 'center'},
            width={"size": 10, "offset": 1})),
            dbc.Row(
            [
                (dbc.Col(html.Div([
                    html.Div("Population"),
                    dcc.Dropdown(
                        id='population',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 1})),
                (dbc.Col(html.Div([
                    html.Div("Vehicle Ownership"),
                    dcc.Dropdown(
                        id='VpC',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 0.5})),
                (dbc.Col(html.Div([
                    html.Div("Aluminium Content"),
                    dcc.Dropdown(
                        id='al_content',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 0.5})),
                (dbc.Col(html.Div([
                    html.Div("EV Penetration"),
                    dcc.Dropdown(
                        id='powertrain',
                        options=[
                            {'label': 'Constant', 'value': 'Constant'},
                            {'label': 'Stated Policies', 'value': 'SP'},
                            {'label': 'Sustainable Development', 'value': 'SUS'},                          
                            {'label': 'High EV Penetration', 'value': 'High_EV'}
                        ],
                        value='SP'
                    )
                ]), width={"size": 2, "offset": 0.5})),
                (dbc.Col(html.Div([
                    html.Div("Car Segments"),
                    dcc.Dropdown(
                        id='segment',
                        options=[
                            {'label': 'Constant', 'value': 'Constant'},
                            {'label': 'SUVs', 'value': 'SUV'},
                            {'label': 'Small cars', 'value': 'Small_Cars'},
                        ],
                        value='Constant'
                    )
                ]), width={"size": 2, "offset": 0.5})),
            ],
            align="end",
        ),               
         dbc.Row(html.Div(dcc.Graph(id="graph",style={'width': '95vw', 'height': '75vh'}))),
         dbc.Row(
            [
                # (dbc.Col(html.Div([
                #     html.Div("Scenario"),
                #     dcc.Dropdown(
                #         id='scenario',
                #         options=[
                #             {'label': 'Baseline', 'value': 'Baseline'},
                #             {'label': 'High EV penetration', 'value': 'Scenario1'},
                #             {'label': 'ICEV - SUV', 'value': 'Scenario2'},
                #             {'label': 'Autonomous Vehicles', 'value': 'Scenario3'},
                #             {'label': 'Smaller cars', 'value': 'Scenario4'}
                #         ],
                #         value='Baseline'
                #     )
                # ]), width={"size": 3, "offset": 1})),
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
                                  )), width=8)
            ],
            align="end",
        ),
    ]
)


df_data = pd.read_excel('results/flows_plotly_parameters.xlsx')

# max_value is used so that the size of flow is scaled to the biggest one:
# what really matter is the size of the nodes, so it could be improved
max_value = df_data.loc[:, (df_data.columns != 'Time') &\
                           (df_data.columns != 'Population_Scenario') & (df_data.columns !='Vehicle_Ownership_Scenario') &\
                           (df_data.columns != 'Powertrain_Scenario') & (df_data.columns !='Segment_Scenario') &\
                           (df_data.columns != 'Al_Content_Scenario')
                        ].max().max()

@sankey_app_parameters.callback(
    Output("graph", "figure"), 
    [Input("year", "value")],
    [Input("population", "value")],
    [Input("VpC", "value")],
    [Input("al_content", "value")],
    [Input("powertrain", "value")],
    [Input("segment", "value")],
    )

def display_sankey(year, population, VpC, al_content, powertrain, segment):
    df = df_data[(df_data['Time']==year)  & \
                    (df_data['Population_Scenario']==population) & (df_data['Vehicle_Ownership_Scenario']==VpC) & \
                    (df_data['Powertrain_Scenario']==powertrain) & (df_data['Segment_Scenario']==segment) & \
                    (df_data['Al_Content_Scenario']==al_content)
                   ]
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "white", width = 0.5),
          label = ["0. Environment", "1. Raw Material Market", "2. Production", "3. Use", "4. Collection",
                   "5. Dismantling", "6. Shredding of dismantled components", "7. Sorting and Shredding of mixed scrap", "8. Alloy Sorting", "9. Scrap Surplus", ""],
          x = [0.05, 0.10, 0.27, 0.42, 0.53, 0.62, 0.82, 0.72, 0.82, 0.27, 1.1],
          y = [0.3, 0.5, 0.5, 0.5, 0.5, 0.16, 0.16, 0.5, 0.8, 0.7, 1.1],
          color = ["#594F4F", "#594F4F", "#594F4F", "#594F4F", "#594F4F",
                   "#594F4F", "#594F4F", "#594F4F", "#594F4F","#FE4365","white"]
        ),
        link = dict(
          source = [0, 1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 1, 8, 9, 10], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = [1, 2, 3, 4, 0, 5, 7, 6, 7, 0, 1, 0, 1, 8, 1, 9, 8, 9, 10],
          color = ["lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "#FE4365", 
                   "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue",
                   "#FE4365", "#83AF9B", "#FE4365","#83AF9B","lightsteelblue", "lightsteelblue", "#FE4365","white", "white", "white"],
          value = [df['F_0_1'], df['F_1_2'],
                   df['F_2_3'], df['F_3_4'],
                   df['F_4_0'], df['F_4_5'], 
                   df['F_4_7'], df['F_5_6'], 
                   df['F_5_7'], df['F_6_0'], 
                   df['F_6_1'], df['F_7_0'], 
                   df['F_7_1'], df['F_7_8'],
                   df['F_8_1'], df['F_1_9'], 0.001, 0.001, max_value], 
                   ), 
        textfont=dict(color="black", size=15))]
        )
    
    scenario_dict = {
        'Baseline': 'Baseline',
        'Scenario1': 'High EV penetration',
        'Scenario2': 'ICEV - SUV',
        'Scenario3': 'Autonomous Vehicles',
        'Scenario4': 'Smaller cars'
        }
    
    fig.update_layout(
            title_text= "Global flows for " + str(year) + " (Mt/yr)", font=dict(size = 13, color = 'black'),
            paper_bgcolor='white'
            )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig

if __name__ == '__main__':
    sankey_app_parameters.run_server(debug=True)
 