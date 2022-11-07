# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:54:11 2020

@author: romainb

This file creates a Dash app that display a dynamic Sankey diagram
of Aluminium use in passenger cars

It uses the excel file 'results/flows_plotly_parameters.xlsx' as a data source

It needs to be run from the Anaconda prompt:
$ cd *current_directory*
$ python sankey_app_parameters.py

After the app is launched on localhost, it shoud be available on the local server at:
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
import plotly.express as px
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
                            {'label': 'Net Zero', 'value': 'NZE'}
                        ],
                        value='SP'
                    )
                ]), width={"size": 2, "offset": 0.5})),
            ],
            align="center",
        ),
            dbc.Row(
            [
                (dbc.Col(html.Div([
                    html.Div("Car Segments"),
                    dcc.Dropdown(
                        id='segment',
                        options=[
                            {'label': 'Baseline', 'value': 'Baseline'},
                            {'label': 'SUVs', 'value': 'SUV'},
                            {'label': 'Small cars', 'value': 'Small_Cars'},
                        ],
                        value='Baseline'
                    )
                ]), width={"size": 2, "offset": 1})),
                (dbc.Col(html.Div([
                    html.Div("Vehicle Lifetime"),
                    dcc.Dropdown(
                        id='lifetime',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'},
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 0.5})),
                (dbc.Col(html.Div([
                    html.Div("Alloy Sorting"),
                    dcc.Dropdown(
                        id='alloy_sorting',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'},
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 0.5})),  
                (dbc.Col(html.Div([
                    html.Div("Carbon footprint of Al production"),
                    dcc.Dropdown(
                        id='carbon_footprint',
                        options=[
                            {'label': 'BAU Scenario', 'value': 'Constant'},
                            {'label': 'Medium Scenario', 'value': 'Medium'},
                            {'label': 'B2DS Scenario', 'value': 'IAI_B2DS'},
                        ],
                        value='Medium'
                    )
                ]), width={"size": 2, "offset": 0.5})),  
            ],      
            align="center",
        ),

        dbc.Row(
            [
                dbc.Col(html.Div("Year"), width={"size": 1, "offset": 2},
                        style={"height": "5vh", "margin-top":"2%"}                     
                        ),
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
                                          'always_visible': False}
                                  )), width=6,)
            ],
            align="center",
        ),               
        dcc.Tabs(id="tabs_graph", value='tab-about', 
                 style={'width': '100vw'},
                children=[
                dcc.Tab(label='About', value='tab-about'),
                dcc.Tab(label='Sankey diagram', value='tab-sankey'),
                dcc.Tab(label='Aluminium demand', value='tab-al-demand'),  
                dcc.Tab(label='Scrap surplus', value='tab-scrap'), 
                dcc.Tab(label='Carbon footprint', value='tab-cf'),
        ]),
        
        html.Div(id='graph'),
    ]
)


df_data = pd.read_csv(
    'results/flows_plotly_parameters.csv',
     index_col=[1, 2, 3, 4, 5, 6, 7],
     header=[0]
     )

df_lines = df_data.copy()
df_lines["Hash"] = df_lines.index.map(hash)
df_lines.set_index(["Hash","Time"], append=True, inplace=True)
df_lines = df_lines[['F_1_2']]
df_lines = df_lines[df_lines.index.get_level_values('Time').isin([2017,2020,2030,2040,2050])].sort_index()


# max_value is used so that the size of flow is scaled to the biggest one:
# what really matter is the size of the nodes, so it could be improved
max_value = df_data.loc[:, (df_data.columns != 'Time') &\
                           (df_data.columns != 'Population_Scenario') & (df_data.columns !='Vehicle_Ownership_Scenario') &\
                           (df_data.columns != 'Powertrain_Scenario') & (df_data.columns !='Segment_Scenario') &\
                           (df_data.columns != 'Al_Content_Scenario') & (df_data.columns !='Lifetime_Scenario') & \
                           (df_data.columns != 'Alloy_Sorting_Scenario')
                        ].max().max()

cf_data = pd.read_csv('results/carbon_footprint_scenarios_parameters.csv')
cf_data.sort_values(by=['Alloy_Sorting_Scenario','Al_Content_Scenario',
                        'Lifetime_Scenario','Segment_Scenario','Powertrain_Scenario',
                        'Vehicle_Ownership_Scenario','Population_Scenario',
                        'Alloy_Sorting_Scenario','Carbon_Footprint_Scenario','Time'],
                        ascending=True, inplace=True) 
    
@sankey_app_parameters.callback(
    Output('graph', 'children'), 
    [Input("year", "value")],
    [Input("population", "value")],
    [Input("VpC", "value")],
    [Input("al_content", "value")],
    [Input("powertrain", "value")],
    [Input("segment", "value")],
    [Input("lifetime", "value")],
    [Input("alloy_sorting", "value")],
    [Input("carbon_footprint", "value")],
    [Input('tabs_graph', 'value')]
    )

def display_fig(year, population, VpC, al_content, powertrain, segment, 
                lifetime, alloy_sorting, carbon_footprint, tabs_graph):
    print(tabs_graph)
    if tabs_graph == 'tab-about':
        return html.Div([
            html.H3('Interactive visualization for future aluminium flows in passenger cars and associated emissions',
                    style={"height": "2vh", "margin-top":"1%", "margin-left":"1%"}),
            html.Div([
                html.P('Based on the article "Aluminium use in passenger cars pose challenges for recycling and GHG emissions" (under review).'),
                html.P('Please use the tabs to navigate between the different graphs, and the filters to select scenarios parameters'),
                html.P("Complete code and data are available on GitHub:"),
                html.A('https://github.com/romainbilly/Dynamic-MFA-Al-in-Passenger-Cars', 
                       href='https://github.com/romainbilly/Dynamic-MFA-Al-in-Passenger-Cars/tree/dashboard')
            ],
                style={"margin-top":"3%", "margin-left":"1%"})
        ])
    
    
    elif tabs_graph == 'tab-sankey':
        df = df_data.query((
            "Population_Scenario==@population & Vehicle_Ownership_Scenario==@VpC &"
            "Powertrain_Scenario==@powertrain & Segment_Scenario==@segment &"
            "Al_Content_Scenario==@al_content & Lifetime_Scenario==@lifetime &"
            "Alloy_Sorting_Scenario==@alloy_sorting& Time==@year"))
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "white", width = 0.5),
              label = ["0. Environment", "1. Raw Material Market", "2. Production", "3. Use", "4. Collection",
                       "5. Dismantling", "6. Shredding of dismantled components", "7. Sorting and Shredding of mixed scrap", "8. Alloy Sorting", "9. Scrap Surplus", ""],
              x = [0.07, 0.13, 0.27, 0.42, 0.53, 0.62, 0.82, 0.72, 0.82, 0.27, 1.1],
              y = [0.3, 0.5, 0.5, 0.5, 0.5, 0.16, 0.16, 0.5, 0.8, 0.7, 1.1],
              color = ["#594F4F", "#594F4F", "#594F4F", "#594F4F", "#594F4F",
                       "#594F4F", "#594F4F", "#594F4F", "#594F4F","#FE4365","white"]
            ),
            link = dict(
              source = [0, 1, 2, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 1, 8, 9, 10], # indices correspond to labels, eg A1, A2, A1, B1, ...
              target = [1, 2, 3, 4, 0, 5, 7, 6, 7, 0, 1, 0, 1, 8, 1, 9, 8, 9, 10],
              color = ["lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "#FE4365", 
                       "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue",
                       "#FE4365", "#83AF9B", "#FE4365","#83AF9B","lightsteelblue", "#83AF9B", "#FE4365","white", "white", "white"],
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

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        return html.Div([
            html.H4("Global flows for " + str(year) + " (Mt/yr)",
                    style={"height": "2vh", "margin-top":"1%", "margin-left":"1%"}),
            dcc.Graph(id="tab-sankey",
                      figure=fig,
                      style={'width': '95vw', 'height': '55vh'}
            )
        ])

    elif tabs_graph=='tab-al-demand':
        df = df_data.query((
            "Population_Scenario==@population & Vehicle_Ownership_Scenario==@VpC &"
            "Powertrain_Scenario==@powertrain & Segment_Scenario==@segment &"
            "Al_Content_Scenario==@al_content & Lifetime_Scenario==@lifetime &"
            "Alloy_Sorting_Scenario==@alloy_sorting"))
        
        # Figure 1: Line graph for Al demand for all scenarios + chosen one
        df_lines2 = df_lines.query('Alloy_Sorting_Scenario==@alloy_sorting')
        fig = px.line(df_lines2, 
                      x=df_lines2.index.get_level_values(8), #index number for time
                      y='F_1_2', 
                      line_group=df_lines2.index.get_level_values(7), #index number for hash
                      # color=df_lines2.index.get_level_values(3),
                      # hover_data=['F_1_2'], 
                      hover_data=None,
                      render_mode='webgl',
                      labels={'x':'Year',
                              'F_1_2':'Al demand (Mt/yr)',
                              'color':'EV penetration scenario'})
        fig.update_traces(hoverinfo='skip')
        fig.update_traces(hovertemplate=None)
        
        fig.add_trace(
            # use Scatterg1 to force this line on top of the previous plot
            go.Scattergl(x=df['Time'], y=df['F_1_2'],
                        mode='lines',
                        name='Chosen scenario',
                        line=dict(color="Black", width=5), opacity=1)
            )
        # fig.update_traces(hovertemplate='Year: %{x} <br>Aluminium demand: %{y} Mt/yr')
        fig.update_layout(title="Aluminium demand",
                          yaxis_range=[0,230])
        
        # Figure 2: Area graph for primary and secondary Al demand
        layout = go.Layout(
            title="Primary and Secondary Aluminium demand",
            xaxis=dict(
                title="Year"
            ),
            yaxis=dict(
                title="Al demand (Mt/yr)"
            ) ) 
        fig2 = go.Figure(layout=layout)
        fig2.add_trace(go.Scatter(x=df['Time'], y=df['F_7_1'] + df['F_8_1'] -  df['F_1_9'],
                                 name='Secondary Al',mode='lines', stackgroup='one'))
            
        fig2.add_trace(go.Scatter(x=df['Time'], y=df['F_0_1'], mode='lines',
                                 name='Primary Al', stackgroup='one'))
        fig2.update_layout(hovermode="x unified",
                          yaxis_range=[0,230])
       
        return html.Div([
            html.H4('Scenario for aluminium demand in passenger cars (Mt/yr)',
                    style={"height": "2vh", "margin-top":"1%", "margin-left":"1%"}),
            dbc.Row([
                dcc.Graph(
                    id='tab-series',
                    figure=fig,
                    style={'width': '48vw', 'height': '55vh'}),
                dcc.Graph(
                    figure=fig2,
                    style={'width': '48vw', 'height': '55vh'})
            ])
        ])
    
    elif tabs_graph=='tab-scrap':
        df = df_data.query((
            "Population_Scenario==@population & Vehicle_Ownership_Scenario==@VpC &"
            "Powertrain_Scenario==@powertrain & Segment_Scenario==@segment &"
            "Al_Content_Scenario==@al_content & Lifetime_Scenario==@lifetime &"
            "Alloy_Sorting_Scenario==@alloy_sorting"))
        fig = px.line(df, x="Time", y="F_1_9", labels={'Time':'Year','F_1_9':'Scrap surplus (Mt/yr)'})
        
        return html.Div([
           html.H4('Scenario for mixed scrap surplus from passenger cars (Mt/yr)',
                    style={"height": "3vh", "margin-top":"1%", "margin-left":"1%"}),
           dbc.Row([
               dcc.Graph(
                   id='tab-surplus',
                   figure=fig,
                   style={'width': '50vw', 'height': '55vh'})
            ])
        ])
      
    
    elif tabs_graph=='tab-cf':
        cf = cf_data[(cf_data['Population_Scenario']==population) & (cf_data['Vehicle_Ownership_Scenario']==VpC) & \
                      (cf_data['Powertrain_Scenario']==powertrain) & (cf_data['Segment_Scenario']==segment) & \
                      (cf_data['Al_Content_Scenario']==al_content) & (cf_data['Lifetime_Scenario']==lifetime) & \
                      (cf_data['Alloy_Sorting_Scenario']==alloy_sorting) & \
                      (cf_data['Carbon_Footprint_Scenario']==carbon_footprint)
                    ]
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cf['Time'], y=cf['Carbon_footprint_secondary'], mode='lines', 
                                  name='Carbon footprint from Secondary Al', stackgroup='one'))
            
        fig.add_trace(go.Scatter(x=cf['Time'], y=cf['Carbon_footprint_primary'], mode='lines',
                                  name='Carbon footprint from Primary Al', stackgroup='one'))
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Al demand (Mt/yr)")

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=cf['Time'], y=cf['Carbon_footprint_secondary'].cumsum(), mode='lines', 
                                  name='Cumulative carbon footprint from Secondary Al', stackgroup='one'))
            
        fig_cum.add_trace(go.Scatter(x=cf.tail(31)['Time'], y=cf.tail(31)['Carbon_footprint_primary'].cumsum(), mode='lines',
                                  name='Cumulative Carbon footprint from Primary Al', stackgroup='one'))
        fig_cum.update_xaxes(title_text="Year")
        fig_cum.update_yaxes(title_text="Al demand (Mt/yr)")
        fig_cum.update_xaxes(range=[2020,2050])
        
        return html.Div([
            html.H4('Scenario for the carbon footprint of Aluminium used in passenger cars (Mt Co2e/yr)',
                    style={"height": "2vh", "margin-top":"1%", "margin-left":"1%"}),
            dbc.Row([
                dcc.Graph(
                    id='tab-cf',
                    figure=fig,
                    style={'width': '48vw', 'height': '55vh'}),
                dcc.Graph(
                    figure=fig_cum,
                    style={'width': '48vw', 'height': '55vh'})
            ])
        ])

if __name__ == '__main__':
    # for running the app on localhost (on your computer) uncomment the next line:
    #sankey_app_parameters.run_server(debug=True)
    # for running the app on the NTNU Openstack server uncomment the next line:
    sankey_app_parameters.run_server(host="0.0.0.0", port="8050", debug=False)
 
