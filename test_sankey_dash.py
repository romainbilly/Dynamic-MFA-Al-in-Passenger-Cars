# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:54:11 2020

@author: romainb
"""
import plotly.offline
import plotly.graph_objects as go



for t in range(100,151):
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["0. Environment"," 1. Prod", "2. Use", "3. Collection", "4. Dismantling", "5. Shredding", "6. Mixed Scrap Shredding"],
          x = [0.05, 0.2, 0.3, 0.4, 0.5, 0.7, 0.7],
          y = [0.18, 0.4, 0.4, 0.4, 0.18, 0.18, 0.65],
          color = "#594F4F"
        ),
        link = dict(
          source = [0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = [1, 2, 3, 0, 4, 6, 5, 6, 0, 1, 0, 1],
          color = ["lightsteelblue", "lightsteelblue", "lightsteelblue", "#FE4365", "lightsteelblue",
                   "lightsteelblue", "lightsteelblue", "lightsteelblue", "#FE4365", "#83AF9B", "#FE4365","#83AF9B"],
          value = [F_0_1_t[t], F_1_2_t[t], F_2_3_t[t], F_3_0_t[t], F_3_4_t[t], F_3_6_t[t], F_4_5_t[t],
                   F_4_6_t[t], F_5_0_t[t], F_5_1_t[t], F_6_0_t[t], F_6_1_t[t]], 
      ), textfont=dict(color="black", size=15))])
    
    fig.update_layout(
            title_text="Aluminium in Passenger cars in " + str(t + 1900) + ", Flows in Mt/yr", font=dict(size = 20, color = 'black'),
            paper_bgcolor='white')
    
    #plotly.offline.plot(fig, validate=False)
    fig.write_image("results/plots/Sankey" + str(t + 1900) + ".png", width=1500, height=800, scale = 2.0)
plotly.offline.plot(fig, validate=False)