#! python 
"""
Tutorial 1 - Pandas Library
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd   
import plotly.graph_objects as go


def get_data(filename, field_x, field_y, field_z, field_s, field_c):
    """ Import data into an array """
    data = pd.read_csv(filename)
    print(data.head())
    sub_data = data[[field_y, field_x, field_z, field_c]]
    # sub_data.set_index(field_x, inplace=True)
    print(sub_data.head())
    sub_data.plot(linestyle=None, marker='o')
    # plt.show()

    return d_scatter(sub_data, field_x, field_y, field_z, field_s, field_c)


def d_scatter(data, field_x, field_y, field_z, field_s, field_c):
    """ creates a 3d plot using matplotlib """
    x = data[field_x]
    y = data[field_y]
    z = data[field_z]
    c = data[field_c]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=8, color=c,
                                       colorscale='viridis', opacity=0.8) )])
    return fig.show()


if __name__ == "__main__":
    """dimuon dataset: http://opendata.cern.ch/record/303 """
    filepath = "../Datasets/Zee.csv"
    field_z = "py1"
    field_y = "pz1"
    field_x = "pt1"
    field_s = "E1"
    field_c = "M"
    get_data(str(filepath), field_x, field_y, field_z, field_s, field_c)