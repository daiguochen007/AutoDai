# -*- coding: utf-8 -*-
"""
Local IP: 192.168.10.46

"""

import numpy as np
import pandas as pd
 
from flask import Flask
import dash
import dash_core_components as dcc
import dash_html_components as html
 

x = np.linspace(0, 2 * np.pi, 100)
y = 10 * 2 * np.cos(x)


server = Flask(__name__)
app1 = dash.Dash(__name__, server=server, url_base_pathname='/')
app1.layout = html.Div(
    children=[
    html.H1('Dash 测试工具'),
    
    dcc.Graph(
        id='curve',
        figure={
            'data': [{'x': x, 'y': y, 'type': 'Scatter', 'name': 'Testme'},],
            'layout': {'title': 'Curve'}
            })
    ])


@server.route('/page1')
def page_1():
    return "page 1"
 
@server.route('/page2')
def page_2():
    return "page 2"

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=80) #8051
 