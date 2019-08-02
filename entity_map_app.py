import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import datetime
import ast 
import json


# --- INITIALIZE VARIABLES --- # 

# to be used as as options for multi-select 
ENTITY_TYPES = ['resource', 'request', 'blockage', 'duck']
#!! color selection to stand out
COLOR_MAP = {'blockage':'#969190'
            , 'duck':'#f7e61b'
            , 'resource':'#1b36f7'
            , 'request':'#f71b43'}
KEEP_COLS = ['id', 'created_at', 'updated_at', 'event_type', 'device_type']


# use your own mapbox token saved in form {'token':'YOUR_TOKEN'}
# https://docs.mapbox.com/help/how-mapbox-works/access-tokens/
with open('./credentials.json', 'r') as f:
    credentials = json.load(f)
MAPBOX_TOKEN = credentials['token']
px.set_mapbox_access_token(MAPBOX_TOKEN)

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# --- LOAD DATA --- # 

#!! replace with actual data load
isabella = pd.read_csv('isabela_duck_deployment.csv'
                      , converters={'payload': ast.literal_eval})

payload = json_normalize(isabella['payload'])
payload[['lat', 'lon']] = payload['civilian.info.location'].str.split(pat=',', expand=True)
payload['lat'] = payload['lat'].astype(float)
payload['lon'] = payload['lon'].astype(float)

#!! replace with real classes 
payload['entity_type'] = np.random.choice(ENTITY_TYPES, len(payload))

data = pd.concat([isabella[KEEP_COLS], payload], axis=1)

#!! add data quality checks


# --- APP LAYOUT --- # 

app = dash.Dash(__name__
                , external_stylesheets=EXTERNAL_STYLESHEETS)

app.layout = html.Div([

     # Title
    html.H1("OWL Incident Command Dashboard"),

    # Filter 
        html.Div([
            # multi-select 
            dcc.Dropdown(
                id='filter-entity-types',
                options=[{'label': i, 'value':i} for i in ENTITY_TYPES],
                value=ENTITY_TYPES,
                multi=True
            ),
            # storage
            html.Div(id='filter-df' 
                    , style={'display':'none'})
        ]),

    # Map
        html.Div([
            dcc.Graph(id='map'
                    , style={'width':'75%', 'display':'inline-block'})
        ])

])




@app.callback(
    Output('map', 'figure'),
    [Input('filter-entity-types', "value")]
)
def make_map(entity_types):
    df = data[data['entity_type'].isin(entity_types)]
    fig = px.scatter_mapbox(df
                            , lat='lat'
                            , lon='lon'
                            , color='entity_type'
                            , color_discrete_map=COLOR_MAP
                            , text='entity_type'
                            , zoom=15)
    fig.update_mapboxes({'style':'satellite'
                     , 'center':{'lat': data['lat'].mean(), 'lon':data['lon'].mean()}
                    })
    return fig


# -- RUN APP --- # 

if __name__ == '__main__':
    app.run_server(debug=True)