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

from generate_random_data import Generate_Random_Data as grd 

# --- INITIALIZE VARIABLES --- # 

# to be used as as options for multi-select 
ENTITY_TYPES = ['resource', 'civilian', 'blockage', 'duck']
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
# px.set_mapbox_access_token(MAPBOX_TOKEN)

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# -- FUNCTIONS -- # 

def load_ducks(num_ducks=10):
    """Query ducks table"""

    #rand Generate ducks 
    device_types = np.random.choice(['papa-duck', 'mama-duck', 'duck'], size=num_ducks)
    duck_id_candidates = ['5QOZlOxJECBv', 'cVDXZ9fljbIh', 'Qnhq1G5Rdwuv', 'T1PaUmJjbFcO',
       '0IUUAhmA99lq', '2lmrjdoJ237Z', 'hAkKlMLz8qgM', 'eeusfZ8AE8mv',
       'xioGCMhFLPC8', 'iJWI1F0QggHr', 'RV5CEfaDzuTD', 'KxqoWN44irrF',
       'mtldKD0Eslrz', '0P9pqRkY8yXv', 'NLZFD3yj4Oug', 'Hej9OwFoEdnR',
       '1gIOGluTtwiW', 'VEuwAoRLhL25', 'r8Zm272RIu5s', 'Szo4vS0LIFrO',
       'BZlCvDsdq6WR']
    duck_ids = np.random.choice(duck_id_candidates, size=num_ducks)
    ducks = pd.DataFrame({'device_type': device_types, 'device_id':duck_ids})
    #rand
    pr_coor = (18.238334,-66.530801)
    pr_radius = 25000
    ducks['location'] = grd.random_coor(num=num_ducks, radius=pr_radius, center=pr_coor)
    ducks['location'] = ducks['location'].map(lambda x: str(x[0]) + ',' + str(x[1]))

    # Unpack latitude and longitude -- VALIDATE ALWAYS EXIST
    ducks[['lat', 'lon']] = ducks['location'].str.split(pat=',', expand=True)
    ducks['lat'] = ducks['lat'].astype(float)
    ducks['lon'] = ducks['lon'].astype(float)

    return ducks.drop(columns='location')

def load_resources():
    """Query resources table"""
    pass 

def load_civilians():
    """Query requests table"""
        
    #!! replace with actual data load
    civilians = pd.read_csv('isabela_duck_deployment.csv'
                        , converters={'payload': ast.literal_eval})
    return civilians
# unnest json column

def extract_civilian_payload(civilians):
    """Payload is a nested json field 
    civilians: df containing civilian requests data 
    -- Remove #rand code blocks once we have real data -- 
    """

    # Normalize payload part of schema
    payload = json_normalize(civilians['payload'])
    
    
    #rand 
    data_points = len(payload)

    #rand
    pr_coor = (18.238334,-66.530801)
    pr_radius = 25000
    payload['civilian.info.location'] = grd.random_coor(num=data_points, radius=pr_radius, center=pr_coor)
    payload['civilian.info.location'] = payload['civilian.info.location'].map(lambda x: str(x[0]) + ',' + str(x[1]))

    # Unpack latitude and longitude -- VALIDATE ALWAYS EXIST
    payload[['lat', 'lon']] = payload['civilian.info.location'].str.split(pat=',', expand=True)
    payload['lat'] = payload['lat'].astype(float)
    payload['lon'] = payload['lon'].astype(float)

    #rand # Name data
    payload['civilian.info.name'] = grd.random_names(num=data_points)

    #rand # Phone data
    payload['civilian.info.phone'] = grd.random_digits(num=data_points)

    #rand # Occupants data
    payload['num_people'] = grd.random_ints(num=data_points)
    payload['num_pets'] = grd.random_ints(num=data_points)

    #rand # Dangers, first aid, water, food, vacant data
    for col in ['medical', 'fire_explosion', 'flood', 'violence', 'landslide', 'building_collapse',
                'trapped', 'death','first_aid', 'shelter', 'food', 'financial_aid', 'water', 
                'evacuation', 'clothing', 'hygiene','electrical', 'road_blocked', 'fallen_trees',
                'chemical','smoke_fire','animals','swiftwater', 'explosives','immobile', 
                'public_name', 'change_status']:
        payload[col] = grd.binary(num=data_points)

    REDUNDANT_COLS = ['civilian.info.location', 'civilian.needs.first-aid', 'civilian.needs.food',
       'civilian.needs.water', 'civilian.status.danger',
       'civilian.status.vacant']


    return payload.drop(columns=REDUNDANT_COLS)

# --- LOAD DATA --- # 
# load each source of data into its own df

civilians = load_civilians()
payload = extract_civilian_payload(civilians)

civilian_data = pd.concat([civilians[KEEP_COLS], payload], axis=1)


ducks = load_ducks()
print(ducks['lat'].mean())
print(ducks['lon'].mean())

#!! Add other entities

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
    fig = go.Figure()

    # Add trace if requested by entity_types multi-select

    if 'duck' in entity_types:
        fig.add_trace(go.Scattermapbox(
                  lat=ducks['lat']
                , lon =ducks['lon']
                , mode='markers'
                , name='Active Ducks'
                , hovertext=ducks['device_id']
                , marker={'symbol':'dog-park'} # can we color this?
        ))
    #!! How should we display the request information? Table? 
    if 'civilian' in entity_types:
        fig.add_trace(go.Scattermapbox(
                  lat=civilian_data['lat']
                , lon=civilian_data['lon']
                , mode='markers'
                , name='Civilian Requests'
                , hovertext=civilian_data['civilian.info.name']
                , marker={'size':8} 
        ))

    fig.update_layout(
        title='Entity Map',
        autosize=True,
        hovermode='closest',
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=MAPBOX_TOKEN,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=ducks['lat'].mean(),
                lon=ducks['lon'].mean()
            ),
            pitch=0,
            zoom=8,
            style='satellite'
    ))
    # fig.update_mapboxes({'style':'satellite'
    #                  , 'center':{'lat': ducks['lat'].mean(), 'lon':ducks['lon'].mean()}
    #                 })
    return fig


# -- RUN APP --- # 

if __name__ == '__main__':
    app.run_server(debug=True)