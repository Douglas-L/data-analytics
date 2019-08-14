### IMPORTS ###

### Standard Data Imports
import pandas as pd
import numpy as np
import ast

### Random Imports
from generate_random_data import Generate_Random_Data as grd
import random
random.seed(0)

### Dash App Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

### Mapping Imports
import plotly.graph_objects as go
import plotly.express as px

### MAPBOX CREDENTIALS ###
# with open('./credentials.json', 'r') as f:
#     credentials = json.load(f)
# MAPBOX_TOKEN = credentials['token']
px.set_mapbox_access_token('pk.eyJ1IjoidHFyYWhtYW4iLCJhIjoiY2l0bmh2dnU2MDRvZzJ6bDQ4OWFheXU3NCJ9.bY7m05QGUHV1jQvwwHX-FA')
#mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'
# Reading in the data
isabela = pd.read_csv("isabela_duck_deployment.csv")

### PREPROCESSING ###

# Getting the first duck_id from path
isabela['duck_id'] = isabela['path'].apply(lambda x: x[:12])

# Creating a list of unique duck_id's
duck_list = isabela['duck_id'].unique().tolist()

### Creating a dataframe with fake data

# Creating a list of column names for dataframe
columns = ['time','name', 'medical', 'food', 'financial_aid', 'water']
# Generating dataframe with those columns
fake_data = grd.create_df(cols=columns)

# Generating a list of random duck_id's from duck_list and adding it into dataframe
ids = []
for i in duck_list:
    for r in range(random.randint(1,6)):
        ids.append(i)
fake_data['duck_id'] = ids

### Generating random data to fill in dataframe

# Name data
fake_data['name'] = grd.random_names(num=fake_data.shape[0])

# Phone data
fake_data['phone'] = grd.random_digits(num=fake_data.shape[0])

# Occupants data
fake_data['num_people'] = grd.random_ints(num=fake_data.shape[0])

# Pets data
fake_data['num_pets'] = grd.random_ints(num=fake_data.shape[0])

# Data for binary columns
for col in ['medical', 'food', 'financial_aid', 'water']:
        fake_data[col] = grd.binary(num=fake_data.shape[0])

# Function to get coordinates of each duck
def get_coordinates(row):
    if isinstance(isabela.loc[isabela['duck_id']==row['duck_id'],:]['coordinates'].values[0], str):
        center = [ast.literal_eval(i) for i in isabela.loc[isabela['duck_id']==row['duck_id'],:]['coordinates']][0]
    else:
        center = isabela.loc[isabela['duck_id']==row['duck_id'],:]['coordinates'][0]
    distance = random.randint(30,150)
    return (center, grd.random_coor(num=1, radius=distance, center=center))

# Applying 'get_coordinates' to get the coordinates of duck and civilians
fake_data['coordinates'] = fake_data.apply(get_coordinates, axis=1)

# Separating 'coordinates' into 'duck_coordinates' and 'civilian coordinates'
fake_data[['duck_coordinates', 'civilian_coordinates']] = fake_data['coordinates'].apply(pd.Series)

# Extracting the ducks' latitude and longitude from 'duck_coordinates'
fake_data['duck_latitude'] = fake_data['duck_coordinates'].apply(lambda x: x[0])
fake_data['duck_longitude'] = fake_data['duck_coordinates'].apply(lambda x: x[1])

# Extracting the civilians' latitude and longitude from 'civilian_coordinates'
fake_data['civilian_latitude'] = fake_data['civilian_coordinates'].apply(lambda x: x[0][0])
fake_data['civilian_longitude'] = fake_data['civilian_coordinates'].apply(lambda x: x[0][1])

### Dashboard ###

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)

YN = {'Yes':1, 'No':0}
option = [{'label':i, 'value':i} for i in [1,0]]

app.config['suppress_callback_exceptions']=True
# app.scripts.config.serve_locally=True

app.layout = html.Div([
    # Title
    html.H1("OWL Incident Command Map"),

    # Filter for Medical Attention
    html.Div([
        # Dropdown menu for Medical
        html.P(
            'Medical Needed:',
            ),
        html.P(
            dcc.Dropdown(
                id='medical-fiter',
                options=option,
                value=[1,0],
                multi=True,
                )
            ),
        # # Dropdown menu for Food
        # html.P(
        #     'Food Needed:',
        #     ),
        # html.P(
        #     dcc.Dropdown(
        #         id='food-filter',
        #         options=options,
        #         value=[YN.get(i) for i in YN],
        #         multi=True,
        #         )
        #     ),
        # # Dropdown menu for Water
        # html.P(
        #     'Water Needed:',
        #     ),
        # html.P(
        #     dcc.Dropdown(
        #         id='water-filter',
        #         options=options,
        #         value=[YN.get(i) for i in YN],
        #         multi=True,
        #         )
        #     ),

        ],
             style={"width": "15%", "float": "left"},
        ),

    # Map
    html.Div([
        dcc.Graph(id='map',
                  style={'width':'85%', 'display':'inline-block'})
    ])
])

@app.callback(
    Output('map', 'figure'),
    [Input('medical-filter', 'value')]
)
def map_graph(med):
    df = fake_data[fake_data['medical'].isin(med)]
    fig = px.scatter_mapbox(df,
                            lat='civilian_latitude',
                            lon='civilian_longitude',
                            # mode='markers',
                            # marker=go.scattermapbox.Marker(size=9),
                            zoom=15)
    fig.update_mapboxes({'style':'satellite',
                         'center':{'lat': fake_data['duck_latitude'].mean(),
                                   'lon': fake_data['duck_longitude'].mean()}
                         })
    return fig

### RUNNING APP ###

if __name__ == '__main__':
    app.run_server(debug=True)
