import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import dash_table 

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from pandas.api.types import is_string_dtype
import datetime
import ast 
import json

from generate_random_data import Generate_Random_Data as grd 

# comment
#!! issue
#@ unused
#$ replaced by something else

# --- INITIALIZE VARIABLES --- # 

# to be used as as options for multi-select 
ENTITY_TYPES = ['resource', 'civilian', 'blockage', 'duck']

#@ Unused 
# #!! color selection to stand out
# COLOR_MAP = {'blockage':'#969190'
#             , 'duck':'#f7e61b'
#             , 'resource':'#1b36f7'
#             , 'request':'#f71b43'}

#@ for generating civilian data
# KEEP_COLS = ['id', 'created_at', 'updated_at', 'event_type', 'device_type']


NEEDS = ['first_aid',
 'shelter',
 'food',
 'financial_aid',
 'water',
 'evacuation',
 'clothing',
 'hygiene',]

#!! general issues? needs better name to group under
ISSUES = ['medical',
 'fire_explosion',
 'flood',
 'violence',
 'landslide',
 'building_collapse',
 'trapped',
 'death',]

#!! needs better name to group under
SUB_ISSUES = ['electrical',
 'road_blocked',
 'fallen_trees',
 'chemical',
 'smoke_fire',
 'animals',
 'swiftwater',
 'explosives',
 'immobile']

# Create options for Checklist
TRUNC_NEEDS = ['medical', 'food','water', 'shelter'] # medical technically in diff category 
DISPLAY_NEEDS = ['Medical', 'Food', 'Water', 'Shelter']
help_request_options = [{'label': label, 'value': value} 
                            for label, value in zip(DISPLAY_NEEDS, TRUNC_NEEDS)]

#$ Replaced by below
# RELEVANT_COLS = ['id', 'updated_at', 'civilian.info.name', 
#         'civilian.info.occupants', 'civilian.info.phone',
#         'civilian.message','uuid', 'lat', 'lon', 'num_people',
#         'num_pets', 'public_name', 'change_status',
#         'needs', 'issues', 'sub_issues']

RELEVANT_COLS = ['name', 'phone',
        'message','duck_id', 'lat', 'lon', 'num_people',
        'num_pets', 'public_name', 'change_status',
        'needs', 'issues', 'sub_issues']

# use your own mapbox token saved in form {'token':'YOUR_TOKEN'}
# https://docs.mapbox.com/help/how-mapbox-works/access-tokens/
with open('./credentials.json', 'r') as f:
    credentials = json.load(f)
MAPBOX_TOKEN = credentials['token']
# px.set_mapbox_access_token(MAPBOX_TOKEN)

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# -- FUNCTIONS -- # 

#$ 
# def load_ducks(num_ducks=10):
#     """Query ducks table"""

#     #rand Generate ducks 
#     device_types = np.random.choice(['papa-duck', 'mama-duck', 'duck'], size=num_ducks)
#     duck_id_candidates = ['5QOZlOxJECBv', 'cVDXZ9fljbIh', 'Qnhq1G5Rdwuv', 'T1PaUmJjbFcO',
#        '0IUUAhmA99lq', '2lmrjdoJ237Z', 'hAkKlMLz8qgM', 'eeusfZ8AE8mv',
#        'xioGCMhFLPC8', 'iJWI1F0QggHr', 'RV5CEfaDzuTD', 'KxqoWN44irrF',
#        'mtldKD0Eslrz', '0P9pqRkY8yXv', 'NLZFD3yj4Oug', 'Hej9OwFoEdnR',
#        '1gIOGluTtwiW', 'VEuwAoRLhL25', 'r8Zm272RIu5s', 'Szo4vS0LIFrO',
#        'BZlCvDsdq6WR']
#     duck_ids = np.random.choice(duck_id_candidates, size=num_ducks)
#     ducks = pd.DataFrame({'device_type': device_types, 'device_id':duck_ids})
#     #rand
#     pr_coor = (18.5085, -67.07266)
#     pr_radius = 2500
#     ducks['location'] = grd.random_coor(num=num_ducks, radius=pr_radius, center=pr_coor)
#     ducks['location'] = ducks['location'].map(lambda x: str(x[0]) + ',' + str(x[1]))

#     # Unpack latitude and longitude -- VALIDATE ALWAYS EXIST as string with comma
#     ducks[['lat', 'lon']] = ducks['location'].str.split(pat=',', expand=True)
#     ducks['lat'] = ducks['lat'].astype(float)
#     ducks['lon'] = ducks['lon'].astype(float)

#     return ducks.drop(columns='location')


def load_resources():
    """Query resources table"""


    pass 

#@
def load_civilians():
    """Query requests table"""
        
    #!! replace with actual data load
    civilians = pd.read_csv('isabela_duck_deployment.csv'
                        , converters={'payload': ast.literal_eval})
    return civilians

#@
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
    # pr_coor = (18.510640,-67.052121)
    pr_radius = 2500

    # Unpack latitude and longitude -- VALIDATE whether location always exists
    payload[['lat', 'lon']] = payload['civilian.info.location']\
                                            .str.strip('[]')\
                                            .str.split(pat=',', expand=True)
    payload['lat'] = payload['lat'].astype(float)
    payload['lon'] = payload['lon'].astype(float)
    # Add jitter to original 
    payload['location'] = payload.apply(lambda x: grd.random_coor(num=1, radius=pr_radius, center=(x['lat'], x['lon']))[0], axis=1)
    payload['lat'] = payload['location'].map(lambda x:x[0])
    payload['lon'] = payload['location'].map(lambda x:x[1])

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


def concat_string(row):
    """Concatenate column with its content in a dict like display
    but broken by a newline via <br /> instead of ',' that you would get with
    using hovertext parameter in trace object. 
    """
    info = []
    for col in col_names:
        info.append(f"{col}: {row[col]}")
    return '<br />'.join(info)

def consolidate_category(row, cat_list):
    """List names of items ticked in SOS form
    Input
    -----
    row: df apply row
    cat_list: list, column names of relevant fields in category
    Returns
    -----
    A list of column names that were ticked in SOS form
    """
    requested_needs = [c for c in cat_list if row[c]==1]
    return tuple(requested_needs)

def consolidate_bools_for_table(civilian_data):
    """Apply consolidate_category to relevant columns
    data: df, civilian_data
    Concern: too much redundant data stored/sent?
    """
    data = civilian_data.copy()
    data['needs'] = data.apply(consolidate_category, axis=1, cat_list=NEEDS)
    data['issues'] = data.apply(consolidate_category, axis=1, cat_list=ISSUES)
    data['sub_issues'] = data.apply(consolidate_category, axis=1, cat_list=SUB_ISSUES)
    return data 

def pre_generated_civilian(path, civ_coordinate):
    '''works with specifically formatted csv
    Input
    -----
    path: str, file path to csv with pre-generated data
    civ_coordinate: str, name of column containing civilian location coordinates'''
    civilians = pd.read_csv(path
                            , converters={civ_coordinate:ast.literal_eval})
    civilians['lat'] = civilians[civ_coordinate].map(lambda x: x[0][0])
    civilians['lon'] = civilians[civ_coordinate].map(lambda x: x[0][1])
    return civilians

def load_ducks(civilian_data):
    """Load duck ids and locations from pre-generated csv"""

    ducks = civilian_data.groupby(['duck_id', 'duck_coordinates']).size().reset_index()
    ducks['duck_coordinates'] = ducks['duck_coordinates'].map(ast.literal_eval)
    ducks[['lat', 'lon']] = pd.DataFrame(ducks['duck_coordinates'].tolist())
    return ducks

# --- LOAD DATA --- # 
# load each source of data into its own df

#$ replaced by pre-generated below
# generate_requests
# civilians = load_civilians()
# payload = extract_civilian_payload(civilians)
# civilian_data = pd.concat([civilians[KEEP_COLS], payload], axis=1)

# Load pre-generated requests
civilian_data = pre_generated_civilian('./t_fake_data.csv', 'civilian_coordinates')

#!! Works but tool tip covers whole map and creates a redundant column - adjust shown columns
col_names = [x for x in civilian_data.columns if is_string_dtype(civilian_data[x])]
civilian_data['label'] = civilian_data.apply(concat_string, axis=1)

# consolidate categories
civilian_data = consolidate_bools_for_table(civilian_data)
print(civilian_data.iloc[:, -3:].head())


ducks = load_ducks(civilian_data)
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

    # Filter by entity
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

    

    html.Div([
        
        # Filter by request type
            # a single multi
        html.Div([
            html.P('Type of Help Requested:'),
            dcc.Checklist(
                id='filter-request-types',
                options=help_request_options,
                values=TRUNC_NEEDS
            )
            
            ],
            style={'width':'15%', 'float':'left'}
        ),


        html.Div([
        dcc.Graph(id='map'
                , style={'width':'85%', 'display':'inline-block'})
        ]),

    # Map
    
    ]),

        # Table 
        #!! Not displaying array within column properly
        #!! Make interactive - clicking on row highlights map and vice versa
    html.Div([
        html.H1("Civilian Event Details"),
        #!! Not yet interactive
        dash_table.DataTable(
            id='civilian_events'
            , columns =[
                {'name': i, 'id':i} for i in RELEVANT_COLS
            ]
            , data=civilian_data[RELEVANT_COLS].to_dict('records')
            , style_table={
                'overflowX': 'scroll'
                , 'overflowY': 'scroll'
                , 'maxHeight': '500px'
                }
            ,style_cell={
                'minWidth': '0px', 'maxWidth': '180px'
                , 'whiteSpace': 'normal'
                }
        )
    ]),

    # BANs - quick summary counts
    #!! CSS to format properly? 
    html.Div([
        html.H1('Summary Counts of Civilian Requests'),
        #!! Not interactive - not filtered - but should we allow clicking bar to filter?
            # would be tableau like but low priority
        dcc.Graph(
            id='requests-bar',
            figure=px.bar(civilian_data[TRUNC_NEEDS]\
                            .melt(var_name='Request Type')\
                            .groupby('Request Type', as_index=False)\
                            .sum(),
                        x='Request Type',
                        y='value')\
                    .update_yaxes(title='Number of Requests') # chained layout edit
        )
        #& replaced with bar chart - what are better BANs?
        # , html.Div(
        #     html.H2(f"Flood \n {civilian_data['flood'].sum()}")
        #     , style={'width': '19%', 'display': 'inline-block'})
        # , html.Div(
        #     html.H2(f"Medical \n {civilian_data['medical'].sum()}")
        #     , style={'width': '19%', 'display': 'inline-block'})
        # , html.Div(
        #     html.H2(f"Fire Explosion \n {civilian_data['fire_explosion'].sum()}")
        #     , style={'width': '19%', 'display': 'inline-block'})
        # , html.Div(
        #     html.H2(f"Building Collapse \n {civilian_data['building_collapse'].sum()}")
        #     , style={'width': '19%', 'display': 'inline-block'})
        # , html.Div(
        #     html.H2(f"Trapped \n {civilian_data['trapped'].sum()}")
        #     , style={'width': '19%', 'display': 'inline-block'})

    ]),

])




@app.callback(
    Output('map', 'figure'),
    [Input('filter-entity-types', "value"),
     Input('filter-request-types', 'values')]
)
def make_map(entity_types, request_types):

    #!! currently resetting zoom of map each time - need to save fig layout as state
    # Populate string with request type name and use pandas query
    request_mask = '|'.join([f'{col} == 1' for col in request_types])
    typed_data = civilian_data.query(request_mask)

    fig = go.Figure()

    # Add trace if requested by entity_types multi-select

    if 'duck' in entity_types:
        fig.add_trace(go.Scattermapbox(
                  lat=ducks['lat']
                , lon =ducks['lon']
                , mode='markers'
                , name='Active Ducks'
                , hovertext=ducks['duck_id']
                , marker={'symbol':'dog-park'} # can we color this?
        ))
    #!! How should we display the request information? Table? 
        #!! How to show request if asking for multiple things?
    if 'civilian' in entity_types:
        fig.add_trace(go.Scattermapbox(
                  lat=typed_data['lat']
                , lon=typed_data['lon']
                , mode='markers'
                , name='Civilian Requests'
                , hoverinfo='all'
                # , customdata # this is for dcc.Graph interactive properties, not initial display 
                , hovertext=typed_data['name'] # if using the pre-generated data
                # , hovertext=typed_data['label'] # alternative, list comprehension
                , marker={'size':8} 
        ))

    fig.update_layout(
        # title='Entity Map',#@ awkward placement
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
            zoom=10,
            style='satellite'
    ))
    #$ replaced by above
    # fig.update_mapboxes({'style':'satellite'
    #                  , 'center':{'lat': ducks['lat'].mean(), 'lon':ducks['lon'].mean()}
    #                 })
    return fig


# -- RUN APP --- # 

if __name__ == '__main__':
    app.run_server(debug=True)