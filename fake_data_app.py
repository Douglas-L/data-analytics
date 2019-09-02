### IMPORTS ###

### Standard Data Imports
import pandas as pd
import numpy as np
import ast
import json
import textwrap

### Random Imports
from generate_random_data import Generate_Random_Data as grd
import random
random.seed(0)

### Dash App Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

### Mapping Imports Everything
import plotly.graph_objects as go
import plotly.express as px

## MAPBOX CREDENTIALS ###
with open('./credentials.json', 'r') as f:
    credentials = json.load(f)
MAPBOX_TOKEN = credentials['token']

# CSS 
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# for debugging
pd.set_option('max_colwidth', -1)
pd.set_option('display.max_columns', 200)

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
## What does this do? 
# print(fake_data['duck_id'])
### Generating random data to fill in dataframe

FAKE_DATA_LEN = fake_data.shape[0]

# Name data
fake_data['name'] = grd.random_names(num=FAKE_DATA_LEN)

# Phone data
fake_data['phone'] = grd.random_digits(num=FAKE_DATA_LEN)

# Occupants data
fake_data['num_people'] = grd.random_ints(num=FAKE_DATA_LEN)

# Pets data
fake_data['num_pets'] = grd.random_ints(num=FAKE_DATA_LEN)

# Data for binary columns
for col in ['medical', 'food', 'financial_aid', 'water']:
        fake_data[col] = grd.binary(num=FAKE_DATA_LEN)

# Function to get coordinates of each duck
def get_coordinates(row):
    if isinstance(isabela.loc[isabela['duck_id']==row['duck_id'], 'coordinates'].values[0], str):
        center = [ast.literal_eval(i) for i in isabela.loc[isabela['duck_id']==row['duck_id'],'coordinates']][0]
    else:
        center = isabela.loc[isabela['duck_id']==row['duck_id'], 'coordinates'][0]
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


# Parse the duck message path from string into a tuple
def extract_path(path):
    remove_array = path.replace('array', '').replace('(', '').replace(')', '')
    list_of_ducks = ast.literal_eval(remove_array)
    # use tuple of tuples so order is preserved and hashable to determine uniqueness
    duck_tuples = [tuple(duck) for duck in list_of_ducks]
    duck_tuples = tuple(duck_tuples)
    return duck_tuples

# Data cleaning of path
def clean_the_path(path):
    if isinstance(path, str):
        return ast.literal_eval(path.replace('array', '').replace('(', '').replace(')', ''))
    else:
        return path

isabela['clean_path'] = isabela['path_coordinates'].map(clean_the_path)

# Getting the first duck_id
isabela['first_duck'] = isabela['path'].apply(lambda x: x[:12])
print(isabela.dtypes)
print(isabela.head())


### Definitions ### 

ENTITY_TYPES = ['resource', 'civilian', 'blockage', 'duck']
ENTITY_TYPE_LABELS = ['Resources', 'Civilian Requests', 'Blockages', 'Duck Network']

TRUNC_NEEDS = ['medical', 'food','water']

### Dashboard ###

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)

YN = {'Yes':1, 'No':0}
option = [{'label':i, 'value':i} for i in [1,0]]

# needed when assigning callbacks to components that are generated 
# by other callbacks (and therefore not in the initial layout)
app.config['suppress_callback_exceptions']=True 

# app.scripts.config.serve_locally=True



app.layout = html.Div([
    # Title
    html.H1("OWL Incident Command Map"),

    dcc.Tabs(
            id='tabs-display', 
            value='tab-1-requests', 
            children= [
                dcc.Tab(label='Civilian Requests', value='tab-1-requests'),
                dcc.Tab(label='Duck Network Status', value='tab-2-network'),
            ]),
    
    html.Div(id='tabs-content')

])
    
    

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-display', 'value')])
def render_content(tab):
    if tab == 'tab-1-requests':
        print('rendering tab1')
        return html.Div([
                    html.Div([
                        # multi-select 
                        dcc.Dropdown(
                            id='filter-entity-types',
                            options=[{'label': label, 'value':et} for label, et 
                                        in zip(ENTITY_TYPE_LABELS, ENTITY_TYPES)],
                            value=ENTITY_TYPES,
                            multi=True
                        )
                    ]),
                    html.Div([
                        # Dropdown menu for Medical
                        html.P(
                            'Medical Needed:',
                            ),
                        html.P(
                            dcc.Dropdown(
                                id='medical-filter',
                                options=option,
                                value=[YN.get(i) for i in YN],
                                multi=True,
                                )    
                            ),
                        # Dropdown menu for Food
                        html.P(
                            'Food Needed:',
                            ),
                        html.P(
                            dcc.Dropdown(
                                id='food-filter',
                                options=option,
                                value=[YN.get(i) for i in YN],
                                multi=True,
                                )
                            ),
                        # Dropdown menu for Water
                        html.P(
                            'Water Needed:',
                            ),
                        html.P(
                            dcc.Dropdown(
                                id='water-filter',
                                options=option,
                                value=[YN.get(i) for i in YN],
                                multi=True,
                                )
                            ),

                        # Show hover info
                        dcc.Markdown(textwrap.dedent(
                            """
                            **Hover Data**
                            """)
                            ),
                        html.Pre(
                            id='hover-data',
                            style=styles['pre']
                            )
                        ], 
                        style={"width": "15%", "float": "left"}
                    ),

                    # Map
                    html.Div([
                        dcc.Graph(id='map-requests',
                                style={'width':'85%', 'display':'inline-block'})
                    ]),

                    html.Div([
                        html.H1('Summary Counts of Civilian Requests'),
                        #!! Not interactive - not filtered - but should we allow clicking bar to filter?
                            # would be tableau like but low priority
                        dcc.Graph(
                            id='requests-bar',
                            figure=px.bar(fake_data[TRUNC_NEEDS]\
                                            .melt(var_name='Request Type')\
                                            .groupby('Request Type', as_index=False)\
                                            .sum(),
                                        y='Request Type',
                                        x='value',
                                        orientation='h')\
                                    .update_yaxes(title='Number of Requests'), # chained layout edit
                            style={'width':'50%', 'display':'inline-block'}
                        ),
                    ]),
                    
                ]) # end tab 1 div

                
    elif tab == 'tab-2-network':
        print('rendering tab 2')
        return html.Div([
                    html.Div([
                        html.P(
                            'Duck Path'
                        ),
                        html.P(
                            dcc.Dropdown(
                                id='duck_id',
                                options=[{'label':i, 'value':i} for i in isabela['first_duck'].unique()],
                                )
                            ),
                        html.P(
                            dcc.Dropdown(
                                id='path_id',
                                )
                            ),
                        ], 
                        style={"width": "15%", "float": "left"}
                    ),
                    # Map 
                    html.Div([
                        dcc.Graph(id='map-network',
                                style={'width':'85%', 'display':'inline-block'})
                    ]),
                ]) # end tab 2 div

@app.callback(
    Output('map-requests', 'figure'),
    [Input('medical-filter', 'value'),
     Input('food-filter', 'value'),
     Input('water-filter', 'value'),
    ]
)
def map_graph(med, food, water): #, duck_id, path_id
    df = fake_data[fake_data['medical'].isin(med)]
    df = df[df['food'].isin(food)]
    df = df[df['water'].isin(water)]
    print(df.shape)
    fig = go.Figure()
    # add ducks
    fig.add_trace(go.Scattermapbox( 
        #!! Could this be done during preprocessing? 
        lat=isabela['coordinates'].apply(lambda x: ast.literal_eval(x)[0]).tolist(),
        lon=isabela['coordinates'].apply(lambda x: ast.literal_eval(x)[1]).tolist(),
        marker={
            'size':10,
            'color':'#FFFF00'
        }
    ))


    fig.add_trace(go.Scattermapbox(
        lat=df['civilian_latitude'],
        lon=df['civilian_longitude'],
        mode='markers',
        name='Civilian Requests',
        hoverinfo='all',
        customdata= df['name'], #!! for hover box - replace with more informative label
    ))
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        showlegend=False,
        clickmode='event+select',
        mapbox=go.layout.Mapbox(
            accesstoken=MAPBOX_TOKEN,
            center=go.layout.mapbox.Center(
                    lat=fake_data['duck_latitude'].mean(),
                    lon=fake_data['duck_longitude'].mean()
                    ),
            zoom=14,
            ),
        )
    return fig

@app.callback(
    Output('hover-data', 'children'),
    [Input('map-requests', 'hoverData')]
)
def display_hover_data(hoverData):
    '''Display custom info when hovering over an entity
    
    Note: Flask auto escapes html, so need to replace <br> with \n 
    while Dash hovertext accepts <br> as linebreak butnot \n'''

    point = hoverData['points'][0] # go down to hovertext level
    render_text = point['customdata'].replace('<br>', '\n') #@ for custom label, see entity_map_app.py
    return render_text


@app.callback(
    Output('path_id', 'options'),
    [Input('duck_id','value')]
)
def get_options(duck_id):
    """Filter paths to only those beginning at 'first_duck'
    """
    df = isabela[isabela['first_duck']==duck_id]
    return [{'label':idx, 'value':idx} for idx,val in enumerate(df['clean_path'])]

@app.callback(
    Output('map-network', 'figure'),
    [Input('duck_id','value'),
     Input('path_id','value')
    ]
)
def map_network(duck_id, path_id):


    fig = go.Figure()
    # add ducks
    fig.add_trace(go.Scattermapbox( 
        #!! Could this be done during preprocessing? 
        lat=isabela['coordinates'].apply(lambda x: ast.literal_eval(x)[0]).tolist(),
        lon=isabela['coordinates'].apply(lambda x: ast.literal_eval(x)[1]).tolist(),
        marker={
            'size':10,
            'color':'#FFFF00'
        }
    ))

    if duck_id:
        dataframe=isabela[isabela['first_duck']==duck_id]
        if path_id:
            try:
                path = dataframe.iloc[path_id]['clean_path']
            except:
                path = dataframe.iloc[0]['clean_path']
            fig.add_trace(go.Scattermapbox(
                lat = [i[0] for i in path],
                lon = [i[1] for i in path],
                mode='lines+markers',
            ))
    else:
        pass


    fig.update_layout(
        autosize=True,
        hovermode='closest',
        showlegend=False,
        clickmode='event+select',
        mapbox=go.layout.Mapbox(
            accesstoken=MAPBOX_TOKEN,
            center=go.layout.mapbox.Center(
                    lat=fake_data['duck_latitude'].mean(),
                    lon=fake_data['duck_longitude'].mean()
                    ),
            zoom=14,
            ),
        )
    return fig

### RUNNING APP ###

if __name__ == '__main__':
    app.run_server(debug=True)



# Questions to ask:
# Are there specific scenario's for medical emergencies?
# When would you need more details when it is a medical emergency?
# What is the average Responder prepared for? What are they not prepared for?
# Are there any special Responder units for particular scenarios?
# In a desparate situation, when you (the Responder) cannot get in touch with IC, what would be useful information to see in a map?
# What key locations do you (the Responder) need to have on a map?
# Being able to identify who is submitting a message
# Selecting the emergencies that command center can handle
# First Responders Priortize the hover database
#



