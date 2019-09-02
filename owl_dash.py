# Dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

# Data tools imports
import pandas as pd
import numpy as np
import ast
import json

# Mapping imports
import folium
import plotly.plotly as py
import plotly.graph_objs as go

# Reading in the data
isabela_clusterdata = pd.read_csv('03-13 Isabella Clusterdata.csv')

#########################
### Mapping Functions ###
#########################

# Generates the map
def create_map(coordinates):
    return folium.Map(location=coordinates, zoom_start=16, tiles='cartodbdark_matter')

# Plots the ducks on the map
def map_ducks(m, df, papa_id='44E855A4AE30', android=False):
    if not android:
        for i in range(0,len(df)):
            if df.iloc[i]['device_id'] == papa_id:
                folium.CircleMarker(location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],
                                    radius=3,
                                    color='blue',
                                    fill=True,
                                    fill_color='blue',
                                    popup="<br>".join([str(df.iloc[i]['device_id']), 'PapaDuck'])
                                   ).add_to(m)
            else:
                folium.CircleMarker(location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],
                                    radius=2,
                                    color='red',
                                    popup="<br>".join([str(df.iloc[i]['device_id']), str(df.iloc[i]['device_type'])])
                                   ).add_to(m)
    else:
        for j in range(0,len(df)):
            folium.CircleMarker(location=[df.iloc[j]['latitude'], df.iloc[j]['longitude']],
                                radius=2,
                                color='purple',
                                popup="<br>".join([str(df.iloc[j]['uuid']), str(df.iloc[j]['from_device_id'])])
                                ).add_to(m)
    return m

# Plots the path of the messages and saves it as html
def plot_path(gps_location, clusterdata, device_observations, papa_id='44E855A4AE30', arrow=True):
    #clusterdata = clusterdata.drop_duplicates(subset='path', keep='last')
    for idx,val in enumerate(clusterdata['path_coordinates']):
        test = folium.Map(location=gps_location, zoom_start=14, tiles='cartodbdark_matter')
        for i in range(0,len(device_observations)):
            if device_observations.iloc[i]['device_id'] == papa_id:
                folium.CircleMarker(location=[device_observations.iloc[i]['latitude'], device_observations.iloc[i]['longitude']],
                                    radius=4,
                                    color='blue',
                                    fill_color='blue',
                                    popup="<br>".join([str(device_observations.iloc[i]['device_id']), 'PapaDuck'])
                                   ).add_to(test)
            else:
                folium.CircleMarker(location=[device_observations.iloc[i]['latitude'], device_observations.iloc[i]['longitude']],
                                    radius=3,
                                    color='red',
                                    fill_color='red',
                                    popup="<br>".join([str(device_observations.iloc[i]['device_id'])])
                                   ).add_to(test)
        folium.PolyLine(val, weight=1, color='green').add_to(test)
        if arrow:
            arr = []
            if len(val)>1:
                for i in range(len(val)-1):
                    if (len(val[i]) != 0 and len(val[i+1]) != 0):
                        arr.append(get_arrows(locations=[val[i], val[i+1]]))
                for tri in arr:
                    tri[0].add_to(test)
        for j in val:
            if len(j)==2:
                folium.CircleMarker(location=[j[0], j[1]],
                                    radius=1,
                                    color='green'
                                    ).add_to(test)
        test.save(outfile=str(idx)+'isabela.html')
    return print('Complete!')

isabela_map = create_map([18.509883, -67.067172])
isabela_map.save('isabela_map.html')

########################
### Helper Functions ###
########################

# Function that extracts coordinates from the 'payload' column
def get_isabela_coordinates(x):
    try:
        return list(ast.literal_eval(ast.literal_eval(x).get('civilian').get('info').get('location')))
    except:
        return None

# Getting the device path from the 'path'
def get_path(x, coordinates_df):
    path = []
    if x is not None:
        for i in x.split(','):
            try:
                path.append(coordinates_df[coordinates_df['device_id'] == i][['latitude', 'longitude']].values[0])
            except:
                path.append(coordinates_df[coordinates_df['device_id'] == i][['latitude', 'longitude']].values)
        return path
    else:
        return path

# Function that extracts the path of message
def get_isabela_path(x):
    try:
        return x.get('path')
    except:
        return None

# Getting coordinates

def get_lat_lon(df, col='coordinates'):
    return pd.DataFrame(df[col].tolist(), index=df.index)

#####################
### Data Cleaning ###
#####################

## Getting path and coordinates respectively from 'payload'
#isabela_clusterdata['path'] = isabela_clusterdata['payload'].apply(get_isabela_path)

# Filtering out the bad data
isabela_clusterdata['coordinates'] = isabela_clusterdata['payload'].apply(get_isabela_coordinates)
isabela_clusterdata['nick'] = isabela_clusterdata['coordinates'].apply(lambda x: x is not None)
data_isabela = isabela_clusterdata[isabela_clusterdata['nick']==True]
data_isabela = data_isabela[data_isabela.index<=631]

# Getting the lat and lon from 'payload'
data_isabela[['latitude', 'longitude']] = get_lat_lon(data_isabela)

# Removing the false device_id
data_isabela.drop(['device_id'], axis=1, inplace=True)

# Getting the true path
data_isabela['id_path'] = data_isabela['payload'].apply(lambda row: ast.literal_eval(row).get('path'))

# Removing duplicate paths
data_isabela.drop_duplicates(subset='id_path', keep='last', inplace=True)

# Getting the first duck from the path
data_isabela['device_id'] = data_isabela['id_path'].apply(lambda x: x[:12])

# Device locations for Isabela
device_observations = data_isabela[['device_id', 'latitude', 'longitude']]

# Removing duplicates
device_observations.drop_duplicates(subset='device_id', keep='last', inplace=True)

# Adding devices that were not recorded
device_observations = device_observations.append(pd.DataFrame([['705BA9BF713C', 18.508743, -67.072898],['AC65ABBF713C',18.508729, -67.072763]], columns=['device_id', 'latitude', 'longitude']))

# Checking to see if what ducks are in the path for each message
for i in device_observations['device_id']:
    data_isabela[str(i)] = data_isabela['id_path'].apply(lambda x: 1 if i in x else 0)

# Counting the number of times this duck was used during transmission
device_observations['count'] = device_observations['device_id'].apply(lambda x: data_isabela[x].sum())

# Combining the lat and long into one column
device_observations['coordinates'] = device_observations[['latitude', 'longitude']].values.tolist()

# Calculaing the distance of papa duck and the current duck
from geopy.distance import vincenty
device_observations['dist_from_papa'] = device_observations['coordinates'].apply(lambda x: vincenty(x, [18.508568, -67.072667]).m)

# Calculaing the number of paths one duck has to reach papa duck
data_isabela['len_of_path'] = data_isabela['id_path'].apply(lambda x: len(x.split(',')))

#################
### Dashboard ###
#################

# Create global chart template
mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'  # noqa: E501

layout = dict(
    autosize=True,
    height=500,
    font=dict(color='#CCCCCC'),
    titlefont=dict(color='#CCCCCC', size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor="#191A1A",
    paper_bgcolor="#020202",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="dark",
        center=dict(
            lon=-67.067172,
            lat=18.509883
        ),
        zoom=12,
    )
)

d = [
    go.Scattermapbox(
        lat = device_observations['latitude'].tolist(),
        lon = device_observations['longitude'].tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(size=9),
        text=device_observations['device_id'].tolist()
    )
]

app = dash.Dash(name='__main__')

app.layout = html.Div([
    html.Div([
        html.H1(
            'Ducks Deployed'
        ),
        dcc.Graph(
            id='main-map',
            figure = dict(data=d, layout=layout)
        )
    ]),
    html.Div([
        html.H3(
            'Stats'
        ),
        html.Div(
            id='stat_output',
            style={'margin-top':'10'}
        ),
        html.Div(
            dcc.Graph(id='individual_duck'),
            style={'margin-top':'10'}
        )
    ]),
    html.Div([
        dcc.Slider(
            id='duck_slider',
            min=0,
        )
    ])
])

# main-map
# @app.callback(
#     Output('main-map', 'figure'),
#     [Input('yes_no', 'value')]
# )
# def make_map(yes_no):
#     if yes_no=='y':
#         d = [
#             go.Scattermapbox(
#                 lat = device_observations['latitude'].tolist(),
#                 lon = device_observations['longitude'].tolist(),
#                 mode='markers',
#                 marker=go.scattermapbox.Marker(size=9),
#                 text=device_observations['device_id'].tolist()
#             )
#         ]
#         figure = dict(data=d, layout=layout)
#     else:
#         figure = dict(data=[], layout=layout)
#     return figure

@app.callback(
    Output('stat_output', 'children'),
    [Input('main-map', 'clickData')]
)
def filter_duck(main_map):
    if main_map is None:
        pass
        # main_map = json.loads('{"points": [{\
        #     "curveNumber": 0,\
        #     "pointNumber": 0,\
        #     "pointIndex": 0,\
        #     "lon": -67.072667,\
        #     "lat": 18.508568,\
        #     "text": "44E855A4AE30"}]}')
    else:
        duck = device_observations[device_observations['device_id']==main_map['points'][0]['text']]
        return ('Distance from Papa Duck: {} m'+"\n"+'Messages Transmitted: {}').format(duck['dist_from_papa'].values[0], duck['count'].values[0])

@app.callback(
    Output('individual_duck', 'figure'),
    [Input('main-map', 'clickData')]
)
def map_paths(main_map):
    duck = device_observations[device_observations['device_id']==main_map['points'][0]['text']]

# @app.callback(
#     Output('main-map', 'figure'),
#     [Input('yes_no', 'value')]
# )
# def get_map(yes_no):
#     if yes_no == 'yes':
#         return

if __name__ == '__main__':
    app.run_server(debug=True)
