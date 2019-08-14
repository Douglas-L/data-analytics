
# coding: utf-8

# In[1]:


# Imports

import numpy as np
import pandas as pd
import geopy
from geopy.distance import VincentyDistance


# In[2]:


def duck_coordinates(mdc, bearing=90*np.arange(0,4,.8), land_type='urban', clusters=1, duck_type='duckling'):
    '''
    Creates coordinates for mama ducks or ducklings
    
    Parameters:
    mdc: (mother duck coordinate) coordinate of center of clusterduck
    
    bearing: how far each new duck is separated from each other in respect to the mdc (in degrees). Default is set
             to [0, 72, 144, 216, 288].
    
    land_type: can be one out of three possibilities ('urban', 'suburban', or 'rural'). Determines the distance between 
               central duck and duckling. Default is set to 'urban'
    
    clusters: number of cluster ducks wanted. Default is set to 1
    
    duck_type: type of duck ('mama', 'papa', or 'duckling' ). Default is 'duckling'
    
    Returns:
    cluster: a dictionary of coordinates for each cluster
    duck_coordinates: coordinates of all the ducks
    '''
    cluster={}
    duck_coordinates = []
    for i in range(clusters):
        coordinates = []
        for coor in mdc:
            lat, lon = coor
            origin = geopy.Point(lat, lon)
            if duck_type=='mama':
                if land_type=='urban':
                    d = 0.804672
                elif land_type=='suburban':
                    d = 2.01168
                else:
                    d = 4.02336
                bearing = 90*np.arange(0.4,4, .8)
                for b in bearing:
                    destination = VincentyDistance(kilometers=d).destination(origin, b)
                    coordinates.append([destination.latitude, destination.longitude])
            elif duck_type=='duckling':
                if land_type=='urban':
                    d = 0.402336
                elif land_type=='suburban':
                    d = 2.01168
                else:
                    d = 4.02336
                for b in bearing:
                    destination = VincentyDistance(kilometers=d).destination(origin, b)
                    coordinates.append([destination.latitude, destination.longitude])
            elif duck_type=='papa':
                pass
#                 if land_type=='urban':
#                     d = 0.804672
#                 elif land_type=='suburban':
#                     d = 2.01168
#                 else:
#                     d = 4.02336
#                 destination = VincentyDistance(kilometers=d).destination(origin, b)
#                 coordinates.append([destination.latitude, destination.longitude])
            else:
                print('That type of duck went extinct!')
        cluster[i] = coordinates
        mdc = coordinates
        duck_coordinates+=coordinates
    return cluster, duck_coordinates


# In[3]:


def plot_ducks(center, duck_coordinates, radius=402.336, circle=False, duck_type='duckling', tiles='OpenStreetMap'):
    '''
    Plots the ducks on a Folium map with its range (displaying range is optional)
    
    Parameters:
    center: location of the incident
    duck_coordinates: coordinates of the ducks
    radius: radius of range in meters. Default is 402.336 meters 
    circle: display the range of each duck. Default is set to False
    duck_type: the type of duck ('duckling', 'mama', 'papa'). Determines the color and width of the point on the map
    tiles: the style of map preferred from the Folium options
    
    Return:
    m: a Folium object (map with the ducks)
    '''
    
    m = folium.Map(location=[center], zoom_start=18, tiles=tiles)
    
    if duck_type == 'mama':
        color = 'yellow'
        width = 4
    elif duck_type == 'duckling':
        color = 'green'
        width = 2
    elif duck_type == 'papa':
        color = 'red'
        width = 6
    else:
        print("That type of duck went extinct!")
    
    for i in duck_coordinates:
        folium.CircleMarker(location=[i[0], i[1]],
                            radius=width,
                            color=color,
                            fill=True,
                            popup=",".join([str(j) for j in i])).add_to(m)
        
        if circle:
            folium.Circle(location=[i[0], i[1]],
                          color=color,
                          radius=radius).add_to(m)
    return m

