# Imports
import pandas as pd
import numpy as np
import names
import random
import sys
import math

def create_df(cols):
    '''
    cols: list of column names
    '''
    df = pd.DataFrame(columns=cols)
    return df

def random_names(num):
    '''
    num: the number of names needed
    '''
    names_list = []
    for i in range(num):
        names_list.append(names.get_full_name())
    return names_list

def random_coor(num, radius, center):
    '''
    num: number of coordinates needed
    radius: radius of desired area
    center: location to start from (center of desired area)
    '''

    coor = []
    r = radius/111300
    x0, y0 = center
    for i in range(num):
        u  = float(random.uniform(0.0,1.0))
        v = float(random.uniform(0.0,1.0))

        w = r * math.sqrt(u)
        t = 2 * math.pi * v
        x = w * math.cos(t)
        y = w * math.sin(t)

        xLat  = x + x0
        yLong = y + y0
        coor.append((xLat,yLong))

    return coor

from random import randrange
import datetime

def random_date(start,num):
    '''
    start: start date and time in datetime format
    num: number of dates needed
    '''

    t = []
    current = start
    for i in range(num):
        curr = current + datetime.timedelta(minutes=randrange(60), seconds=randrange(60))
        t.append(curr.strftime("%d/%m/%y %H:%M:%S"))
    return t

from random import randint

def random_digits(num, digits=11):
    '''
    num: number of phone numbers needed
    digits: number of digits in each phone number (default is 11)
    '''

    phone = []
    for i in range(num):
        range_start = 10**(digits-1)
        range_end = (10**digits)-1
        phone.append(randint(range_start, range_end))

    return phone

def binary(num):
    '''
    num: number of entries needed
    '''
    return np.random.randint(2, size=num)

def random_ints(num):
    '''
    num: number of integers needed
    '''
    list_int = []
    for i in range(num):
        list_int.append(random.randint(1,12))

    return list_int
