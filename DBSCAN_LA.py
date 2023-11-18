# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:13:50 2022

@author: Yifan Shao
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

from math import pi, cos, sin, atan2, sqrt, radians, atan, fabs,asin

def haversine(latlon1, latlon2): #get the real distance
    if (latlon1 - latlon2).all():
        lat1, lon1 = latlon1 #location of first place
        lat2, lon2 = latlon2 #location of second place
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 #get the distance
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a)) 
        r = 6370996.81  #r of earth
        distance = c * r #Get the spherical distance
    else:
        distance = 0
    return distance



df = pd.read_excel("Group Project Flytipping Data.xlsx")
df_output = df[df["AreaCode"] == "LA"] #only use data in LA city
df_la = df[df["AreaCode"] == "LA"]
df_la = df_la[["Custom.LATITUDE","Custom.LONGITUDE"]] #only anlyse the location

# try_eps = [] #try to find best eps and min_samples
# for eps in np.arange(30,180,5): #try for every number
#     for min_samples in np.arange(30,120,5):
#         dbscan = DBSCAN(eps = eps, min_samples = min_samples, algorithm='ball_tree', metric=haversine).fit(df_la)
#         clusters_num = len([i for i in set(dbscan.labels_) if i != -1]) #how many clusters
#         outliners_num = np.sum(np.where(dbscan.labels_ == -1, 1, 0)) #how many outliners
#         cluster_result = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values) #how many point in each clusters
#         try_eps.append({'eps':eps,'min_samples':min_samples,'clusters_num':clusters_num,'outliners_num':outliners_num,'cluster_result':cluster_result})    
# df_try_eps = pd.DataFrame(try_eps)
# df_try_eps.to_excel('try_eps_LA.xlsx') 


clustered_data = DBSCAN(eps=140, min_samples=75, algorithm='ball_tree', metric=haversine).fit(df_la) #the clustering for fewer spots #spot 80 40
labels = clustered_data.labels_
df_la["labels"] = labels #add the labels to the dataframe 
df_output["Hotspots"] = labels
df_la_2 = df_la[df_la["labels"] != -1] #delet the outliners
fig = px.scatter_mapbox(df_la_2,
                        lat="Custom.LATITUDE",
                        lon="Custom.LONGITUDE",
                        color = "labels",
                        zoom=10,
                        height=800,
                        width=800)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show() #output the plot
df_output.to_excel('data_with_hotspots_LA.xlsx') 