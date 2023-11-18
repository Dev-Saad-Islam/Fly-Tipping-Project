from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


# Read data from the dataset
data = pd.read_excel("Group Project Flytipping Data.xlsx")
data = data[data["AreaCode"] == "LA"]

# Plotting data on map with size as labels
fig = px.scatter_mapbox(data,
                        lat="Custom.LATITUDE",
                        lon="Custom.LONGITUDE",
                        hover_name="AddressLine1",
                        hover_data=["AddressLine1"],
                        color="Size",
                        zoom=10,
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# Handling missing values
data = data.dropna()

# Feature selection (only selecting geo-location data)
df_location = data[['Custom.LONGITUDE','Custom.LATITUDE']]

# Standardization on selected features
scaler = StandardScaler()
standardisedData = pd.DataFrame(scaler.fit_transform(df_location))

# Convert final data to NumPy array
data_train = standardisedData.to_numpy()

# Use Elbow method to get the best possible number of clusters from a given range
model_test = KMeans()
visualizer = KElbowVisualizer(model_test, k=(2,100), timings= True)
visualizer.fit(standardisedData)
visualizer.show()

# Training Kmeans with n_clusters from the result of Elbow method
model = KMeans(n_clusters = visualizer.elbow_value_)

# Get labels for all dataset
label = model.fit_predict(data_train)
# Add labels to dataframe for visualization
data["label"] = label

# Show final clusters on map with predictions as labels
fig = px.scatter_mapbox(data,
                        lat="Custom.LATITUDE",
                        lon="Custom.LONGITUDE",
                        hover_name="label",
                        hover_data=["label"],
                        color="label",
                        zoom=12,
                        height=800,
                        width=1500)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_image('fig.png', engine= "kaleido")

