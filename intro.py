import streamlit as st
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import utm
import numpy as np
import pydeck as pdk
from PIL import Image

st.set_page_config(page_title="XECUTERS",
                   page_icon=Image.open('ai_dump_truck.png'))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)

# Functions
def calculate_distance(northing1, easting1, northing2, easting2):
    """Calculate the distance between two points in UTM coordinates"""
    dx = easting2 - easting1
    dy = northing2 - northing1
    distance = math.sqrt(dx**2 + dy**2)
    return distance

def identify_road_type(elevation):
    """Identify the type of road surface based on the elevation"""
    if elevation < 100:
        return 'flat'
    elif elevation < 200:
        return 'slightly uphill'
    else:
        return 'steep uphill'

# Load the data into a Pandas dataframe
# @st.cache(allow_output_mutation=True, max_entries=5)
def load_data():
    dataframe = pd.read_csv('data_group0_out.csv')
    return dataframe


st.title('Teck Resources Mining Data Machine Learning Modelling')
st.info("Goal: For a mining operation, find ways to move the same amount of material from the shovels to dumps with the least amount of fuel consumed.")
# dataframe = pd.read_csv('data_group0_out.csv')
dataframe = load_data()

st.subheader("Data Analysis using Teck Resource's file converted to CSV format")

st.caption("Data types")
st.write(dataframe.dtypes)
st.caption("Simple glipmse of the pre-processed and cleaned data")
st.write(dataframe.head())
st.caption("Descriptive statistics of the data")
st.write(dataframe.describe())


truck_test = dataframe.loc[(dataframe['GPSELEVATION'] == 308.763) & (dataframe['TRUCK_ID'] == 32) & (dataframe['FUEL_RATE'] == 196)]
truck_test_northing = truck_test.values.tolist()[0][2]
truck_test_easting = truck_test.values.tolist()[0][3]

# Convert the UTM coordinates to latitude and longitude
lat, lon = utm.to_latlon(truck_test_easting, truck_test_northing, 18, 'S')

truck_index = pd.Index(range(0, 21349, 1))
truck_test = dataframe.loc[(dataframe['TRUCK_ID'] == 3)]
truck_copy = truck_test.set_index(truck_index)

# df_coords = pd.DataFrame()
st.title("Map of truck #3's location points using coordinates")
st.subheader("Hauling Status")
df_coords = []
for index, row in truck_copy[:10000].iterrows():
    temp_status = truck_copy.iloc[index]["STATUS"]
    temp_elev = truck_copy.iloc[index]["GPSELEVATION"] + 1220
    temp_lat = truck_copy.iloc[index]["GPSNORTHING"]
    temp_lon = truck_copy.iloc[index]["GPSEASTING"]
    if ((temp_lat > 0) & (temp_status == 'Hauling')):
        lat, lon = utm.to_latlon(temp_lon, temp_lat, 18, 'S')
    coordinate = [(lat, lon, temp_elev)]
    df_coords += coordinate

dfs = pd.DataFrame(
    df_coords,
    columns=['lat', 'lon', 'elev'])
# Create a Map object
map_ = st.map(dfs)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=0.5526119116880938,
        longitude=-77.43611596518961,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data = dfs,
            get_position='[lon, lat]',
            get_elevation='[elev]',
            radius=200,
            elevation_range=[0, 2000],
            pickable=True,
            get_color='[100, 30, 0, 160]',
           extruded=True,
        ),
       
    ]

)

)

st.subheader("Empty Status")
df_coords = []
for index, row in truck_copy[:10000].iterrows():
    temp_status = truck_copy.iloc[index]["STATUS"]
    temp_elev = truck_copy.iloc[index]["GPSELEVATION"] + 1220
    temp_lat = truck_copy.iloc[index]["GPSNORTHING"]
    temp_lon = truck_copy.iloc[index]["GPSEASTING"]
    if ((temp_lat > 0) & (temp_status == 'Empty')):
        lat, lon = utm.to_latlon(temp_lon, temp_lat, 18, 'S')
    coordinate = [(lat, lon, temp_elev)]
    df_coords += coordinate

dfs = pd.DataFrame(
    df_coords,
    columns=['lat', 'lon', 'elev'])
# Create a Map object
map_ = st.map(dfs)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=0.5526119116880938,
        longitude=-77.43611596518961,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data = dfs,
            get_position='[lon, lat]',
            get_elevation='[elev]',
            radius=200,
            elevation_range=[0, 2000],
            pickable=True,
            get_color='[100, 30, 0, 160]',
           extruded=True,
        ),
       
    ]

)

)

# Convert the 'GPSNORTHING' and 'GPSEASTING' columns to 'float' values
dataframe['GPSNORTHING'] = dataframe['GPSNORTHING'].astype(float)
dataframe['GPSEASTING'] = dataframe['GPSEASTING'].astype(float)

# Calculate the distance between consecutive GPS readings
dataframe['DISTANCE'] = dataframe[['GPSNORTHING', 'GPSEASTING']].shift(-1).apply(lambda x: calculate_distance(x[0], x[1], x[0], x[1]), axis=1)

# Identify the road type based on the elevation
dataframe['ROAD_TYPE'] = dataframe['GPSELEVATION'].apply(identify_road_type)

# Convert the 'TIMESTAMP' column to 'datetime' values
dataframe['TIMESTAMP'] = pd.to_datetime(dataframe['TIMESTAMP'])

# Calculate the elapsed time between consecutive GPS readings
dataframe['ELAPSED_TIME'] = dataframe['TIMESTAMP'].diff()

# Calculate the average speed for each truck by dividing the distance traveled by the elapsed time
dataframe['AVERAGE_SPEED'] = dataframe['DISTANCE'] / dataframe['ELAPSED_TIME'].dt.total_seconds()

# Drop rows with invalid data
dataframe.dropna(inplace=True)
dataframe.reset_index(drop=True, inplace=True)

# Convert the 'ROAD_TYPE' column to numerical values
dataframe['ROAD_TYPE'] = dataframe['ROAD_TYPE'].map({'flat': 0, 'gentle uphill': 1, 'steep uphill': 2, 'gentle downhill': 3, 'steep downhill': 4})

# Split the data into training and test sets
X = dataframe[["TRUCK_ID", "PAYLOAD", "SHOVEL_ID", "GPSELEVATION"]]
y = dataframe['FUEL_RATE']
# Convert the 'ELAPSED_TIME' column to seconds
dataframe['ELAPSED_TIME'] = dataframe['ELAPSED_TIME'].dt.total_seconds()

# Remove rows with missing or invalid values
dataframe = dataframe.dropna()
dataframe = dataframe[~dataframe.isin([np.inf, -np.inf]).any(1)]

model_type = st.selectbox(
                'Model Type',
                 ('Linear Regression','Random Forest Regressor'))

if model_type == 'Random Forest Regressor':
                st.info("A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.")
elif model_type == 'Linear Regression':
                st.info("LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataframe[["TRUCK_ID", "PAYLOAD", "SHOVEL_ID", "GPSELEVATION"]], dataframe['FUEL_RATE'], test_size=0.2)

@st.cache(allow_output_mutation=True, show_spinner=False, max_entries=5)
def get_pipeline():
    print(model_type)
    if model_type == 'Random Forest Regressor':
        regressor = RandomForestRegressor(n_estimators=100)
    elif model_type == 'Linear Regression':
        regressor = LinearRegression()

    pipeline = make_pipeline(
        StandardScaler(),
        regressor
    )
    pipeline.fit(X_train, y_train)
    return pipeline
with st.spinner('Fitting model...'):
    # Timer starts
    startTime = time.time()
    pipeline = get_pipeline()
        # Total time elapsed since the timer started

    totalTime = round((time.time() - startTime), 2)
    st.success('Model is ready! Time taken: ' + str(totalTime) + 's')

# Mean Absolute error of model
predictions = pipeline.predict(X_test) 
mae = mean_absolute_error(y_test, predictions)

# Calculate the r-squared value for the linear regression model
rsv = r2_score(y_test, predictions)
st.info("The mean absolute error (MAE) takes the absolute difference between the actual and forecasted values and finds the average.")
st.metric(label='Mean absolute error of model',value="{}".format(mae))
# st.info("Mean squared error of model: " + mse)

st.caption("""
<hr>
""", unsafe_allow_html=True)

st.header('Prediction')

st.info("""
    The predicted outputs shows the fuel rate prediction based on the inputs given on the left column.
""")

col1, col2 = st.columns(2)

prediction_inputs = []
default_inputs = dataframe.values[0].tolist()

# print(default_inputs)

# Append the items from the example dataset to the default inputs and outputs
all_column_names = list(dataframe.columns)
default_input_columns = all_column_names
default_output_columns = ['FUEL_RATE']

input_column_names = st.multiselect(
            'Inputs', all_column_names, default=["TRUCK_ID", "PAYLOAD", "SHOVEL_ID", "GPSELEVATION"])

output_column_names = st.multiselect(
            'Outputs', all_column_names, default=default_output_columns)
for index, column_name in enumerate(input_column_names):
                min_value = dataframe[column_name].min()
                max_value = dataframe[column_name].max()
                mean_value = dataframe[column_name].mean()
                prediction_inputs.append(col1.slider(column_name, min_value=int(min_value),
                                                     max_value=int(max_value), value=int(mean_value)))

col2.subheader('Predicted Output')

predictions = pipeline.predict([prediction_inputs])
"data", st.session_state

if len(output_column_names) > 1:
    [prediction_outputs] = predictions
else:
    prediction_outputs = predictions

for (i, item) in enumerate(prediction_outputs):
    if str(i) not in st.session_state:
        st.session_state[i] = item

i = 0
for index, predicted_value in enumerate(prediction_outputs):

    column_name = output_column_names[index]
    """ Delta is change compared to previous prediction output """
    col2.metric(label=column_name, value="{}".format(predicted_value), delta="{}".format(
        predicted_value - st.session_state[str(index)]))
# Make predictions on the test data
# predictions = model.predict(X_test)

# # Visualize the results
# st.title('Random Forest Regressor Model Performance')
# st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': predictions}))

# # Train a linear regression model
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # Evaluate the models on the test set
# lr_predictions = lr.predict(X_test)
# # Visualize the results
# st.title('Linear Regression Model Performance')
# st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': lr_predictions}))
