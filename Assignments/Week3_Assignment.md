<h1 style='text-align: center'>Week 3 Assignment, Applied Data Science Capstone</h1>
<h2 style='text-align: center'>Segmenting and Clustering Neighborhoods in Toronto</h2>

<h3> Import libraries and load API keys </h3>


```python
import requests
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import folium
from geopy import Nominatim
import matplotlib.cm as cm
import matplotlib.colors as colors

print('Libraries imported.')
```

    Libraries imported.
    


```python
import config

FSAPI = config.FourSquareAPI()
```

<h3> Data read-in from Wikipedia </h3>


```python
src_url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

response = requests.get(src_url)

df_raw = pd.read_html(response.text)[0]
df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



<h3> Data cleaning - the raw data is kept, just in case. </h3>


```python
df_proc = df_raw.copy()
```

Rename the columns appropriately


```python
df_proc.columns = ['PostalCode', 'Borough', 'Neighborhood']
```

Drop any boroughs which are not assigned:


```python
df_proc = df_proc[df_proc['Borough']!='Not assigned'].reset_index().drop('index', axis=1)
```

Rename any neighborhoods that are not assigned to their respective borough:


```python
df_proc['Neighborhood'].replace('Not assigned', df_proc['Borough'], inplace=True)
```

Group the dataframe by postal code to sum up the neighborhoods:


```python
df_proc = df_proc.groupby(by='PostalCode').sum().reset_index()
```

<h3>Take a look at the resulting dataframe:</h3>


```python
df_proc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_proc.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>103</td>
      <td>103</td>
      <td>103</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>103</td>
      <td>10</td>
      <td>99</td>
    </tr>
    <tr>
      <th>top</th>
      <td>M1H</td>
      <td>North York</td>
      <td>Downsview</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>24</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_proc.shape
```




    (103, 3)



<h3> The dataframe is now cleaned and ready for further evaluation. </h3>
<h3> The next step is assigning latitudal and longitudinal coordinates for each postal code using Geocoder. </h3>

<h3> Define a function for getting the required values. </h3>


```python
def get_latlng(postal_codes):
    '''
    Returns a list each for latitude and longitude by entering the postal codes of boroughs in Toronto, Ontario, CA.
    '''
    latitudes = []
    longitudes = []
    geolocator = Nominatim(user_agent='toronto_explorer')
    
    for code in postal_codes:
        location = None
        it = 0
        while location == None and it < 20: # For timeout requests or invalid API calls
            location = geolocator.geocode('{}, Toronto, Ontario'.format(code))
            if it >= 19:
                print('Timeout! Returning None.')
            it += 1
        if location != None:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)
    return latitudes, longitudes
```


```python
### Change to True to use the function above - not recommended, as some of the postal codes are not found on Nominatim, so this will return a lot of timeouts.
if not True:
    latitudes, longitudes = get_latlng(df_proc['PostalCode'])
    df_proc['Latitude'] = latitudes
    df_proc['Longitude'] = longitudes
else:
    latlng = pd.read_csv('Geospatial_Coordinates.csv')
    # Check if order is correct
    if (sum(latlng['Postal Code'] == df_proc['PostalCode'])) == (df_proc['PostalCode'].shape[0]):
        df_proc['Latitude'] = latlng['Latitude']
        df_proc['Longitude'] = latlng['Longitude']
df_proc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



<h3> As the latitudal and longitudal coordinates for each neighborhood are now known, it is possible to explore the neighborhoods and their venues using FourSquare API. Similar to the exercise before, let's get the top 100 venues for every postal code within a radius of 1 km. </h3>


```python
def get_venues(df_input, limit, radius):
    
    venues = []
    for code, name, lat, lng in zip(df_input['PostalCode'], df_input['Neighborhood'], df_input['Latitude'], df_input['Longitude']):
        # Define URL
        print(name)
        url_fsq = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        FSAPI.id, FSAPI.secret, FSAPI.version, lat, lng, radius, limit)
        # HTTP GET Request
        results = requests.get(url_fsq).json()['response']['groups'][0]['items']
        
        venues.append([(
            code,
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name']) for v in results])
    
    nearby_venues = pd.DataFrame([item for venue_list in venues for item in venue_list])
    nearby_venues.columns = ['PostalCode', 'Neighborhood', 'Neighborhood Latitude',
                             'Neighborhood Longitude', 'Venue Name', 'Venue Latitude',
                             'Venue Longitude', 'Venue Category']
    
    return nearby_venues
```


```python
toronto_venues = get_venues(df_proc, limit=100, radius=1000)
```

    Malvern, Rouge
    Rouge Hill, Port Union, Highland Creek
    Guildwood, Morningside, West Hill
    Woburn
    Cedarbrae
    Scarborough Village
    Kennedy Park, Ionview, East Birchmount Park
    Golden Mile, Clairlea, Oakridge
    Cliffside, Cliffcrest, Scarborough Village West
    Birch Cliff, Cliffside West
    Dorset Park, Wexford Heights, Scarborough Town Centre
    Wexford, Maryvale
    Agincourt
    Clarks Corners, Tam O'Shanter, Sullivan
    Milliken, Agincourt North, Steeles East, L'Amoreaux East
    Steeles West, L'Amoreaux West
    Upper Rouge
    Hillcrest Village
    Fairview, Henry Farm, Oriole
    Bayview Village
    York Mills, Silver Hills
    Willowdale, Newtonbrook
    Willowdale, Willowdale East
    York Mills West
    Willowdale, Willowdale West
    Parkwoods
    Don Mills
    Don Mills
    Bathurst Manor, Wilson Heights, Downsview North
    Northwood Park, York University
    Downsview
    Downsview
    Downsview
    Downsview
    Victoria Village
    Parkview Hill, Woodbine Gardens
    Woodbine Heights
    The Beaches
    Leaside
    Thorncliffe Park
    East Toronto, Broadview North (Old East York)
    The Danforth West, Riverdale
    India Bazaar, The Beaches West
    Studio District
    Lawrence Park
    Davisville North
    North Toronto West, Lawrence Park
    Davisville
    Moore Park, Summerhill East
    Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park
    Rosedale
    St. James Town, Cabbagetown
    Church and Wellesley
    Regent Park, Harbourfront
    Garden District, Ryerson
    St. James Town
    Berczy Park
    Central Bay Street
    Richmond, Adelaide, King
    Harbourfront East, Union Station, Toronto Islands
    Toronto Dominion Centre, Design Exchange
    Commerce Court, Victoria Hotel
    Bedford Park, Lawrence Manor East
    Roselawn
    Forest Hill North & West, Forest Hill Road Park
    The Annex, North Midtown, Yorkville
    University of Toronto, Harbord
    Kensington Market, Chinatown, Grange Park
    CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport
    Stn A PO Boxes
    First Canadian Place, Underground city
    Lawrence Manor, Lawrence Heights
    Glencairn
    Humewood-Cedarvale
    Caledonia-Fairbanks
    Christie
    Dufferin, Dovercourt Village
    Little Portugal, Trinity
    Brockton, Parkdale Village, Exhibition Place
    North Park, Maple Leaf Park, Upwood Park
    Del Ray, Mount Dennis, Keelsdale and Silverthorn
    Runnymede, The Junction North
    High Park, The Junction South
    Parkdale, Roncesvalles
    Runnymede, Swansea
    Queen's Park, Ontario Provincial Government
    Canada Post Gateway Processing Centre
    Business reply mail Processing Centre, South Central Letter Processing Plant Toronto
    New Toronto, Mimico South, Humber Bay Shores
    Alderwood, Long Branch
    The Kingsway, Montgomery Road, Old Mill North
    Old Mill South, King's Mill Park, Sunnylea, Humber Bay, Mimico NE, The Queensway East, Royal York South East, Kingsway Park South East
    Mimico NW, The Queensway West, South of Bloor, Kingsway Park South West, Royal York South West
    Islington Avenue, Humber Valley Village
    West Deane Park, Princess Gardens, Martin Grove, Islington, Cloverdale
    Eringate, Bloordale Gardens, Old Burnhamthorpe, Markland Wood
    Humber Summit
    Humberlea, Emery
    Weston
    Westmount
    Kingsview Village, St. Phillips, Martin Grove Gardens, Richview Gardens
    South Steeles, Silverstone, Humbergate, Jamestown, Mount Olive, Beaumond Heights, Thistletown, Albion Gardens
    Northwest, West Humber - Clairville
    

<h3> Clean and explore this dataset a bit. </h3>


```python
# Clean any venue that is labeled 'Neighborhood' to avoid confusion down the road
toronto_venues = toronto_venues[toronto_venues['Venue Category']!='Neighborhood']
toronto_venues.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue Name</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>Images Salon &amp; Spa</td>
      <td>43.802283</td>
      <td>-79.198565</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1B</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>Harvey's</td>
      <td>43.800020</td>
      <td>-79.198307</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1B</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>RBC Royal Bank</td>
      <td>43.798782</td>
      <td>-79.197090</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1B</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>Wendy’s</td>
      <td>43.807448</td>
      <td>-79.199056</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1B</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>Wendy's</td>
      <td>43.802008</td>
      <td>-79.198080</td>
      <td>Fast Food Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
toronto_venues.shape
```




    (4888, 8)



<h3> Convert the dataframe's categorical values to numeric values by using groupby, so it can later be used for clustering. </h3>


```python
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix='', prefix_sep='', columns=['Venue Category'])
toronto_onehot.insert(0, 'Neighborhood', toronto_venues['Neighborhood'])
toronto_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport</th>
      <th>Airport Lounge</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Malvern, Rouge</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 330 columns</p>
</div>



<h3> Group the dataframe by neighborhoods to explore the most common venues per neighborhood </h3>


```python
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>Airport</th>
      <th>Airport Lounge</th>
      <th>American Restaurant</th>
      <th>Amphitheater</th>
      <th>...</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agincourt</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.021277</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alderwood, Long Branch</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bayview Village</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.024390</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02439</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 330 columns</p>
</div>




```python
toronto_grouped.shape
```




    (98, 330)



<h3> Let's get the 10 most common venues for every neighborhood </h3>


```python
no_top_venues = 10
columns = ['Neighborhood']
for ind in np.arange(no_top_venues):
    if ind==0:
        columns.append('Most Common Venue')
    elif ind==1:
        columns.append('2nd Most Common Venue')
    elif ind==2:
        columns.append('3rd Most Common Venue')
    else:
        columns.append('{}th Most Common Venue'.format(ind+1))

venues_grouped_sorted = pd.DataFrame(columns=columns)
venues_grouped_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for index, row in toronto_grouped.iterrows():
    row = row.iloc[1:] # Exclude neighborhood
    row.sort_values(ascending=False, inplace=True)
    venues_grouped_sorted.iloc[index, 1:] = row.index.values[0:no_top_venues]
    
venues_grouped_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agincourt</td>
      <td>Chinese Restaurant</td>
      <td>Shopping Mall</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Bakery</td>
      <td>Malay Restaurant</td>
      <td>Lounge</td>
      <td>Motorcycle Shop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alderwood, Long Branch</td>
      <td>Discount Store</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Pharmacy</td>
      <td>Moroccan Restaurant</td>
      <td>Dance Studio</td>
      <td>Garden Center</td>
      <td>Gas Station</td>
      <td>Donut Shop</td>
      <td>Bagel Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Convenience Store</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Mediterranean Restaurant</td>
      <td>Community Center</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bayview Village</td>
      <td>Bank</td>
      <td>Gas Station</td>
      <td>Grocery Store</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Café</td>
      <td>Trail</td>
      <td>Park</td>
      <td>Skating Rink</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Juice Bar</td>
      <td>Thai Restaurant</td>
      <td>Baby Store</td>
      <td>Bagel Shop</td>
      <td>Bakery</td>
    </tr>
  </tbody>
</table>
</div>



<h3> Now we can quantify the dataframe by clustering the neighborhoods by their respective most common venues! </h3>


```python
# No. of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

kmeans = KMeans(n_clusters = kclusters, random_state=0).fit(toronto_grouped_clustering)

kmeans.labels_
```




    array([0, 0, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 1,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 1, 0,
           2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 3, 0, 1, 2, 2, 1, 2, 2, 2,
           2, 2, 1, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
           0, 0, 0, 2, 2, 0, 2, 2, 1, 4])



<h3> Add the clustering results to the grouped dataframe and visualize it using Folium. </h3>


```python
try:
    venues_grouped_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
except ValueError:
    pass

toronto_merged = df_proc.copy()
toronto_merged = toronto_merged.join(venues_grouped_sorted.set_index('Neighborhood'), on='Neighborhood')
# Some neighborhoods are listed several times due to having several postal codes - during joining
# they result in NaNs, and for the sake of the exercise, they will be dropped (as they already are
# during the groupby operation a few lines before, which is where the incompatibility stems from).

toronto_merged = toronto_merged[~toronto_merged['Cluster Labels'].isnull()]

toronto_merged.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>2.0</td>
      <td>Fast Food Restaurant</td>
      <td>Trail</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Spa</td>
      <td>Supermarket</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Caribbean Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
      <td>1.0</td>
      <td>Breakfast Spot</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Italian Restaurant</td>
      <td>Fireworks Store</td>
      <td>Falafel Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Flea Market</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
      <td>0.0</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Liquor Store</td>
      <td>Supermarket</td>
      <td>Greek Restaurant</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
      <td>2.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Mobile Phone Shop</td>
      <td>Indian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farm</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
      <td>2.0</td>
      <td>Bakery</td>
      <td>Gas Station</td>
      <td>Bank</td>
      <td>Indian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Sporting Goods Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Burger Joint</td>
    </tr>
  </tbody>
</table>
</div>




```python
toronto_merged.shape
```




    (102, 16)



<h3> Now that we have acquired and modeled all the required data, let's visualize it using Folium. </h3>


```python
geolocator = Nominatim(user_agent='toronto_explorer')
location = geolocator.geocode('Toronto, Ontario')
color_list = [colors.rgb2hex(i) for i in cm.hot(np.linspace(0, 1, kclusters))]

map_clusters = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)

for lat, lng, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster: ' + str(cluster), parse_html=True)
    folium.CircleMarker(
    [lat, lng],
    radius=5,
    color=color_list[int(cluster)],
    popup=label).add_to(map_clusters)

map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_3931d548401c49459513e589ee788873%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_3931d548401c49459513e589ee788873%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_3931d548401c49459513e589ee788873%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_3931d548401c49459513e589ee788873%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.6534817%2C%20-79.3839347%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2010%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_4e41a39fca2e4fec9781156ca08f3b4e%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_614357efd11d4b1e9d3aed5c14480a3e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.806686299999996%2C%20-79.19435340000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7608866735bd400290239cac20fc723a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_844ec4daf74242c2b596193eb5a5bf0b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_844ec4daf74242c2b596193eb5a5bf0b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMalvern%2C%20Rouge%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7608866735bd400290239cac20fc723a.setContent%28html_844ec4daf74242c2b596193eb5a5bf0b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_614357efd11d4b1e9d3aed5c14480a3e.bindPopup%28popup_7608866735bd400290239cac20fc723a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_11c55de83b2f4452925df68185a22fa7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7845351%2C%20-79.16049709999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a19d6b2f21d845659cc2ad10dacc2cb0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5a43c91207114a72aa0b8161193904f9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5a43c91207114a72aa0b8161193904f9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERouge%20Hill%2C%20Port%20Union%2C%20Highland%20Creek%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a19d6b2f21d845659cc2ad10dacc2cb0.setContent%28html_5a43c91207114a72aa0b8161193904f9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_11c55de83b2f4452925df68185a22fa7.bindPopup%28popup_a19d6b2f21d845659cc2ad10dacc2cb0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d05c58fe1e7d4cc5ae0a4ccf8be6f3aa%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7635726%2C%20-79.1887115%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2c30cb72e00f4857bde318bc02011b46%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_86f9033177c74d5e919ec00b85521c25%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_86f9033177c74d5e919ec00b85521c25%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGuildwood%2C%20Morningside%2C%20West%20Hill%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2c30cb72e00f4857bde318bc02011b46.setContent%28html_86f9033177c74d5e919ec00b85521c25%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d05c58fe1e7d4cc5ae0a4ccf8be6f3aa.bindPopup%28popup_2c30cb72e00f4857bde318bc02011b46%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f2e08bca9ae6441a813dd1c416562699%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7709921%2C%20-79.21691740000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d9f9f658e7094358b62f2ae8ef3d5234%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bd6de538cbb841ada59d9e1de9b6c244%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bd6de538cbb841ada59d9e1de9b6c244%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoburn%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d9f9f658e7094358b62f2ae8ef3d5234.setContent%28html_bd6de538cbb841ada59d9e1de9b6c244%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f2e08bca9ae6441a813dd1c416562699.bindPopup%28popup_d9f9f658e7094358b62f2ae8ef3d5234%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1800afb0da4044c8953f634a6cf3a72b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.773136%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_70e8a9429b1f4edbb9c7ca7d09029d3b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3a392382b8e4427da3a0d3fb6773a58e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3a392382b8e4427da3a0d3fb6773a58e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECedarbrae%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_70e8a9429b1f4edbb9c7ca7d09029d3b.setContent%28html_3a392382b8e4427da3a0d3fb6773a58e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1800afb0da4044c8953f634a6cf3a72b.bindPopup%28popup_70e8a9429b1f4edbb9c7ca7d09029d3b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7b00f13ac6c24091bb5831e56a40549b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7447342%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6e84df64412d42f39406c44296721be5%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_93eebe187d3b4b6cbc7ce7a03a0b22d8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_93eebe187d3b4b6cbc7ce7a03a0b22d8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EScarborough%20Village%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6e84df64412d42f39406c44296721be5.setContent%28html_93eebe187d3b4b6cbc7ce7a03a0b22d8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7b00f13ac6c24091bb5831e56a40549b.bindPopup%28popup_6e84df64412d42f39406c44296721be5%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a63fc7b04ba144a4a68be500ae64bd49%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7279292%2C%20-79.26202940000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_548812dca6064a0eb2916a664a35fcb1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_aa108d89840a40dea3bb34af58b17cbe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_aa108d89840a40dea3bb34af58b17cbe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKennedy%20Park%2C%20Ionview%2C%20East%20Birchmount%20Park%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_548812dca6064a0eb2916a664a35fcb1.setContent%28html_aa108d89840a40dea3bb34af58b17cbe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a63fc7b04ba144a4a68be500ae64bd49.bindPopup%28popup_548812dca6064a0eb2916a664a35fcb1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_baf43e10b6db4fdab8f4b29835542bda%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.711111700000004%2C%20-79.2845772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_37f6d5fdc9af48e898da787d15402da7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_de713deefacb40ad86b5d4400d6433dd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_de713deefacb40ad86b5d4400d6433dd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGolden%20Mile%2C%20Clairlea%2C%20Oakridge%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_37f6d5fdc9af48e898da787d15402da7.setContent%28html_de713deefacb40ad86b5d4400d6433dd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_baf43e10b6db4fdab8f4b29835542bda.bindPopup%28popup_37f6d5fdc9af48e898da787d15402da7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_31a83df695814461b46a8c3e44f2c0b9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.716316%2C%20-79.23947609999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5263deff3db44be2a0e34ce31f1eb06e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d5963b003c704448b24f81b483111813%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d5963b003c704448b24f81b483111813%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECliffside%2C%20Cliffcrest%2C%20Scarborough%20Village%20West%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5263deff3db44be2a0e34ce31f1eb06e.setContent%28html_d5963b003c704448b24f81b483111813%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_31a83df695814461b46a8c3e44f2c0b9.bindPopup%28popup_5263deff3db44be2a0e34ce31f1eb06e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_15fdd430c8b44c61be68b32d21f33b2e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.692657000000004%2C%20-79.2648481%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_830b359225464ed58c243b70d77eedff%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_475355070bbf40d4b6f2bfb8a89c4327%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_475355070bbf40d4b6f2bfb8a89c4327%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBirch%20Cliff%2C%20Cliffside%20West%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_830b359225464ed58c243b70d77eedff.setContent%28html_475355070bbf40d4b6f2bfb8a89c4327%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_15fdd430c8b44c61be68b32d21f33b2e.bindPopup%28popup_830b359225464ed58c243b70d77eedff%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a0c5b8ecbba542d2a10ebe9776f8d1b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7574096%2C%20-79.27330400000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c5dd783f72c34f138e5d0efc069c40fd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9448ddc1a0c54c2c83f6c56520fb5841%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9448ddc1a0c54c2c83f6c56520fb5841%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDorset%20Park%2C%20Wexford%20Heights%2C%20Scarborough%20Town%20Centre%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c5dd783f72c34f138e5d0efc069c40fd.setContent%28html_9448ddc1a0c54c2c83f6c56520fb5841%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a0c5b8ecbba542d2a10ebe9776f8d1b2.bindPopup%28popup_c5dd783f72c34f138e5d0efc069c40fd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_71ac63859272440ca7188d39f11b8ee0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.750071500000004%2C%20-79.2958491%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_57b9e41e5331483e8573d2464c93aa7b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7679efe363024054aca4a6e06654b55b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7679efe363024054aca4a6e06654b55b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWexford%2C%20Maryvale%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_57b9e41e5331483e8573d2464c93aa7b.setContent%28html_7679efe363024054aca4a6e06654b55b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_71ac63859272440ca7188d39f11b8ee0.bindPopup%28popup_57b9e41e5331483e8573d2464c93aa7b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bc97eafdfa504c268fcdd5ec7c915079%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7942003%2C%20-79.26202940000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2ba482a12218470084769ff44d633acf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b3a97a744f574f259935cc0ea203b32e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b3a97a744f574f259935cc0ea203b32e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAgincourt%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2ba482a12218470084769ff44d633acf.setContent%28html_b3a97a744f574f259935cc0ea203b32e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bc97eafdfa504c268fcdd5ec7c915079.bindPopup%28popup_2ba482a12218470084769ff44d633acf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_64b3c91418f6474ca88d9d98f7bf908b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7816375%2C%20-79.3043021%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0cb52361965b4106958b0ff9c6ed89f0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b65dfb2d5c444e58a7c63e2ba6a0f424%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b65dfb2d5c444e58a7c63e2ba6a0f424%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EClarks%20Corners%2C%20Tam%20O%26%2339%3BShanter%2C%20Sullivan%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0cb52361965b4106958b0ff9c6ed89f0.setContent%28html_b65dfb2d5c444e58a7c63e2ba6a0f424%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_64b3c91418f6474ca88d9d98f7bf908b.bindPopup%28popup_0cb52361965b4106958b0ff9c6ed89f0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f261132f1a354ccaa2dca5532c9bf567%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.8152522%2C%20-79.2845772%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6b0e4638ab164119b498bfe901005f3b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5c46dcc54efe42d5af7108b9f8f36738%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5c46dcc54efe42d5af7108b9f8f36738%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMilliken%2C%20Agincourt%20North%2C%20Steeles%20East%2C%20L%26%2339%3BAmoreaux%20East%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6b0e4638ab164119b498bfe901005f3b.setContent%28html_5c46dcc54efe42d5af7108b9f8f36738%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f261132f1a354ccaa2dca5532c9bf567.bindPopup%28popup_6b0e4638ab164119b498bfe901005f3b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0789c28a7d374d848c8464c797d74a47%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.799525200000005%2C%20-79.3183887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_db93d5ab474248609ac10d05f7c034d2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3b69295fc6484b28a1ecffe4f702d678%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3b69295fc6484b28a1ecffe4f702d678%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESteeles%20West%2C%20L%26%2339%3BAmoreaux%20West%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_db93d5ab474248609ac10d05f7c034d2.setContent%28html_3b69295fc6484b28a1ecffe4f702d678%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0789c28a7d374d848c8464c797d74a47.bindPopup%28popup_db93d5ab474248609ac10d05f7c034d2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9c31bde24db249ee9ffe3455df98e4fb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.8037622%2C%20-79.3634517%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_61342d7dafa44b349b979b0cf3524fc3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_be60216217764599808f6a46b0fd30c8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_be60216217764599808f6a46b0fd30c8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHillcrest%20Village%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_61342d7dafa44b349b979b0cf3524fc3.setContent%28html_be60216217764599808f6a46b0fd30c8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9c31bde24db249ee9ffe3455df98e4fb.bindPopup%28popup_61342d7dafa44b349b979b0cf3524fc3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b033cff8c1404e89bf79ba58211d2266%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7785175%2C%20-79.3465557%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d9fc2fe288fd4949b767eb63316028c7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_20dee5abcbbd47be91928b7308c64869%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_20dee5abcbbd47be91928b7308c64869%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFairview%2C%20Henry%20Farm%2C%20Oriole%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d9fc2fe288fd4949b767eb63316028c7.setContent%28html_20dee5abcbbd47be91928b7308c64869%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b033cff8c1404e89bf79ba58211d2266.bindPopup%28popup_d9fc2fe288fd4949b767eb63316028c7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_32ddc99926d14c1a83a2d894c52b0c9a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7869473%2C%20-79.385975%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c1d2ce8056cc4cdaaa81e8ddae12abb6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4efdbf6c45b941e7b25c3b8631728b55%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4efdbf6c45b941e7b25c3b8631728b55%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBayview%20Village%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c1d2ce8056cc4cdaaa81e8ddae12abb6.setContent%28html_4efdbf6c45b941e7b25c3b8631728b55%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_32ddc99926d14c1a83a2d894c52b0c9a.bindPopup%28popup_c1d2ce8056cc4cdaaa81e8ddae12abb6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_661adcbffc2848f4bf3f9addf1e5dd37%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7574902%2C%20-79.37471409999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffffff%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ffffff%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cd082fac8bdc42d9a826ec1817bbd965%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3c6117b8c3324bd48555382dd35456b6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3c6117b8c3324bd48555382dd35456b6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYork%20Mills%2C%20Silver%20Hills%20Cluster%3A%204.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cd082fac8bdc42d9a826ec1817bbd965.setContent%28html_3c6117b8c3324bd48555382dd35456b6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_661adcbffc2848f4bf3f9addf1e5dd37.bindPopup%28popup_cd082fac8bdc42d9a826ec1817bbd965%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b344f8f9be46498b8ff97e1936a3ca73%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.789053%2C%20-79.40849279999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_79d36dd1f84e44329b875e96ed5872b0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4ab03ff699154615af31c5691942c7d1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4ab03ff699154615af31c5691942c7d1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Newtonbrook%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_79d36dd1f84e44329b875e96ed5872b0.setContent%28html_4ab03ff699154615af31c5691942c7d1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b344f8f9be46498b8ff97e1936a3ca73.bindPopup%28popup_79d36dd1f84e44329b875e96ed5872b0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dace8def242e4b868350ef2153ee6d25%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7701199%2C%20-79.40849279999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_03a7e8841f414212816ce9e291f8ca54%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_943354ab78cd43cfb50d3850515a1ca6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_943354ab78cd43cfb50d3850515a1ca6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Willowdale%20East%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_03a7e8841f414212816ce9e291f8ca54.setContent%28html_943354ab78cd43cfb50d3850515a1ca6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_dace8def242e4b868350ef2153ee6d25.bindPopup%28popup_03a7e8841f414212816ce9e291f8ca54%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fa25724add364bfd92fafc6f4323f435%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.752758299999996%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1a6f6250011d41c9a9e693e8b5233177%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ffbee5db185048b899b30933a599ae7c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ffbee5db185048b899b30933a599ae7c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EYork%20Mills%20West%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1a6f6250011d41c9a9e693e8b5233177.setContent%28html_ffbee5db185048b899b30933a599ae7c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fa25724add364bfd92fafc6f4323f435.bindPopup%28popup_1a6f6250011d41c9a9e693e8b5233177%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_29bcf7af279847a7a480e62662d95969%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7827364%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f5021858be314fffbdbdca2ad4faeee9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7399012742c34443a400b3f646d3dc44%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7399012742c34443a400b3f646d3dc44%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWillowdale%2C%20Willowdale%20West%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f5021858be314fffbdbdca2ad4faeee9.setContent%28html_7399012742c34443a400b3f646d3dc44%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_29bcf7af279847a7a480e62662d95969.bindPopup%28popup_f5021858be314fffbdbdca2ad4faeee9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_402f327fbc064d0d979c7555c35da347%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7532586%2C%20-79.3296565%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4526d1537a79488db84fa2281467b946%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_96043e3a086748cbad42329b1dfb394f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_96043e3a086748cbad42329b1dfb394f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkwoods%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4526d1537a79488db84fa2281467b946.setContent%28html_96043e3a086748cbad42329b1dfb394f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_402f327fbc064d0d979c7555c35da347.bindPopup%28popup_4526d1537a79488db84fa2281467b946%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_93812a3577cf49c0bdc9e215ad47e173%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.745905799999996%2C%20-79.352188%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ddb07ffef0fb4a1d98198c0c117a2303%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8777773d25ac4ecf8cae4963519ab953%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8777773d25ac4ecf8cae4963519ab953%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDon%20Mills%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ddb07ffef0fb4a1d98198c0c117a2303.setContent%28html_8777773d25ac4ecf8cae4963519ab953%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_93812a3577cf49c0bdc9e215ad47e173.bindPopup%28popup_ddb07ffef0fb4a1d98198c0c117a2303%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2c5181eb970f478fab303afddd63d55a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.72589970000001%2C%20-79.340923%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_576d1f85d6cf4d2ca92f7ebd5097cdf9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4e6bcf69d4ce4e5893ccea4cdef7cd0c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4e6bcf69d4ce4e5893ccea4cdef7cd0c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDon%20Mills%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_576d1f85d6cf4d2ca92f7ebd5097cdf9.setContent%28html_4e6bcf69d4ce4e5893ccea4cdef7cd0c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2c5181eb970f478fab303afddd63d55a.bindPopup%28popup_576d1f85d6cf4d2ca92f7ebd5097cdf9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2a8439a1c915439ab9e3efab24baa363%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7543283%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c9f520ee47c141798bf4896ed5d6ae12%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_93372e2fbff94a33a3ace73b56687804%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_93372e2fbff94a33a3ace73b56687804%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBathurst%20Manor%2C%20Wilson%20Heights%2C%20Downsview%20North%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c9f520ee47c141798bf4896ed5d6ae12.setContent%28html_93372e2fbff94a33a3ace73b56687804%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2a8439a1c915439ab9e3efab24baa363.bindPopup%28popup_c9f520ee47c141798bf4896ed5d6ae12%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3b9af7fd997c403c9d619e5ef3cb5005%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7679803%2C%20-79.48726190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5a7736d2aeb445b38b0338f0d48fc205%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8237897ce1ed4de0b3604f64d5698a80%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8237897ce1ed4de0b3604f64d5698a80%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorthwood%20Park%2C%20York%20University%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5a7736d2aeb445b38b0338f0d48fc205.setContent%28html_8237897ce1ed4de0b3604f64d5698a80%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3b9af7fd997c403c9d619e5ef3cb5005.bindPopup%28popup_5a7736d2aeb445b38b0338f0d48fc205%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6653b74471b946c49300b7f8491c978d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.737473200000004%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f0a7b8328b024b6d9e88b26e6d959f22%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1e80c3e960ab41019a10fa543ca1522b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1e80c3e960ab41019a10fa543ca1522b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f0a7b8328b024b6d9e88b26e6d959f22.setContent%28html_1e80c3e960ab41019a10fa543ca1522b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6653b74471b946c49300b7f8491c978d.bindPopup%28popup_f0a7b8328b024b6d9e88b26e6d959f22%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_26aa5a64c5524925a02dc5c408adbfdd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7390146%2C%20-79.5069436%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0d128d1686a947649972207ee81a1ad2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e4ce81a4354a463ba6c3149d2bf1a14f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e4ce81a4354a463ba6c3149d2bf1a14f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0d128d1686a947649972207ee81a1ad2.setContent%28html_e4ce81a4354a463ba6c3149d2bf1a14f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_26aa5a64c5524925a02dc5c408adbfdd.bindPopup%28popup_0d128d1686a947649972207ee81a1ad2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e99a345c2a2c44e7bb5311d1653af0b3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7284964%2C%20-79.49569740000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a3098451ef424ea9a57ede15dfaa39d4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2ccf040aa10f4505b0d5021abbf56e5a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2ccf040aa10f4505b0d5021abbf56e5a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a3098451ef424ea9a57ede15dfaa39d4.setContent%28html_2ccf040aa10f4505b0d5021abbf56e5a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e99a345c2a2c44e7bb5311d1653af0b3.bindPopup%28popup_a3098451ef424ea9a57ede15dfaa39d4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_18db1f7a94e945d69d37d12b602cdd02%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7616313%2C%20-79.52099940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_818e019990e442e69fce014847e3b917%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_99705a4e2f424cad9e464bb762544d8b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_99705a4e2f424cad9e464bb762544d8b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDownsview%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_818e019990e442e69fce014847e3b917.setContent%28html_99705a4e2f424cad9e464bb762544d8b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_18db1f7a94e945d69d37d12b602cdd02.bindPopup%28popup_818e019990e442e69fce014847e3b917%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_11c78fc70fcb4eb0bc7c38069e15a149%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.725882299999995%2C%20-79.31557159999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_96dd2c342c44403ebfe1888b4ea32a8a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4923c6b9ce7244b085654787e31264c5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4923c6b9ce7244b085654787e31264c5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EVictoria%20Village%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_96dd2c342c44403ebfe1888b4ea32a8a.setContent%28html_4923c6b9ce7244b085654787e31264c5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_11c78fc70fcb4eb0bc7c38069e15a149.bindPopup%28popup_96dd2c342c44403ebfe1888b4ea32a8a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7aa26b9e2df437b980dd92db1fa7b01%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7063972%2C%20-79.309937%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9b173b3e457c4ecca9e3302e0ada54e9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e20db789208e431f8dfa26ccd9e933b1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e20db789208e431f8dfa26ccd9e933b1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkview%20Hill%2C%20Woodbine%20Gardens%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9b173b3e457c4ecca9e3302e0ada54e9.setContent%28html_e20db789208e431f8dfa26ccd9e933b1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a7aa26b9e2df437b980dd92db1fa7b01.bindPopup%28popup_9b173b3e457c4ecca9e3302e0ada54e9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c5a4b7e4e7804e1f8aab295475d8163d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.695343900000005%2C%20-79.3183887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1330c99eed52402eb54222237d587456%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_21a41a26bfdd4e7dae2dbd03d8a4c8da%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_21a41a26bfdd4e7dae2dbd03d8a4c8da%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWoodbine%20Heights%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1330c99eed52402eb54222237d587456.setContent%28html_21a41a26bfdd4e7dae2dbd03d8a4c8da%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c5a4b7e4e7804e1f8aab295475d8163d.bindPopup%28popup_1330c99eed52402eb54222237d587456%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_76ec6cb66d06415399304eddcf1ae73e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.67635739999999%2C%20-79.2930312%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_551dc6ec028740b7bb28f1246e9e2f91%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_595661cebe184709ba6979296339609a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_595661cebe184709ba6979296339609a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Beaches%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_551dc6ec028740b7bb28f1246e9e2f91.setContent%28html_595661cebe184709ba6979296339609a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_76ec6cb66d06415399304eddcf1ae73e.bindPopup%28popup_551dc6ec028740b7bb28f1246e9e2f91%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_0f153832fe7d416687174282e7f266f9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7090604%2C%20-79.3634517%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a29aa1f3e32a4d6fbf0831cc12fde12d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_690bba13bd304ab089db24838aff8251%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_690bba13bd304ab089db24838aff8251%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELeaside%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a29aa1f3e32a4d6fbf0831cc12fde12d.setContent%28html_690bba13bd304ab089db24838aff8251%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_0f153832fe7d416687174282e7f266f9.bindPopup%28popup_a29aa1f3e32a4d6fbf0831cc12fde12d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6d1738d79c3f4555b9fcdfe22b18fe05%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7053689%2C%20-79.34937190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5acf581fc3764e05aeb6ee1202057a25%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_50399c374f164d40a1899004b657f8f1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_50399c374f164d40a1899004b657f8f1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThorncliffe%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5acf581fc3764e05aeb6ee1202057a25.setContent%28html_50399c374f164d40a1899004b657f8f1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6d1738d79c3f4555b9fcdfe22b18fe05.bindPopup%28popup_5acf581fc3764e05aeb6ee1202057a25%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a22a9318f473411bb416e235c4d80f26%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.685347%2C%20-79.3381065%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7725e13b66964ea0b5d1be39e40cdf76%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ff970ecf463c438882d94d730f631d1d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ff970ecf463c438882d94d730f631d1d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEast%20Toronto%2C%20Broadview%20North%20%28Old%20East%20York%29%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7725e13b66964ea0b5d1be39e40cdf76.setContent%28html_ff970ecf463c438882d94d730f631d1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a22a9318f473411bb416e235c4d80f26.bindPopup%28popup_7725e13b66964ea0b5d1be39e40cdf76%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_99c01c7bfc5648a5adc6929200ea5deb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795571%2C%20-79.352188%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d4e9f07fb33f4451b3579c06301284b8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_79dcd558cf994e57b387a9689f8a15c2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_79dcd558cf994e57b387a9689f8a15c2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Danforth%20West%2C%20Riverdale%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d4e9f07fb33f4451b3579c06301284b8.setContent%28html_79dcd558cf994e57b387a9689f8a15c2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_99c01c7bfc5648a5adc6929200ea5deb.bindPopup%28popup_d4e9f07fb33f4451b3579c06301284b8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5d9248b2095646c99426d0d46dcf0825%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6689985%2C%20-79.31557159999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c38c8d60761245f6ade21e9014728c08%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_92023e37e80b402ab193fdab68de96e6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_92023e37e80b402ab193fdab68de96e6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIndia%20Bazaar%2C%20The%20Beaches%20West%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c38c8d60761245f6ade21e9014728c08.setContent%28html_92023e37e80b402ab193fdab68de96e6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5d9248b2095646c99426d0d46dcf0825.bindPopup%28popup_c38c8d60761245f6ade21e9014728c08%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1725c54f4f9a48d0ae25ac0386b5b196%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6595255%2C%20-79.340923%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_41684aa9dde64590bf3ed4c6c4d98c60%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8493af6e119944a49f9a81ca09a38cf2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8493af6e119944a49f9a81ca09a38cf2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStudio%20District%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_41684aa9dde64590bf3ed4c6c4d98c60.setContent%28html_8493af6e119944a49f9a81ca09a38cf2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1725c54f4f9a48d0ae25ac0386b5b196.bindPopup%28popup_41684aa9dde64590bf3ed4c6c4d98c60%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a6985bb9f6284d179470c07a5e0f167e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7280205%2C%20-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_83f1e179fc224a58b3545f3a3a785eaa%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_cd5d9f4ca1cf4b84b359a591194240dc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_cd5d9f4ca1cf4b84b359a591194240dc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_83f1e179fc224a58b3545f3a3a785eaa.setContent%28html_cd5d9f4ca1cf4b84b359a591194240dc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a6985bb9f6284d179470c07a5e0f167e.bindPopup%28popup_83f1e179fc224a58b3545f3a3a785eaa%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9305b5dea43d43168dcb1f367e8c4249%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7127511%2C%20-79.3901975%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_85974541f1374c01b549795818e605dc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4ace7a245bb6496081791b1c383714b6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4ace7a245bb6496081791b1c383714b6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20North%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_85974541f1374c01b549795818e605dc.setContent%28html_4ace7a245bb6496081791b1c383714b6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9305b5dea43d43168dcb1f367e8c4249.bindPopup%28popup_85974541f1374c01b549795818e605dc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_96177a44a4a547ebb8d9cd2c31741be9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7153834%2C%20-79.40567840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bc04c3185004455c832349b1a295dd62%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2db825aa9e0c469ebd340e698d211824%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2db825aa9e0c469ebd340e698d211824%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Toronto%20West%2C%20Lawrence%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bc04c3185004455c832349b1a295dd62.setContent%28html_2db825aa9e0c469ebd340e698d211824%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_96177a44a4a547ebb8d9cd2c31741be9.bindPopup%28popup_bc04c3185004455c832349b1a295dd62%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_632420e6c0484d55b381498d9c7acf80%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7043244%2C%20-79.3887901%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_21bf308250d24c459d9871898d983565%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e8937e1f2df246ad8b3ba725677f7a4f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e8937e1f2df246ad8b3ba725677f7a4f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_21bf308250d24c459d9871898d983565.setContent%28html_e8937e1f2df246ad8b3ba725677f7a4f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_632420e6c0484d55b381498d9c7acf80.bindPopup%28popup_21bf308250d24c459d9871898d983565%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e41c6cc7f2a54546a63467a4b4977fa5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6895743%2C%20-79.38315990000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cfffc4083a5343ac9098733da370817c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_42eac268bf7347dcaf3a89c9e1c40966%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_42eac268bf7347dcaf3a89c9e1c40966%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoore%20Park%2C%20Summerhill%20East%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cfffc4083a5343ac9098733da370817c.setContent%28html_42eac268bf7347dcaf3a89c9e1c40966%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e41c6cc7f2a54546a63467a4b4977fa5.bindPopup%28popup_cfffc4083a5343ac9098733da370817c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4a4538e027cd4a2ba00d3c747b139962%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.68641229999999%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_80c3a3ea0bf042da8bccb5497269eb62%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7401a02dfe1f4999b55833ad6eaf2910%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7401a02dfe1f4999b55833ad6eaf2910%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESummerhill%20West%2C%20Rathnelly%2C%20South%20Hill%2C%20Forest%20Hill%20SE%2C%20Deer%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_80c3a3ea0bf042da8bccb5497269eb62.setContent%28html_7401a02dfe1f4999b55833ad6eaf2910%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4a4538e027cd4a2ba00d3c747b139962.bindPopup%28popup_80c3a3ea0bf042da8bccb5497269eb62%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_80a315cf83504d0e8c04d340ed103822%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6795626%2C%20-79.37752940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d9500b1c171c48e39ca472c341d6e9e3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a8c2d86da5724a12b52181fa8aed88f4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a8c2d86da5724a12b52181fa8aed88f4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d9500b1c171c48e39ca472c341d6e9e3.setContent%28html_a8c2d86da5724a12b52181fa8aed88f4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_80a315cf83504d0e8c04d340ed103822.bindPopup%28popup_d9500b1c171c48e39ca472c341d6e9e3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9aa18a870e064bc1a553ba99040617ec%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.667967%2C%20-79.3676753%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_71333f2a11494538b6453950a2ba1898%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_272cfaee894d49c89f36e91dc6d373f6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_272cfaee894d49c89f36e91dc6d373f6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Cabbagetown%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_71333f2a11494538b6453950a2ba1898.setContent%28html_272cfaee894d49c89f36e91dc6d373f6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9aa18a870e064bc1a553ba99040617ec.bindPopup%28popup_71333f2a11494538b6453950a2ba1898%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6c41061418b446d0b059575e96b37366%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6658599%2C%20-79.38315990000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1e6f8293c6534e7ca6b877701430833a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_04fcc1accb0643ce8b95dd00f5ae8f77%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_04fcc1accb0643ce8b95dd00f5ae8f77%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChurch%20and%20Wellesley%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1e6f8293c6534e7ca6b877701430833a.setContent%28html_04fcc1accb0643ce8b95dd00f5ae8f77%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6c41061418b446d0b059575e96b37366.bindPopup%28popup_1e6f8293c6534e7ca6b877701430833a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3449291723cd44ed9a37ae58a4df4f05%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6542599%2C%20-79.3606359%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8f1c94698cbb40aab40d64cbe950f7fd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_583e5c5cd564490b882f19fe7145baac%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_583e5c5cd564490b882f19fe7145baac%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERegent%20Park%2C%20Harbourfront%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8f1c94698cbb40aab40d64cbe950f7fd.setContent%28html_583e5c5cd564490b882f19fe7145baac%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3449291723cd44ed9a37ae58a4df4f05.bindPopup%28popup_8f1c94698cbb40aab40d64cbe950f7fd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_75a9186daa244d15991896cec797f832%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6571618%2C%20-79.37893709999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4adb59411c7c495490abd159df68cdd0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_82243199596949eb863757ef456b9c51%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_82243199596949eb863757ef456b9c51%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGarden%20District%2C%20Ryerson%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4adb59411c7c495490abd159df68cdd0.setContent%28html_82243199596949eb863757ef456b9c51%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_75a9186daa244d15991896cec797f832.bindPopup%28popup_4adb59411c7c495490abd159df68cdd0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5a89323e53e94a9aa509c060cd41d543%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6514939%2C%20-79.3754179%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_853125ccf45b43d4851b951e9e0ef10b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fda60c3c87f947e4befd1a2613c04059%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fda60c3c87f947e4befd1a2613c04059%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_853125ccf45b43d4851b951e9e0ef10b.setContent%28html_fda60c3c87f947e4befd1a2613c04059%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5a89323e53e94a9aa509c060cd41d543.bindPopup%28popup_853125ccf45b43d4851b951e9e0ef10b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d89002ed53c449bb81feb3736a3e6599%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.644770799999996%2C%20-79.3733064%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b629e6926d484a2a93e613f3cfc33327%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f6f5962994c84de2aecebaa98c7425da%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f6f5962994c84de2aecebaa98c7425da%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBerczy%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b629e6926d484a2a93e613f3cfc33327.setContent%28html_f6f5962994c84de2aecebaa98c7425da%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d89002ed53c449bb81feb3736a3e6599.bindPopup%28popup_b629e6926d484a2a93e613f3cfc33327%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6c03a239af0b48d0ab94973abfc931d3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6579524%2C%20-79.3873826%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6985fa397e844476b3e6391cf82eea16%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_866520eabb4142a6836a6fea9dea6c9a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_866520eabb4142a6836a6fea9dea6c9a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Bay%20Street%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6985fa397e844476b3e6391cf82eea16.setContent%28html_866520eabb4142a6836a6fea9dea6c9a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6c03a239af0b48d0ab94973abfc931d3.bindPopup%28popup_6985fa397e844476b3e6391cf82eea16%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5d815fa224564d82a5d2ffde99c8031b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.65057120000001%2C%20-79.3845675%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5bfa4c62965b4e04b6398df1eb981592%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8a894c7fbe714ec7a336dd7ad039b443%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8a894c7fbe714ec7a336dd7ad039b443%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%2C%20Adelaide%2C%20King%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5bfa4c62965b4e04b6398df1eb981592.setContent%28html_8a894c7fbe714ec7a336dd7ad039b443%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5d815fa224564d82a5d2ffde99c8031b.bindPopup%28popup_5bfa4c62965b4e04b6398df1eb981592%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e41ba4e5c8ad4d5d8f7593c1fa638628%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6408157%2C%20-79.38175229999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_98e7a3ea90b040d08da4d3bc3ab090f0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f16fe0548fe94aab94dbb6bc0dff2282%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f16fe0548fe94aab94dbb6bc0dff2282%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHarbourfront%20East%2C%20Union%20Station%2C%20Toronto%20Islands%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_98e7a3ea90b040d08da4d3bc3ab090f0.setContent%28html_f16fe0548fe94aab94dbb6bc0dff2282%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e41ba4e5c8ad4d5d8f7593c1fa638628.bindPopup%28popup_98e7a3ea90b040d08da4d3bc3ab090f0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2d97997fb3084aab9b75895925a5dad2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6471768%2C%20-79.38157640000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d54e901d528247e7b3619a4bd22c31cf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4343644de7944c1397d44fa56f017f04%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4343644de7944c1397d44fa56f017f04%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%20Dominion%20Centre%2C%20Design%20Exchange%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d54e901d528247e7b3619a4bd22c31cf.setContent%28html_4343644de7944c1397d44fa56f017f04%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2d97997fb3084aab9b75895925a5dad2.bindPopup%28popup_d54e901d528247e7b3619a4bd22c31cf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bab56f00668e4869ae6ca8ec0f9309a4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6481985%2C%20-79.37981690000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_045cffb665084e0c842d54134d896746%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e4233fa2d2a9412782de64d7b7948dc9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e4233fa2d2a9412782de64d7b7948dc9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECommerce%20Court%2C%20Victoria%20Hotel%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_045cffb665084e0c842d54134d896746.setContent%28html_e4233fa2d2a9412782de64d7b7948dc9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bab56f00668e4869ae6ca8ec0f9309a4.bindPopup%28popup_045cffb665084e0c842d54134d896746%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_178101acf7e740ef84e81873ea028d2a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7332825%2C%20-79.4197497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9e6330103c754ba182bd99da627522d2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_71c4cfbb5a4b492c8518e536fd33435c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_71c4cfbb5a4b492c8518e536fd33435c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBedford%20Park%2C%20Lawrence%20Manor%20East%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9e6330103c754ba182bd99da627522d2.setContent%28html_71c4cfbb5a4b492c8518e536fd33435c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_178101acf7e740ef84e81873ea028d2a.bindPopup%28popup_9e6330103c754ba182bd99da627522d2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_648dce3ed40442dd88e8cd684297fe8b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7116948%2C%20-79.41693559999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8fbe944afd53472c9176415781c64867%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fe8d1848db7d4464b70bb44d6650ca0b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fe8d1848db7d4464b70bb44d6650ca0b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERoselawn%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8fbe944afd53472c9176415781c64867.setContent%28html_fe8d1848db7d4464b70bb44d6650ca0b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_648dce3ed40442dd88e8cd684297fe8b.bindPopup%28popup_8fbe944afd53472c9176415781c64867%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a37c40a738404356b1d1ead897804f85%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6969476%2C%20-79.41130720000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a2ce59984ef2492db957a22f8f47bb57%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e6a68c0c3b6e42c2917fba2f76f77bd7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e6a68c0c3b6e42c2917fba2f76f77bd7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hill%20North%20%26amp%3B%20West%2C%20Forest%20Hill%20Road%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a2ce59984ef2492db957a22f8f47bb57.setContent%28html_e6a68c0c3b6e42c2917fba2f76f77bd7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a37c40a738404356b1d1ead897804f85.bindPopup%28popup_a2ce59984ef2492db957a22f8f47bb57%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_33673999dd4f42269ab9b0d6e7daff32%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6727097%2C%20-79.40567840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_300068604b08409d9c56014cf2e0167e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bfebbea77105447e80e010fea271e8aa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bfebbea77105447e80e010fea271e8aa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Annex%2C%20North%20Midtown%2C%20Yorkville%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_300068604b08409d9c56014cf2e0167e.setContent%28html_bfebbea77105447e80e010fea271e8aa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_33673999dd4f42269ab9b0d6e7daff32.bindPopup%28popup_300068604b08409d9c56014cf2e0167e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_957461dfdaea4046b3ec95582012b238%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6626956%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_871ee18e85ec4fd9a335c235d16f531f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_58acef99e29549fa91bc712a896e47e3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_58acef99e29549fa91bc712a896e47e3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20of%20Toronto%2C%20Harbord%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_871ee18e85ec4fd9a335c235d16f531f.setContent%28html_58acef99e29549fa91bc712a896e47e3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_957461dfdaea4046b3ec95582012b238.bindPopup%28popup_871ee18e85ec4fd9a335c235d16f531f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4ff6f258e3744b8fb933970ef719793b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6532057%2C%20-79.4000493%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7959445faf2242c5a3f9720fc9240ee2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_504b4992668b41fd8a67c2474068a758%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_504b4992668b41fd8a67c2474068a758%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington%20Market%2C%20Chinatown%2C%20Grange%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7959445faf2242c5a3f9720fc9240ee2.setContent%28html_504b4992668b41fd8a67c2474068a758%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4ff6f258e3744b8fb933970ef719793b.bindPopup%28popup_7959445faf2242c5a3f9720fc9240ee2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7b5bd1c42c1e463eab2455e9a0d49fa7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6289467%2C%20-79.3944199%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f5d7a3e810ad41eb9eb26e759b62cf72%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_28b57ae6a9b24beda43e47e415d275ab%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_28b57ae6a9b24beda43e47e415d275ab%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECN%20Tower%2C%20King%20and%20Spadina%2C%20Railway%20Lands%2C%20Harbourfront%20West%2C%20Bathurst%20Quay%2C%20South%20Niagara%2C%20Island%20airport%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f5d7a3e810ad41eb9eb26e759b62cf72.setContent%28html_28b57ae6a9b24beda43e47e415d275ab%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7b5bd1c42c1e463eab2455e9a0d49fa7.bindPopup%28popup_f5d7a3e810ad41eb9eb26e759b62cf72%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_eec4d163b083455aad4494bc6a04f4bd%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6464352%2C%20-79.37484599999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b1aa0dad0eb341289fb17f2618c5e699%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_08f9308ec6df474aa31be60d2f6fc6fe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_08f9308ec6df474aa31be60d2f6fc6fe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStn%20A%20PO%20Boxes%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b1aa0dad0eb341289fb17f2618c5e699.setContent%28html_08f9308ec6df474aa31be60d2f6fc6fe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_eec4d163b083455aad4494bc6a04f4bd.bindPopup%28popup_b1aa0dad0eb341289fb17f2618c5e699%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7c7428de15884040bdc8b15b45b4f641%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6484292%2C%20-79.3822802%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_37ff6de79e264c47803d7683ac0aaa2b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ffdc8461ac264f2a8973ec5565177f04%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ffdc8461ac264f2a8973ec5565177f04%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFirst%20Canadian%20Place%2C%20Underground%20city%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_37ff6de79e264c47803d7683ac0aaa2b.setContent%28html_ffdc8461ac264f2a8973ec5565177f04%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7c7428de15884040bdc8b15b45b4f641.bindPopup%28popup_37ff6de79e264c47803d7683ac0aaa2b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e383f09a09ff4e4fb90792265b995777%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.718517999999996%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6b144f557ee24fc4bcd6cb6c13f60b4c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a3a85ce5de3f4e57928b09d656809896%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a3a85ce5de3f4e57928b09d656809896%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Manor%2C%20Lawrence%20Heights%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6b144f557ee24fc4bcd6cb6c13f60b4c.setContent%28html_a3a85ce5de3f4e57928b09d656809896%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e383f09a09ff4e4fb90792265b995777.bindPopup%28popup_6b144f557ee24fc4bcd6cb6c13f60b4c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_13b86d96ebd6407d8f5d1b72fb90c460%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.709577%2C%20-79.44507259999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c9c5660c19a34e0abec655f0b22f3580%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_2ca0a98ecf0745cd9b03a96b7c00462d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_2ca0a98ecf0745cd9b03a96b7c00462d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGlencairn%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c9c5660c19a34e0abec655f0b22f3580.setContent%28html_2ca0a98ecf0745cd9b03a96b7c00462d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_13b86d96ebd6407d8f5d1b72fb90c460.bindPopup%28popup_c9c5660c19a34e0abec655f0b22f3580%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c08f12e00fd64cc8b10de89d184161d5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6937813%2C%20-79.42819140000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_deb53f1457cd491fba47aea3b4bd4df4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_86452726e2e842398f3267922a4b3a53%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_86452726e2e842398f3267922a4b3a53%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumewood-Cedarvale%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_deb53f1457cd491fba47aea3b4bd4df4.setContent%28html_86452726e2e842398f3267922a4b3a53%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c08f12e00fd64cc8b10de89d184161d5.bindPopup%28popup_deb53f1457cd491fba47aea3b4bd4df4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1ac47b3a826e4b58affc67891677fa24%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6890256%2C%20-79.453512%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_76aea145f2424881a3582a50deca27f9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fc9c9c1faf904d3ebec24667d1f3a183%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fc9c9c1faf904d3ebec24667d1f3a183%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECaledonia-Fairbanks%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_76aea145f2424881a3582a50deca27f9.setContent%28html_fc9c9c1faf904d3ebec24667d1f3a183%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1ac47b3a826e4b58affc67891677fa24.bindPopup%28popup_76aea145f2424881a3582a50deca27f9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_351941e263ca49db8cb39c943ee84f10%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.669542%2C%20-79.4225637%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_29819dfa63684153ad54a137cebac462%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_49a2dbfe47124848afe4286aaa2f0d6f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_49a2dbfe47124848afe4286aaa2f0d6f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChristie%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_29819dfa63684153ad54a137cebac462.setContent%28html_49a2dbfe47124848afe4286aaa2f0d6f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_351941e263ca49db8cb39c943ee84f10.bindPopup%28popup_29819dfa63684153ad54a137cebac462%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2935a18d18544649930e1267ce79578e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.66900510000001%2C%20-79.4422593%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d189b4f689ab41458c6d469327a804c1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_89998f6636fc498fa71820195f0c628c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_89998f6636fc498fa71820195f0c628c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDufferin%2C%20Dovercourt%20Village%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d189b4f689ab41458c6d469327a804c1.setContent%28html_89998f6636fc498fa71820195f0c628c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2935a18d18544649930e1267ce79578e.bindPopup%28popup_d189b4f689ab41458c6d469327a804c1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ce43f384f4634818bcd3c255dd45543a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.647926700000006%2C%20-79.4197497%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ef1d664d1d8f42b2b0284ef8923e2cda%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e3a96d70dc8b4781ab389809e41a6d33%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e3a96d70dc8b4781ab389809e41a6d33%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELittle%20Portugal%2C%20Trinity%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ef1d664d1d8f42b2b0284ef8923e2cda.setContent%28html_e3a96d70dc8b4781ab389809e41a6d33%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ce43f384f4634818bcd3c255dd45543a.bindPopup%28popup_ef1d664d1d8f42b2b0284ef8923e2cda%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1defe58a711046b08746161ad73df88e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6368472%2C%20-79.42819140000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_14bad9a80a6144e4b435956ba99bfe95%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f76c4d5f38bf496dab6f690f3af3f7d7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f76c4d5f38bf496dab6f690f3af3f7d7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrockton%2C%20Parkdale%20Village%2C%20Exhibition%20Place%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_14bad9a80a6144e4b435956ba99bfe95.setContent%28html_f76c4d5f38bf496dab6f690f3af3f7d7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1defe58a711046b08746161ad73df88e.bindPopup%28popup_14bad9a80a6144e4b435956ba99bfe95%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_22558ef7d1c74c3fa3c26e4189b95cc0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.713756200000006%2C%20-79.4900738%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_34e19b3b2ecf466798f0d4f0c273cfc4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_275941a6224148199a708c32c8d88360%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_275941a6224148199a708c32c8d88360%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Park%2C%20Maple%20Leaf%20Park%2C%20Upwood%20Park%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_34e19b3b2ecf466798f0d4f0c273cfc4.setContent%28html_275941a6224148199a708c32c8d88360%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_22558ef7d1c74c3fa3c26e4189b95cc0.bindPopup%28popup_34e19b3b2ecf466798f0d4f0c273cfc4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_85dc4d6f22ae4b79bb9d3cea26936514%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6911158%2C%20-79.47601329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_068a9514e61f4b7e8956aca40c4c0fd0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e7e4a98f71fa4cc2a74eef37c4cd0fcf%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e7e4a98f71fa4cc2a74eef37c4cd0fcf%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDel%20Ray%2C%20Mount%20Dennis%2C%20Keelsdale%20and%20Silverthorn%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_068a9514e61f4b7e8956aca40c4c0fd0.setContent%28html_e7e4a98f71fa4cc2a74eef37c4cd0fcf%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_85dc4d6f22ae4b79bb9d3cea26936514.bindPopup%28popup_068a9514e61f4b7e8956aca40c4c0fd0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5051dedf112048a388e425dc9b22dfc7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.67318529999999%2C%20-79.48726190000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fed75d60895242f4bfbd253e377f3069%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c21d558d0f1f421285b00c1821251dae%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c21d558d0f1f421285b00c1821251dae%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20The%20Junction%20North%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fed75d60895242f4bfbd253e377f3069.setContent%28html_c21d558d0f1f421285b00c1821251dae%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5051dedf112048a388e425dc9b22dfc7.bindPopup%28popup_fed75d60895242f4bfbd253e377f3069%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9c55e20caee44f2a9ac159822bf69092%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6616083%2C%20-79.46476329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_20c98d9b82c94520be1923e0b523ba20%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7d0cddc955b9424da22419b1fab175a9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7d0cddc955b9424da22419b1fab175a9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHigh%20Park%2C%20The%20Junction%20South%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_20c98d9b82c94520be1923e0b523ba20.setContent%28html_7d0cddc955b9424da22419b1fab175a9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9c55e20caee44f2a9ac159822bf69092.bindPopup%28popup_20c98d9b82c94520be1923e0b523ba20%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2a9a3a69fc76453d9de6bd41f5178346%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6489597%2C%20-79.456325%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4473878b4132415f9099a7aa3d18f494%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bf8b266a482341f2a182b3d320a1cf8e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bf8b266a482341f2a182b3d320a1cf8e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkdale%2C%20Roncesvalles%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4473878b4132415f9099a7aa3d18f494.setContent%28html_bf8b266a482341f2a182b3d320a1cf8e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2a9a3a69fc76453d9de6bd41f5178346.bindPopup%28popup_4473878b4132415f9099a7aa3d18f494%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bc9ca6c88d4e4961a6c2dacf52c56b61%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6515706%2C%20-79.4844499%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d76e48e003404803919e677d98d5b0cb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f013d29cce8d4504a09e0d17a806ce00%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f013d29cce8d4504a09e0d17a806ce00%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20Swansea%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d76e48e003404803919e677d98d5b0cb.setContent%28html_f013d29cce8d4504a09e0d17a806ce00%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bc9ca6c88d4e4961a6c2dacf52c56b61.bindPopup%28popup_d76e48e003404803919e677d98d5b0cb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2b21b222452549c9807bc144372d56c8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6623015%2C%20-79.3894938%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7b5c7c58efb749fcb52ec743d5aba769%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_bbe410366acc4ee18e8c874fdefe0ab7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_bbe410366acc4ee18e8c874fdefe0ab7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueen%26%2339%3Bs%20Park%2C%20Ontario%20Provincial%20Government%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7b5c7c58efb749fcb52ec743d5aba769.setContent%28html_bbe410366acc4ee18e8c874fdefe0ab7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2b21b222452549c9807bc144372d56c8.bindPopup%28popup_7b5c7c58efb749fcb52ec743d5aba769%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a8d7ae3ff3ce45359a89422fceac0177%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6369656%2C%20-79.61581899999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1e610e396d4f4a1dbd2832ba839dec01%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1622b3eb1b854945b052081cd94acb01%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1622b3eb1b854945b052081cd94acb01%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECanada%20Post%20Gateway%20Processing%20Centre%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1e610e396d4f4a1dbd2832ba839dec01.setContent%28html_1622b3eb1b854945b052081cd94acb01%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a8d7ae3ff3ce45359a89422fceac0177.bindPopup%28popup_1e610e396d4f4a1dbd2832ba839dec01%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e54c7e56fde94b94999e504acbfccfa3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6627439%2C%20-79.321558%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7009e84e78c64f9c9933326c490be4f6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c5d9d75d1eac410594a57ab2f859682a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c5d9d75d1eac410594a57ab2f859682a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBusiness%20reply%20mail%20Processing%20Centre%2C%20South%20Central%20Letter%20Processing%20Plant%20Toronto%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7009e84e78c64f9c9933326c490be4f6.setContent%28html_c5d9d75d1eac410594a57ab2f859682a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e54c7e56fde94b94999e504acbfccfa3.bindPopup%28popup_7009e84e78c64f9c9933326c490be4f6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c2bc03304d834a80977b1c5ff2a0a3c8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6056466%2C%20-79.50132070000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e983298add56499081cda17405413a79%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_22f194e33c9145c4baa8e3c74ef61248%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_22f194e33c9145c4baa8e3c74ef61248%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENew%20Toronto%2C%20Mimico%20South%2C%20Humber%20Bay%20Shores%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e983298add56499081cda17405413a79.setContent%28html_22f194e33c9145c4baa8e3c74ef61248%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c2bc03304d834a80977b1c5ff2a0a3c8.bindPopup%28popup_e983298add56499081cda17405413a79%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d24093e04eb34ba5b03704fd4e5decc6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.60241370000001%2C%20-79.54348409999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5a8240752080486bb2eae61c31122664%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a6cbdccded3d4e8bb226e816f59301a7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a6cbdccded3d4e8bb226e816f59301a7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EAlderwood%2C%20Long%20Branch%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5a8240752080486bb2eae61c31122664.setContent%28html_a6cbdccded3d4e8bb226e816f59301a7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d24093e04eb34ba5b03704fd4e5decc6.bindPopup%28popup_5a8240752080486bb2eae61c31122664%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ec8347c565124780b2190adac2b05166%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.653653600000005%2C%20-79.5069436%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d61b7c3390b84cba917c751402951607%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_edbc68e653b84040ac6329967bf6109c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_edbc68e653b84040ac6329967bf6109c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Kingsway%2C%20Montgomery%20Road%2C%20Old%20Mill%20North%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d61b7c3390b84cba917c751402951607.setContent%28html_edbc68e653b84040ac6329967bf6109c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ec8347c565124780b2190adac2b05166.bindPopup%28popup_d61b7c3390b84cba917c751402951607%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2052656b915b4b4e8c17cf7d4941bcca%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6362579%2C%20-79.49850909999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_510a5cf7465545fbbecb6bd6f8a418a7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eccedac75d344766b4eaab96f7ac78e5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eccedac75d344766b4eaab96f7ac78e5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EOld%20Mill%20South%2C%20King%26%2339%3Bs%20Mill%20Park%2C%20Sunnylea%2C%20Humber%20Bay%2C%20Mimico%20NE%2C%20The%20Queensway%20East%2C%20Royal%20York%20South%20East%2C%20Kingsway%20Park%20South%20East%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_510a5cf7465545fbbecb6bd6f8a418a7.setContent%28html_eccedac75d344766b4eaab96f7ac78e5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2052656b915b4b4e8c17cf7d4941bcca.bindPopup%28popup_510a5cf7465545fbbecb6bd6f8a418a7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_713d0c8be83a4e2894bf3b79c91ebfe8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6288408%2C%20-79.52099940000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e3c50f4174244b9faeb687ecfd2b7a26%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eb4f57f21bdd412997c3cab4977dcd2b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eb4f57f21bdd412997c3cab4977dcd2b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMimico%20NW%2C%20The%20Queensway%20West%2C%20South%20of%20Bloor%2C%20Kingsway%20Park%20South%20West%2C%20Royal%20York%20South%20West%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e3c50f4174244b9faeb687ecfd2b7a26.setContent%28html_eb4f57f21bdd412997c3cab4977dcd2b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_713d0c8be83a4e2894bf3b79c91ebfe8.bindPopup%28popup_e3c50f4174244b9faeb687ecfd2b7a26%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3ec1af498de14acd8a5b382f6b6f0e0e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6678556%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d35f1337df84445dbca54da4046058ee%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_97ba8c9baf5e4b0384f67e601d996909%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_97ba8c9baf5e4b0384f67e601d996909%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIslington%20Avenue%2C%20Humber%20Valley%20Village%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d35f1337df84445dbca54da4046058ee.setContent%28html_97ba8c9baf5e4b0384f67e601d996909%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3ec1af498de14acd8a5b382f6b6f0e0e.bindPopup%28popup_d35f1337df84445dbca54da4046058ee%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac7517f01d2540c5a13486a89b878135%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6509432%2C%20-79.55472440000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_be8640d20d284acb9eefc45ea22b265c%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_50289f82bdf84b5f940d095a26838ca7%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_50289f82bdf84b5f940d095a26838ca7%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWest%20Deane%20Park%2C%20Princess%20Gardens%2C%20Martin%20Grove%2C%20Islington%2C%20Cloverdale%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_be8640d20d284acb9eefc45ea22b265c.setContent%28html_50289f82bdf84b5f940d095a26838ca7%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ac7517f01d2540c5a13486a89b878135.bindPopup%28popup_be8640d20d284acb9eefc45ea22b265c%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ed0f699a42fb4893906b4606cf8e1f96%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6435152%2C%20-79.57720079999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff5c00%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ff5c00%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_05cb3cab951c490596b537bce6ca1bbc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_185f82976f0d47ab93ab61f73789fb79%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_185f82976f0d47ab93ab61f73789fb79%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EEringate%2C%20Bloordale%20Gardens%2C%20Old%20Burnhamthorpe%2C%20Markland%20Wood%20Cluster%3A%202.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_05cb3cab951c490596b537bce6ca1bbc.setContent%28html_185f82976f0d47ab93ab61f73789fb79%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ed0f699a42fb4893906b4606cf8e1f96.bindPopup%28popup_05cb3cab951c490596b537bce6ca1bbc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ac270aa9eb4847aab447695abb11ce72%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7563033%2C%20-79.56596329999999%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_40c958a8ff304e82ad03554bc93e38e8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ed25a77949ce4eafba35ddf0909a0e99%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ed25a77949ce4eafba35ddf0909a0e99%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumber%20Summit%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_40c958a8ff304e82ad03554bc93e38e8.setContent%28html_ed25a77949ce4eafba35ddf0909a0e99%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ac270aa9eb4847aab447695abb11ce72.bindPopup%28popup_40c958a8ff304e82ad03554bc93e38e8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_dab7f37b70c14db08013ea2c05825376%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7247659%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23b30000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23b30000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e46eaceb7c8c4ac9a494e194e85a55c0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3c8a0c42588d451a8f58151adebd1d6b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3c8a0c42588d451a8f58151adebd1d6b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHumberlea%2C%20Emery%20Cluster%3A%201.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e46eaceb7c8c4ac9a494e194e85a55c0.setContent%28html_3c8a0c42588d451a8f58151adebd1d6b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_dab7f37b70c14db08013ea2c05825376.bindPopup%28popup_e46eaceb7c8c4ac9a494e194e85a55c0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a0b5e957963c4000978054127ab7efb9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.706876%2C%20-79.51818840000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_72b7d151f11c43d994b411e8880befa0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1034e75a73af426f8680521b543caf52%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1034e75a73af426f8680521b543caf52%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWeston%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_72b7d151f11c43d994b411e8880befa0.setContent%28html_1034e75a73af426f8680521b543caf52%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a0b5e957963c4000978054127ab7efb9.bindPopup%28popup_72b7d151f11c43d994b411e8880befa0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_269f4222086f4fcaa0e5e36a26f3b296%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.696319%2C%20-79.53224240000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b7741b5ca0e346f1a6f0574bf38d3be6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_030b92b548ec4c1ab19c0a7c104f6402%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_030b92b548ec4c1ab19c0a7c104f6402%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EWestmount%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b7741b5ca0e346f1a6f0574bf38d3be6.setContent%28html_030b92b548ec4c1ab19c0a7c104f6402%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_269f4222086f4fcaa0e5e36a26f3b296.bindPopup%28popup_b7741b5ca0e346f1a6f0574bf38d3be6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8bfde229398b43659f0e3a407849272b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6889054%2C%20-79.55472440000001%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f32eaf29c0d14b06bc6f7abcf5169d06%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_979a243311454a65a71559e55aced167%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_979a243311454a65a71559e55aced167%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKingsview%20Village%2C%20St.%20Phillips%2C%20Martin%20Grove%20Gardens%2C%20Richview%20Gardens%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f32eaf29c0d14b06bc6f7abcf5169d06.setContent%28html_979a243311454a65a71559e55aced167%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8bfde229398b43659f0e3a407849272b.bindPopup%28popup_f32eaf29c0d14b06bc6f7abcf5169d06%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_1fcfce78beab4fc89d8df35355a05643%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.739416399999996%2C%20-79.5884369%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%230b0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%230b0000%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3f7bf7da747347fb9ccdc3bbeb18d595%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f7bf3d906b094f3fa3c9740f4a7f912c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f7bf3d906b094f3fa3c9740f4a7f912c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESouth%20Steeles%2C%20Silverstone%2C%20Humbergate%2C%20Jamestown%2C%20Mount%20Olive%2C%20Beaumond%20Heights%2C%20Thistletown%2C%20Albion%20Gardens%20Cluster%3A%200.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3f7bf7da747347fb9ccdc3bbeb18d595.setContent%28html_f7bf3d906b094f3fa3c9740f4a7f912c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_1fcfce78beab4fc89d8df35355a05643.bindPopup%28popup_3f7bf7da747347fb9ccdc3bbeb18d595%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6d38de391fda4a138fabc4c73ab77e40%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.706748299999994%2C%20-79.5940544%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffff07%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20false%2C%20%22fillColor%22%3A%20%22%23ffff07%22%2C%20%22fillOpacity%22%3A%200.2%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_3931d548401c49459513e589ee788873%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1cd21d0a4f5a4a6bb830e19936ef7f91%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_68e9b0995bd649968115a1a658ace81f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_68e9b0995bd649968115a1a658ace81f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorthwest%2C%20West%20Humber%20-%20Clairville%20Cluster%3A%203.0%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1cd21d0a4f5a4a6bb830e19936ef7f91.setContent%28html_68e9b0995bd649968115a1a658ace81f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6d38de391fda4a138fabc4c73ab77e40.bindPopup%28popup_1cd21d0a4f5a4a6bb830e19936ef7f91%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



<h3> The result of the clustering is actually quite nice - we can clearly see Toronto's downtown area (orange), as well as more suburban areas (black and red) and two outliers (yellow, white). For further examination, let's take a look at the cluster groups! </h3>


```python
toronto_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
      <td>2.0</td>
      <td>Fast Food Restaurant</td>
      <td>Trail</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Spa</td>
      <td>Supermarket</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Caribbean Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
      <td>1.0</td>
      <td>Breakfast Spot</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Italian Restaurant</td>
      <td>Fireworks Store</td>
      <td>Falafel Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Flea Market</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
      <td>0.0</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Liquor Store</td>
      <td>Supermarket</td>
      <td>Greek Restaurant</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
      <td>2.0</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Mobile Phone Shop</td>
      <td>Indian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farm</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
      <td>2.0</td>
      <td>Bakery</td>
      <td>Gas Station</td>
      <td>Bank</td>
      <td>Indian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Sporting Goods Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Burger Joint</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For convenient display
from IPython.display import display_html

def html_string():
    html_str = ''
    for i in range(kclusters):
        html_str += 'Cluster Group: '+ str(i) + '<br>'
        df = toronto_merged[toronto_merged['Cluster Labels'] == i]
        df.loc[:, 'Neighborhood']
        
        
        html_str += toronto_merged[toronto_merged['Cluster Labels'] == i].to_html()    
    
html_str = ''
for i in range(kclusters):
    html_str += 'Cluster Group: '+ str(i) +'<br>'
    df = toronto_merged[toronto_merged['Cluster Labels'] == i]
    df = df.iloc[:, [2] + list(np.arange(6, toronto_merged.shape[1]))]
    html_str += df.to_html()
    html_str += 2*'<br>'
display_html(html_str.replace('table','table style="display:inline"'),raw=True)
```


Cluster Group: 0<br><table style="display:inline" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Guildwood, Morningside, West Hill</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Liquor Store</td>
      <td>Supermarket</td>
      <td>Greek Restaurant</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Scarborough Village</td>
      <td>Convenience Store</td>
      <td>Ice Cream Shop</td>
      <td>Bowling Alley</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>Grocery Store</td>
      <td>Coffee Shop</td>
      <td>Intersection</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kennedy Park, Ionview, East Birchmount Park</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Discount Store</td>
      <td>Grocery Store</td>
      <td>Fast Food Restaurant</td>
      <td>Rental Car Location</td>
      <td>Light Rail Station</td>
      <td>Bank</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cliffside, Cliffcrest, Scarborough Village West</td>
      <td>Pizza Place</td>
      <td>Beach</td>
      <td>Ice Cream Shop</td>
      <td>Sports Bar</td>
      <td>Restaurant</td>
      <td>Auto Garage</td>
      <td>Park</td>
      <td>Pharmacy</td>
      <td>Field</td>
      <td>Fireworks Store</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Wexford, Maryvale</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pizza Place</td>
      <td>Burger Joint</td>
      <td>Flea Market</td>
      <td>Grocery Store</td>
      <td>Soccer Field</td>
      <td>Fish Market</td>
      <td>Seafood Restaurant</td>
      <td>Supermarket</td>
      <td>Korean Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Agincourt</td>
      <td>Chinese Restaurant</td>
      <td>Shopping Mall</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Bakery</td>
      <td>Malay Restaurant</td>
      <td>Lounge</td>
      <td>Motorcycle Shop</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Clarks Corners, Tam O'Shanter, Sullivan</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Pizza Place</td>
      <td>Convenience Store</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Gas Station</td>
      <td>Market</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Milliken, Agincourt North, Steeles East, L'Amoreaux East</td>
      <td>Chinese Restaurant</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Intersection</td>
      <td>Pharmacy</td>
      <td>Dessert Shop</td>
      <td>Coffee Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Steeles West, L'Amoreaux West</td>
      <td>Chinese Restaurant</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Intersection</td>
      <td>Bank</td>
      <td>Hotpot Restaurant</td>
      <td>Caribbean Restaurant</td>
      <td>Sandwich Place</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hillcrest Village</td>
      <td>Pharmacy</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Intersection</td>
      <td>Fast Food Restaurant</td>
      <td>Shopping Mall</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Willowdale, Willowdale West</td>
      <td>Pharmacy</td>
      <td>Baby Store</td>
      <td>Butcher</td>
      <td>Discount Store</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Bus Line</td>
      <td>Pizza Place</td>
      <td>Convenience Store</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Northwood Park, York University</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Furniture / Home Store</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Modern European Restaurant</td>
      <td>Sandwich Place</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Caledonia-Fairbanks</td>
      <td>Pharmacy</td>
      <td>Mexican Restaurant</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Bus Stop</td>
      <td>Grocery Store</td>
      <td>Coffee Shop</td>
      <td>Discount Store</td>
      <td>Falafel Restaurant</td>
      <td>Bus Line</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Alderwood, Long Branch</td>
      <td>Discount Store</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Pharmacy</td>
      <td>Moroccan Restaurant</td>
      <td>Dance Studio</td>
      <td>Garden Center</td>
      <td>Gas Station</td>
      <td>Donut Shop</td>
      <td>Bagel Shop</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Weston</td>
      <td>Pizza Place</td>
      <td>Train Station</td>
      <td>Soccer Field</td>
      <td>Diner</td>
      <td>Jewelry Store</td>
      <td>Discount Store</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Westmount</td>
      <td>Pizza Place</td>
      <td>Gas Station</td>
      <td>Flea Market</td>
      <td>Discount Store</td>
      <td>Supermarket</td>
      <td>Sandwich Place</td>
      <td>Coffee Shop</td>
      <td>Intersection</td>
      <td>Golf Course</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>100</th>
      <td>Kingsview Village, St. Phillips, Martin Grove Gardens, Richview Gardens</td>
      <td>Pharmacy</td>
      <td>Bank</td>
      <td>Supermarket</td>
      <td>Gas Station</td>
      <td>Pizza Place</td>
      <td>Sandwich Place</td>
      <td>American Restaurant</td>
      <td>Coffee Shop</td>
      <td>Intersection</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>101</th>
      <td>South Steeles, Silverstone, Humbergate, Jamestown, Mount Olive, Beaumond Heights, Thistletown, Albion Gardens</td>
      <td>Pizza Place</td>
      <td>Grocery Store</td>
      <td>Pharmacy</td>
      <td>Park</td>
      <td>Beer Store</td>
      <td>Auto Garage</td>
      <td>Fried Chicken Joint</td>
      <td>Sandwich Place</td>
      <td>Video Store</td>
      <td>Bus Line</td>
    </tr>
  </tbody>
</table style="display:inline"><br><br>Cluster Group: 1<br><table style="display:inline" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>Breakfast Spot</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Italian Restaurant</td>
      <td>Fireworks Store</td>
      <td>Falafel Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Flea Market</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Birch Cliff, Cliffside West</td>
      <td>Park</td>
      <td>General Entertainment</td>
      <td>Thai Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>College Stadium</td>
      <td>Café</td>
      <td>Diner</td>
      <td>Restaurant</td>
      <td>Dessert Shop</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>23</th>
      <td>York Mills West</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Pet Store</td>
      <td>Dog Run</td>
      <td>French Restaurant</td>
      <td>Golf Course</td>
      <td>Gas Station</td>
      <td>Bubble Tea Shop</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Parkwoods</td>
      <td>Park</td>
      <td>Pharmacy</td>
      <td>Bus Stop</td>
      <td>Shopping Mall</td>
      <td>ATM</td>
      <td>Shop &amp; Service</td>
      <td>Fast Food Restaurant</td>
      <td>Tennis Court</td>
      <td>Laundry Service</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Del Ray, Mount Dennis, Keelsdale and Silverthorn</td>
      <td>Furniture / Home Store</td>
      <td>Grocery Store</td>
      <td>Discount Store</td>
      <td>Shopping Mall</td>
      <td>Gas Station</td>
      <td>Wine Shop</td>
      <td>Playground</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Old Mill South, King's Mill Park, Sunnylea, Humber Bay, Mimico NE, The Queensway East, Royal York South East, Kingsway Park South East</td>
      <td>Park</td>
      <td>Italian Restaurant</td>
      <td>Bus Stop</td>
      <td>Ice Cream Shop</td>
      <td>Shopping Mall</td>
      <td>Eastern European Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Islington Avenue, Humber Valley Village</td>
      <td>Pharmacy</td>
      <td>Bakery</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Playground</td>
      <td>Shopping Mall</td>
      <td>Golf Course</td>
      <td>Café</td>
      <td>Grocery Store</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>94</th>
      <td>West Deane Park, Princess Gardens, Martin Grove, Islington, Cloverdale</td>
      <td>Park</td>
      <td>Hotel</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Theater</td>
      <td>Grocery Store</td>
      <td>Gym</td>
      <td>Clothing Store</td>
      <td>Bank</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Humber Summit</td>
      <td>Electronics Store</td>
      <td>Italian Restaurant</td>
      <td>Pharmacy</td>
      <td>Shopping Mall</td>
      <td>Park</td>
      <td>Arts &amp; Crafts Store</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Pizza Place</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Humberlea, Emery</td>
      <td>Golf Course</td>
      <td>Bakery</td>
      <td>Convenience Store</td>
      <td>Storage Facility</td>
      <td>Park</td>
      <td>Discount Store</td>
      <td>Gas Station</td>
      <td>Event Space</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table style="display:inline"><br><br>Cluster Group: 2<br><table style="display:inline" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Malvern, Rouge</td>
      <td>Fast Food Restaurant</td>
      <td>Trail</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Spa</td>
      <td>Supermarket</td>
      <td>Bank</td>
      <td>Bakery</td>
      <td>Caribbean Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Woburn</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Mobile Phone Shop</td>
      <td>Indian Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farm</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cedarbrae</td>
      <td>Bakery</td>
      <td>Gas Station</td>
      <td>Bank</td>
      <td>Indian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Sporting Goods Shop</td>
      <td>Caribbean Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Golden Mile, Clairlea, Oakridge</td>
      <td>Intersection</td>
      <td>Bus Line</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Convenience Store</td>
      <td>Beer Store</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Soccer Field</td>
      <td>General Entertainment</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dorset Park, Wexford Heights, Scarborough Town Centre</td>
      <td>Electronics Store</td>
      <td>Bakery</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Light Rail Station</td>
      <td>Indian Restaurant</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fairview, Henry Farm, Oriole</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Juice Bar</td>
      <td>Bakery</td>
      <td>Bank</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Fried Chicken Joint</td>
      <td>Liquor Store</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bayview Village</td>
      <td>Bank</td>
      <td>Gas Station</td>
      <td>Grocery Store</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Café</td>
      <td>Trail</td>
      <td>Park</td>
      <td>Skating Rink</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Willowdale, Newtonbrook</td>
      <td>Korean Restaurant</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Diner</td>
      <td>Middle Eastern Restaurant</td>
      <td>Grocery Store</td>
      <td>Dessert Shop</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Willowdale, Willowdale East</td>
      <td>Coffee Shop</td>
      <td>Bubble Tea Shop</td>
      <td>Ramen Restaurant</td>
      <td>Korean Restaurant</td>
      <td>Pizza Place</td>
      <td>Japanese Restaurant</td>
      <td>Sandwich Place</td>
      <td>Fast Food Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Don Mills</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Pizza Place</td>
      <td>Supermarket</td>
      <td>Burger Joint</td>
      <td>Bank</td>
      <td>Athletics &amp; Sports</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Don Mills</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Pizza Place</td>
      <td>Supermarket</td>
      <td>Burger Joint</td>
      <td>Bank</td>
      <td>Athletics &amp; Sports</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Convenience Store</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Supermarket</td>
      <td>Mediterranean Restaurant</td>
      <td>Community Center</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Downsview</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Gas Station</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Downsview</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Gas Station</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Downsview</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Gas Station</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Downsview</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Gas Station</td>
      <td>Pharmacy</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Victoria Village</td>
      <td>Coffee Shop</td>
      <td>Portuguese Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Lounge</td>
      <td>Hockey Arena</td>
      <td>Golf Course</td>
      <td>Playground</td>
      <td>Men's Store</td>
      <td>Intersection</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Parkview Hill, Woodbine Gardens</td>
      <td>Brewery</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Construction &amp; Landscaping</td>
      <td>Intersection</td>
      <td>Athletics &amp; Sports</td>
      <td>Rock Climbing Spot</td>
      <td>Gym / Fitness Center</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Woodbine Heights</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Skating Rink</td>
      <td>Café</td>
      <td>Thai Restaurant</td>
      <td>Athletics &amp; Sports</td>
      <td>Farmers Market</td>
      <td>Bus Stop</td>
    </tr>
    <tr>
      <th>37</th>
      <td>The Beaches</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Pizza Place</td>
      <td>Beach</td>
      <td>Japanese Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Gastropub</td>
      <td>Burger Joint</td>
      <td>Bar</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Leaside</td>
      <td>Coffee Shop</td>
      <td>Sporting Goods Shop</td>
      <td>Furniture / Home Store</td>
      <td>Grocery Store</td>
      <td>Burger Joint</td>
      <td>Electronics Store</td>
      <td>Bank</td>
      <td>Sandwich Place</td>
      <td>Brewery</td>
      <td>Department Store</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Thorncliffe Park</td>
      <td>Coffee Shop</td>
      <td>Indian Restaurant</td>
      <td>Grocery Store</td>
      <td>Turkish Restaurant</td>
      <td>Shopping Mall</td>
      <td>Sandwich Place</td>
      <td>Burger Joint</td>
      <td>Brewery</td>
      <td>Pizza Place</td>
      <td>Afghan Restaurant</td>
    </tr>
    <tr>
      <th>40</th>
      <td>East Toronto, Broadview North (Old East York)</td>
      <td>Greek Restaurant</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Pharmacy</td>
      <td>Beer Bar</td>
      <td>Fast Food Restaurant</td>
      <td>Ethiopian Restaurant</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>41</th>
      <td>The Danforth West, Riverdale</td>
      <td>Greek Restaurant</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bank</td>
      <td>Italian Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Discount Store</td>
      <td>Yoga Studio</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>42</th>
      <td>India Bazaar, The Beaches West</td>
      <td>Indian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Park</td>
      <td>Beach</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Brewery</td>
      <td>Bakery</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Studio District</td>
      <td>Coffee Shop</td>
      <td>Vietnamese Restaurant</td>
      <td>American Restaurant</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Diner</td>
      <td>Brewery</td>
      <td>French Restaurant</td>
      <td>Café</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Lawrence Park</td>
      <td>Café</td>
      <td>Bookstore</td>
      <td>Park</td>
      <td>College Quad</td>
      <td>College Gym</td>
      <td>Coffee Shop</td>
      <td>Trail</td>
      <td>Pharmacy</td>
      <td>Gym / Fitness Center</td>
      <td>Zoo</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Davisville North</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Fast Food Restaurant</td>
      <td>Pizza Place</td>
      <td>Sushi Restaurant</td>
      <td>Dessert Shop</td>
      <td>Movie Theater</td>
    </tr>
    <tr>
      <th>46</th>
      <td>North Toronto West, Lawrence Park</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Sporting Goods Shop</td>
      <td>Skating Rink</td>
      <td>Park</td>
      <td>Diner</td>
      <td>Café</td>
      <td>Mexican Restaurant</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Davisville</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Indian Restaurant</td>
      <td>Restaurant</td>
      <td>Dessert Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Moore Park, Summerhill East</td>
      <td>Italian Restaurant</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Gym</td>
      <td>Thai Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Spa</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Rosedale</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Grocery Store</td>
      <td>Sandwich Place</td>
      <td>Bank</td>
      <td>Candy Store</td>
      <td>BBQ Joint</td>
      <td>Bistro</td>
      <td>Juice Bar</td>
      <td>Athletics &amp; Sports</td>
    </tr>
    <tr>
      <th>51</th>
      <td>St. James Town, Cabbagetown</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Gastropub</td>
      <td>Diner</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
      <td>Jewelry Store</td>
      <td>Theater</td>
      <td>Gift Shop</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Church and Wellesley</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Hotel</td>
      <td>Diner</td>
      <td>Bookstore</td>
      <td>Caribbean Restaurant</td>
      <td>Restaurant</td>
      <td>Ramen Restaurant</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Regent Park, Harbourfront</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Bakery</td>
      <td>Breakfast Spot</td>
      <td>Diner</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Garden District, Ryerson</td>
      <td>Coffee Shop</td>
      <td>Gastropub</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Diner</td>
      <td>Theater</td>
      <td>Clothing Store</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>55</th>
      <td>St. James Town</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Gastropub</td>
      <td>Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>Theater</td>
      <td>Creperie</td>
      <td>Art Gallery</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Park</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Bakery</td>
      <td>Gastropub</td>
      <td>Art Gallery</td>
      <td>Creperie</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Ramen Restaurant</td>
      <td>Clothing Store</td>
      <td>Park</td>
      <td>Diner</td>
      <td>Sushi Restaurant</td>
      <td>Sandwich Place</td>
      <td>Tea Room</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Richmond, Adelaide, King</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Theater</td>
      <td>Gastropub</td>
      <td>Pizza Place</td>
      <td>Plaza</td>
      <td>Restaurant</td>
      <td>Sandwich Place</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Theater</td>
      <td>Brewery</td>
      <td>Aquarium</td>
      <td>Baseball Stadium</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Theater</td>
      <td>Cocktail Bar</td>
      <td>Seafood Restaurant</td>
      <td>Sandwich Place</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Seafood Restaurant</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Concert Hall</td>
      <td>Theater</td>
      <td>Gastropub</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Park</td>
      <td>Sandwich Place</td>
      <td>Juice Bar</td>
      <td>Thai Restaurant</td>
      <td>Baby Store</td>
      <td>Bagel Shop</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Roselawn</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Spa</td>
      <td>Café</td>
      <td>Pharmacy</td>
      <td>Coffee Shop</td>
      <td>Bagel Shop</td>
      <td>Bakery</td>
      <td>Gym</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Forest Hill North &amp; West, Forest Hill Road Park</td>
      <td>Park</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bank</td>
      <td>Pharmacy</td>
      <td>Burger Joint</td>
      <td>Sushi Restaurant</td>
      <td>Bakery</td>
      <td>Japanese Restaurant</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Gym</td>
      <td>Pub</td>
      <td>Grocery Store</td>
      <td>Restaurant</td>
      <td>Bakery</td>
      <td>Museum</td>
    </tr>
    <tr>
      <th>66</th>
      <td>University of Toronto, Harbord</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Beer Bar</td>
      <td>Mexican Restaurant</td>
      <td>Bookstore</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Yoga Studio</td>
      <td>Bar</td>
      <td>Caribbean Restaurant</td>
      <td>Art Gallery</td>
      <td>Taco Place</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>68</th>
      <td>CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport</td>
      <td>Harbor / Marina</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Sculpture Garden</td>
      <td>Dog Run</td>
      <td>Scenic Lookout</td>
      <td>Airport</td>
      <td>Airport Lounge</td>
      <td>Garden</td>
      <td>Dance Studio</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Stn A PO Boxes</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Gastropub</td>
      <td>Japanese Restaurant</td>
      <td>Hotel</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>70</th>
      <td>First Canadian Place, Underground city</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Theater</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Cocktail Bar</td>
      <td>American Restaurant</td>
      <td>Concert Hall</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Lawrence Manor, Lawrence Heights</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Restaurant</td>
      <td>Furniture / Home Store</td>
      <td>Fried Chicken Joint</td>
      <td>Sushi Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Boutique</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Glencairn</td>
      <td>Grocery Store</td>
      <td>Fast Food Restaurant</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Gas Station</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Bus Line</td>
      <td>Fish Market</td>
      <td>Flower Shop</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Humewood-Cedarvale</td>
      <td>Convenience Store</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Grocery Store</td>
      <td>Bagel Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bank</td>
      <td>Gastropub</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Christie</td>
      <td>Korean Restaurant</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Grocery Store</td>
      <td>Mexican Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Pizza Place</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Ethiopian Restaurant</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Dufferin, Dovercourt Village</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Bakery</td>
      <td>Camera Store</td>
      <td>Gourmet Shop</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Little Portugal, Trinity</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Bakery</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Pizza Place</td>
      <td>Cocktail Bar</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Bar</td>
      <td>Bakery</td>
      <td>Furniture / Home Store</td>
      <td>Gift Shop</td>
      <td>Tibetan Restaurant</td>
      <td>Performing Arts Venue</td>
      <td>Arts &amp; Crafts Store</td>
    </tr>
    <tr>
      <th>79</th>
      <td>North Park, Maple Leaf Park, Upwood Park</td>
      <td>Coffee Shop</td>
      <td>Convenience Store</td>
      <td>Intersection</td>
      <td>Chinese Restaurant</td>
      <td>Park</td>
      <td>Gas Station</td>
      <td>Athletics &amp; Sports</td>
      <td>Dim Sum Restaurant</td>
      <td>Bakery</td>
      <td>Mediterranean Restaurant</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Runnymede, The Junction North</td>
      <td>Coffee Shop</td>
      <td>Pizza Place</td>
      <td>Brewery</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Gas Station</td>
      <td>Beer Store</td>
      <td>BBQ Joint</td>
      <td>Burger Joint</td>
      <td>Dive Bar</td>
    </tr>
    <tr>
      <th>82</th>
      <td>High Park, The Junction South</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Park</td>
      <td>Italian Restaurant</td>
      <td>Convenience Store</td>
      <td>Bakery</td>
      <td>Metro Station</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Parkdale, Roncesvalles</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Breakfast Spot</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
      <td>Pizza Place</td>
      <td>Grocery Store</td>
      <td>Thai Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Runnymede, Swansea</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Italian Restaurant</td>
      <td>Pub</td>
      <td>Falafel Restaurant</td>
      <td>Bank</td>
      <td>Sushi Restaurant</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Sushi Restaurant</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>Pizza Place</td>
      <td>Bubble Tea Shop</td>
      <td>Thai Restaurant</td>
      <td>Clothing Store</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Canada Post Gateway Processing Centre</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Middle Eastern Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bakery</td>
      <td>Asian Restaurant</td>
      <td>Bus Station</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Business reply mail Processing Centre, South Central Letter Processing Plant Toronto</td>
      <td>Park</td>
      <td>Pizza Place</td>
      <td>Coffee Shop</td>
      <td>Brewery</td>
      <td>Sushi Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Bakery</td>
      <td>Breakfast Spot</td>
      <td>Snack Place</td>
    </tr>
    <tr>
      <th>88</th>
      <td>New Toronto, Mimico South, Humber Bay Shores</td>
      <td>Park</td>
      <td>Dessert Shop</td>
      <td>Skating Rink</td>
      <td>Grocery Store</td>
      <td>Coffee Shop</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
    </tr>
    <tr>
      <th>90</th>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
      <td>Coffee Shop</td>
      <td>Park</td>
      <td>Dessert Shop</td>
      <td>Burger Joint</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Breakfast Spot</td>
      <td>Bank</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Mimico NW, The Queensway West, South of Bloor, Kingsway Park South West, Royal York South West</td>
      <td>Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Burger Joint</td>
      <td>Italian Restaurant</td>
      <td>Bank</td>
      <td>Grocery Store</td>
      <td>Gym</td>
      <td>Yoga Studio</td>
      <td>Bakery</td>
      <td>Burrito Place</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Eringate, Bloordale Gardens, Old Burnhamthorpe, Markland Wood</td>
      <td>Coffee Shop</td>
      <td>Convenience Store</td>
      <td>Shopping Mall</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Café</td>
      <td>Beer Store</td>
      <td>Grocery Store</td>
      <td>Pet Store</td>
      <td>Gas Station</td>
      <td>Pharmacy</td>
    </tr>
  </tbody>
</table style="display:inline"><br><br>Cluster Group: 3<br><table style="display:inline" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>102</th>
      <td>Northwest, West Humber - Clairville</td>
      <td>Hotel</td>
      <td>Rental Car Location</td>
      <td>Coffee Shop</td>
      <td>Farm</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
  </tbody>
</table style="display:inline"><br><br>Cluster Group: 4<br><table style="display:inline" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>York Mills, Silver Hills</td>
      <td>Park</td>
      <td>Pool</td>
      <td>Zoo</td>
      <td>Farm</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
      <td>Electronics Store</td>
      <td>Elementary School</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
  </tbody>
</table style="display:inline"><br><br>


<h3> The most common venue in Cluster Group 2 are coffee shops, which are all located in the center of town. Cluster Group 1 seems to be more rural with an abundance of parks and malls and even golf courses, whereas cluster group 0 seems to have a lot of pizza places and restaurants. Cluster groups 3 and 4 are outliers with farms, zoos, parks and pools as the most common venues. </h3>
