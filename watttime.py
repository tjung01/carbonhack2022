# # This code doesn't work without paid Watttime subscription
# # Working outside of GSF SDK
# # See https://github.com/Green-Software-Foundation/carbon-aware-sdk/blob/dev/docs/quickstart.md
# #     for SDK CLI options
#%%
# # Register
import requests
register_url = 'https://api2.watttime.org/v2/register'
params = {'username': 'tarijung',
         'password': 'S,dCCa#B@#cA#.+P4beV',
         'email': 'jungx367@umn.edu',
         'org': 'UMN'}
rsp = requests.post(register_url, json=params, verify=False)
print(rsp.text)

# %%
import numpy as np
login_url = 'https://api2.watttime.org/v2/login'
token = requests.get(login_url,verify=False, 
                   auth=HTTPBasicAuth('tarijung', 'S,dCCa#B@#cA#.+P4beV')).json()['token']

data_url = 'https://api2.watttime.org/v2/data'
headers = {'Authorization': 'Bearer {}'.format(token)}
start_lat_lon = (43.515,-96.425)
end_lat_lon = (47.933,-90.102)
granularity = [1, 0.1, 0.01]
def return_LatLon_range(start_lat_lon, end_lat_lon, g):
    lat =  np.arange(start_lat_lon[0], end_lat_lon[0],g)
    lon = np.arange(start_lat_lon[1], end_lat_lon[1],g)
    return list(zip(lat,lon))
g_queries = dict(zip(granularity,[return_LatLon_range(start_lat_lon, end_lat_lon, g) for g in granularity]))
g_df = pd.DataFrame()
for g, g_queries in g_queries.items():
    for query in g_queries:
        # # Summer evening
        params = dict(latitude = query[0],
                      longitude= query[1],
                      starttime="2019-08-01T18:00:00-0800",
                      endtime="2019-08-01T19:00:00-0800")
        rsp = requests.get(data_url, headers=headers, params=params).json()
        df = pd.json_normalize(rsp)
        df["Granularity"] = g
        g_df = pd.concat([g_df,df])
        # # Summer afternoon
        params = dict(latitude = query[0],
                      longitude= query[1],
                      starttime="2019-08-01T14:00:00-0800",
                      endtime="2019-08-01T15:00:00-0800")
        rsp = requests.get(data_url, headers=headers, params=params).json()
        df = pd.json_normalize(rsp)
        df["Granularity"] = g
        g_df = pd.concat([g_df,df])
        print(g_df)
# %%
