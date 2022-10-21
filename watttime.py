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

#%%
# # Log in & Token
from requests.auth import HTTPBasicAuth
login_url = 'https://api2.watttime.org/v2/login'
token = requests.get(login_url,verify=False, 
                   auth=HTTPBasicAuth('tarijung', 'S,dCCa#B@#cA#.+P4beV')).json()['token']
# print(rsp.json())

#%%
# List of Grid regions
list_url = 'https://api2.watttime.org/v2/ba-access'
headers = {'Authorization': 'Bearer {}'.format(token)}
params = {'all': 'true'}
grid_regions=requests.get(list_url,verify=False, headers=headers, params=params)
# print(grid_regions.text)

print(grid_regions.json()[0])
# %%
