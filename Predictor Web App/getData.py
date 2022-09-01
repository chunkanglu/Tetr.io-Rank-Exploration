import requests
import numpy as np
import pandas as pd
import json

def get_data():
    return requests.get("https://ch.tetr.io/api/users/lists/league/all")

def transform_data(data):

  full_dict = json.loads(data.text)

  all_players = []

  working_dict = full_dict['data']['users']

  for p in range(len(working_dict)): # For every player

    player = {}

    for a in working_dict[p]: # For every attribute of player

        if a != "league":

          player[a] = working_dict[p][a] # Add attributes that are not part of league

    for a in working_dict[p]["league"]:

        player[a] = working_dict[p]["league"][a]

    all_players.append(player)

  df = pd.DataFrame(all_players, index=[i for i in range(len(all_players))])
    
  return df
