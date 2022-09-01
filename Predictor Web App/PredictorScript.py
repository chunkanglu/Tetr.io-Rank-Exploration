# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/chunkanglu/Tetr.io-Rank-Predictor/blob/main/TetrioRankPredictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # Tetr.io Rank Predictor 
# 
# Goal: To predict Tetr.io rank based on other player statistics such as **APM (Attack-per-Minute)**, **PPS (Pieces-per-Second)**, and **VS Score**.
# 
# ### Note: 
# The comments here derive from the data taken on Februrary 17, 2021. Since this program collects the current live data, there may be variations in the results. 

# %%
import requests
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV

# %% [markdown]
# Get Tetr.io Tetra League data from API

# %%
def get_data():
    return requests.get("https://ch.tetr.io/api/users/lists/league/all")

full_data = get_data()

# %% [markdown]
# Turn data from JSON file into a Pandas DataFrame. We only include the user data and disregard the rest.

# %%
def transform_json_data(full_data):

  full_dict = json.loads(full_data.text)

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

df = transform_data(full_data)

# Optionally save to csv
# df.to_csv('TetrioAll.csv') 


# %%
df


# %%
df.info()

# %% [markdown]
# I see there are 2 random rows in `vs` that have missing values so I drop them

# %%
df.dropna(axis=0, subset=['vs'], inplace=True)

# %% [markdown]
# A `Z` in the rank column means unranked, so we also drop those rows as they don't provide value.

# %%
df['rank'].value_counts()


# %%
df = df[df['rank'] != 'z']

# %% [markdown]
# First, I try to map each rank into a numerical value. (This isn't the best solution as later on it shows that there are too many categories to predict and too few data)

# %%
ranks = {'d': 1, 'd+': 2, 'c-': 3, 'c': 4, 'c+': 5, 'b-': 6, 'b': 7, 'b+': 8, 'a-': 9, 'a':10, 'a+': 11, 's-': 12, 's': 13, 's+': 14, 'ss': 15, 'u': 16, 'x': 17}
df['rank_num'] = df['rank'].map(ranks)
df

# %% [markdown]
# The data is currently organized in descending order by rank. As this will cause problems when I later go to split the data into training and test sets, I shuffle the data around.

# %%
from sklearn.utils import shuffle
df_shuffled = shuffle(df, random_state=1)
df_shuffled = df_shuffled.reset_index(drop=True)
df_shuffled

# %% [markdown]
# Here I get the data that can possibly be used in training a model

# %%
main_df = df_shuffled.loc[:,"gamesplayed":"vs"]
main_df["rank_num"] = df_shuffled["rank_num"]
main_df


# %%
corr_mat = main_df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_mat, annot=True)

# %% [markdown]
# We can't use rating, glicko, rd, since they directly determine rank.
# 
# ---
# 
# 

# %%
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(main_df.rank_num, main_df.apm)
ax.set_xlabel("Numerical Rank")
ax.set_ylabel("Attack Per Minute")
ax.set_title("Rank vs APM")


# %%
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(main_df.rank_num, main_df.pps)
ax.set_xlabel("Numerical Rank")
ax.set_ylabel("Pieces Per Second")
ax.set_title("Rank vs PPS")

# %% [markdown]
# We can see servere outliers in the above 5 pps yet d-tier range

# %%
main_df[main_df.pps > 5]


# %%
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(main_df.rank_num, main_df.vs)
ax.set_xlabel("Numerical Rank")
ax.set_ylabel("VS Score")
ax.set_title("Rank vs VS")

# %% [markdown]
# Remove the pps outliers

# %%
main_df = main_df[main_df.pps < 5]


# %%
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(main_df.rank_num, main_df.pps)
ax.set_xlabel("Numerical Rank")
ax.set_ylabel("Pieces Per Second")
ax.set_title("Rank vs PPS")

# %% [markdown]
# Split the data into X & y for what I use to predict and what I want to predict.

# %%
df_x = main_df.loc[:,['apm', 'pps', 'vs']]
df_x


# %%
df_y = main_df['rank_num']
df_y

# %% [markdown]
# Split the data into train and test sets

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.7, random_state=1)


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# Use the data to train 3 different classification models and see how they perform

# %%
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):

  np.random.seed(42)

  model_scores = {}

  for name, model in models.items():

    model.fit(X_train, y_train)
    model_scores[name] = model.score(X_test, y_test)

  return model_scores


# %%
model_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
model_scores


# %%
scores_df = pd.DataFrame(model_scores, index=["Accuracy"])
scores_df.T.plot.bar()

# %% [markdown]
# As you can see, it's not doing too well as the highest accuracy is only 43%. I try to tune some hyperparameters to see if there is any change even though I don't expect much.

# %%
rf_grid = {"n_estimators": np.arange(10,1000,50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2,20,2),
           "min_samples_leaf": np.arange(1,20,2)}

# Tune RFC with RandomizedSearchCV

np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)

rs_rf.fit(X_train, y_train)


# %%
rs_rf.best_params_


# %%
rs_rf.score(X_test, y_test)


# %%
model = RandomForestClassifier(max_depth=10, min_samples_leaf=9, min_samples_split=16, n_estimators=260)

# %% [markdown]
# In the end, it didn't improve too much. The model couldn't predict the specific ranks correctly most of the time due to many categories to predict as well as rather little distinction between some of the categories. For example, A-, A, and A+ are technically all pretty similar and belong to the same class. By this logic, let's generalize the ranks and merge some of them together. With this, I shrunk the number of categories from 17 to 8, halving the amount and most likely going to provide a significant increase in prediction accuracy.
# %% [markdown]
# I do everything like before to pre-process the data

# %%
ranks_grouped = {'d': 1, 'd+': 1, 'c-': 2, 'c': 2, 'c+': 2, 'b-': 3, 'b': 3, 'b+': 3, 'a-': 4, 'a':4, 'a+': 4, 's-': 5, 's': 5, 's+': 5, 'ss': 6, 'u': 7, 'x': 8}
df['rank_grouped'] = df['rank'].map(ranks_grouped)
df


# %%
df_shuffled = shuffle(df, random_state=1)
df_shuffled = df_shuffled.reset_index(drop=True)


# %%
main_df = df_shuffled.loc[:,"gamesplayed":"vs"]
main_df["rank_grouped"] = df_shuffled["rank_grouped"]
main_df


# %%
df_x = main_df.loc[:,['apm', 'pps', 'vs']]
df_x


# %%
df_y = main_df.loc[:,"rank_grouped"]
df_y


# %%
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.7, random_state=1)


# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# %%
fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# %% [markdown]
# A significant increase in prediction accuracy from 43% up to 74%. I yet again tune hyperparameters to see if I can get slightly better results.

# %%
rf_grid = {"n_estimators": np.arange(10,1000,50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2,20,2),
           "min_samples_leaf": np.arange(1,20,2)}

#Tune RFC with RandomizedSearchCV

np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20, verbose=True)

rs_rf.fit(X_train, y_train)


# %%
rs_rf.best_params_

# %% [markdown]
# These parameters are what I got, and they may change as I use RandomizedSearchCV which does not check all possibilities. If I wanted to get the best possible model, GridSearchCV should be used.

# %%
model = RandomForestClassifier(max_depth=None, min_samples_leaf=5, min_samples_split=12, n_estimators=310)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)


# %%
model.score(X_test, y_test)

# %% [markdown]
# We got the final predicted accuracy of 76%
# %% [markdown]
# I wanted to see the deviation in the predicted values versus the actual values in terms of the prediction frequency. From the graph below, the predictions are on par with the actual values.

# %%
fig, ax = plt.subplots(figsize=(10,5))
plt.hist(y_preds, color="purple")
plt.hist(y_test, color="green")

# %% [markdown]
# Just out of curiosity sake, I look to see some of the wrong predictions and see their differences. 

# %%
compare = pd.DataFrame([y_preds, y_test], index=["Predicted", "Actual"]).T
compare


# %%
compare[compare.Predicted != compare.Actual]

# %% [markdown]
# As I expected, the wrong predictions are only off by 1 rank. 
# %% [markdown]
# ### Afterthought
# 
# There was not too much data to begin with as Tetr.io is a rather small platform, so for what it is, the model did pretty decently. From the diagrams above, we could see that there wasn't a clear distinction between the ranks and there was a lot of overlap which directly affects prediction accuracy. I don't think the model could have done much better at its current state as the players in the data do not progress linearly in ranks aside their other statistics. Overall, 76% accuracy is pretty good considering all the problems that exist in this data. 

