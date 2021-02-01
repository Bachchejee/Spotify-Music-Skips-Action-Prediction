#libraries and modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier

#Load Datasets
train = pd.read_csv("technocolabs training set.csv")
train.head()
#Describe
train.describe()
train.describe(include=['O'])

##Data-preprocessing and EDA Part

train.drop_duplicates(inplace=True)
train.dropna(inplace=True)
train.shape

for colname in ['skip_1','skip_2','skip_3','not_skipped','hist_user_behavior_is_shuffle','premium']:
    train[colname] = train[colname].astype(int, copy=False)
    train['skip'] = train['not_skipped'].replace({ 0 : 1, 1 : 0 })
    train['skip'].value_counts()
train['skip'].value_counts().plot(kind='pie', autopct = "%1.0f%%")

col = ['skip_1','skip_2','skip_3',
       'not_skipped','context_switch','no_pause_before_play',
       'short_pause_before_play','long_pause_before_play','hist_user_behavior_is_shuffle',
       'premium','context_type','hist_user_behavior_reason_start',
       'hist_user_behavior_reason_end']

plt.figure(figsize=(20,25))
n = 1
for colname in col:
    plt.subplot(5,3,n)
    train[colname].value_counts().plot(kind='bar')
    plt.xlabel(colname)
    n +=1

##hide unused columns
df = train.copy()
df.shape

df = df.drop(columns=['skip_1','skip_2','skip_3','not_skipped','date'])
df.shape

df.head()

#Train dataset
df1 = df.drop(['session_id', 'track_id_clean'], axis=1)
df1.shape
dummy_train = pd.get_dummies(df1)
dummy_train.shape

dummy_train = pd.get_dummies(df1)
dummy_train.shape

dummy_train.head()

#Track Features

tf = pd.read_csv('track_feats.csv')
tf.head()
tf2 = pd.read_csv('track_feats.csv')
tf2.head()

track = pd.concat([tf, tf2])
track.shape

#inspecting data
track.info()
track.duplicated().sum()
track.isna().sum()
track.describe()
track.describe(include='O')

#EDA Part

tf[[c for c in tf.columns if tf[c].dtype != 'float64']].head()

sns.distplot(track.release_year)
plt.title("Distribution of Release Years");
track['key'].unique()

keys = track.key.value_counts().sort_index()
sns.barplot(
    x=[ "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    y=keys.values/keys.sum()
)
plt.title("Distribution of Track Keys")
plt.xlabel("Key")

#Analysing numerical columns
track.hist(figsize=(20,15))
#Correlation matrix

plt.figure(figsize=(20,15))
sns.heatmap(tf.corr(), annot=True)

# Merge track and Training Data
track.shape
df.shape
df.rename(columns={'track_id_clean': 'track_id'}, inplace=True)
dummy_train.head(2)
dummy_train.shape


#Train Test Split

X = dummy_train.drop(columns=["skip"])
y = dummy_train.skip
X_resampled, Y_resampled = SMOTE(sampling_strategy=1.0, random_state=2).fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2)

#Standards for Data
scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)

# Applying Logistic Regression
log = LogisticRegressionCV( cv=3).fit(sX_train,y_train)
print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))


#Random Forest

rfc = RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))

#feature selection
from boruta import BorutaPy
rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(np.array(sX_train), np.array(y_train))

print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)

selected_rf_features = pd.DataFrame({'Feature':list(X.columns),'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')

X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))

#Creating model with important features 
rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)

print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test))

#Apply XGBoost
xg = xgb.XGBClassifier()
xg.fit(X_important_train, y_train, eval_set=[(X_important_train, y_train),(X_important_val, y_val)], early_stopping_rounds=10, verbose=True)

print("XGB Train score: %s" % xg.score(X_important_train,y_train))
print("XGB Val score:   %s" % xg.score(X_important_val,y_val))
print("XGB Test score:  %s" % xg.score(X_important_test,y_test))

#Train and test data
ts1 = pd.read_csv('test_data.csv')
ts2 = pd.read_csv('test_data_20.csv')
test_set = pd.concat([ts1,ts2])
test_set.shape

test_set.head()
test_set['skip'] =  test_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_test_data = test_set.skip

t1 = test_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean','hist_user_behavior_reason_end_appload'], axis=1)
t1.shape

#Load Validation data
vs1 = pd.read_csv('val_data.csv')
vs2 = pd.read_csv('val_data_20.csv')
val_set = pd.concat([vs1,vs2])
val_set.shape
val_set['skip'] =  val_set['not_skipped'].replace({ 0 : 1, 1 : 0 })
y_val_data = val_set.skip

v1 = val_set.drop(['skip_1','skip_2','skip', 'skip_3',	'not_skipped', 'session_id', 'track_id_clean', 'hist_user_behavior_reason_end_appload'],axis=1)
v1.shape

#train, Test and Validation dataset
X_val_set = boruta_selector.transform(np.array(v1))
X_test_set = boruta_selector.transform(np.array(t1))

#test score data
print("XGB Val score:   %s" % xg.score(X_val_set, y_val_data))
print("XGB Test score:  %s" % xg.score(X_test_set, y_test_data))

print("RF Val score:   %s" % rf_important.score(X_val_set, y_val_data))
print("RF Test score:  %s" % rf_important.score(X_test_set, y_test_data))

#Scaling Data
scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_val = scaler.transform(X_val)
sX_test = scaler.transform(X_test)


# Applying Logistic Regression
log = LogisticRegressionCV(cv=3).fit(sX_train,y_train)

print("Log Train score: %s" % log.score(sX_train,y_train))
print("Log Val score:   %s" % log.score(sX_val,y_val))
print("Log Test score:  %s" % log.score(sX_test,y_test))

#Apply Random forest Classifier

rfc = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
print("RFC Train score: %s" % rfc.score(X_train,y_train))
print("RFC Val score:   %s" % rfc.score(X_val,y_val))
print("RFC Test score:  %s" % rfc.score(X_test,y_test))

#Feature selction using Boruta
rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(np.array(sX_train), np.array(y_train))

print("Selected Features: ", boruta_selector.support_)    # check selected features
print("Ranking: ",boruta_selector.ranking_)               # check ranking of features
print("No. of significant features: ", boruta_selector.n_features_)

selected_rf_features = pd.DataFrame({'Feature':list(X.columns),'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')

X_important_train = boruta_selector.transform(np.array(X_train))
X_important_val = boruta_selector.transform(np.array(X_val))
X_important_test = boruta_selector.transform(np.array(X_test))

#Random Forest
rf_important = RandomForestClassifier(random_state=1, n_estimators=100, n_jobs = -1)
rf_important.fit(X_important_train, y_train)

print("RFC Train score: %s" % rf_important.score(X_important_train, y_train))
print("RFC Val score:   %s" % rf_important.score(X_important_val, y_val))
print("RFC Test score:  %s" % rf_important.score(X_important_test, y_test))

#LGBM
lgbm = LGBMClassifier( ).fit( X_important_train, y_train,
       eval_set=[(X_important_train, y_train),(X_important_val, y_val)],
       early_stopping_rounds=10, verbose=True)

print()
print("LGBM Train score: %s" % lgbm.score(X_important_train,y_train))
print("LGBM Val score:   %s" % lgbm.score(X_important_val,y_val))
