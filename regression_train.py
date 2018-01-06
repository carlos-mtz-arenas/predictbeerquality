import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# overall functions ::::::::::: START

def train_data(x, y, model):
  model.fit(x, y)
  # save the trainned model
  joblib.dump(model, './outputmodels/trained_model.pkl')

def test_data(x, y, x_test, y_test):
  mse = mean_absolute_error(y, model.predict(x))
  print 'Training set mean absolute error: %.4f' % mse
  mse = mean_absolute_error(y_test, model.predict(x_test))
  print 'Test set mean absolute error: %.4f' % mse

def delete_unnecessary_columns(df, columns):
  for column in columns:
    del df[column]

# overall functions ::::::::::: END


# meta data to be used!!
downloaded_file='./canadian-beers/data.csv'
y_vector = 'score'
unnecessary_columns=['rank', 'name', 'abv']
one_hot_encoding_cols=['style', 'ratings', 'brewery']

# read in data from CSV
pd_iterator = pd.read_csv(downloaded_file)

# if we are splitting the data, maybe this will be a textfilereader thing
if type(pd_iterator) is pd.io.parsers.TextFileReader:
  # basically parse it to the dataframe stuff instead of using the damn textfilereader object!
  df = pd.concat(pd_iterator, ignore_index=True)
else:
  df = pd_iterator

print 'loaded the data already'

delete_unnecessary_columns(df, unnecessary_columns)

# remove rows with missing values
df.dropna(axis=0,
  how='any',
  thresh=None,
  subset=None,
  inplace=True
)

# convert non-numerical data using one-hot encoding
#copy=df.copy()
#del copy[y_vector]
features_df = pd.get_dummies(df, columns=one_hot_encoding_cols)
#features_df=df.copy()
#print 'the actual features_df is'
#print features_df.columns.values

# remove price (as this will be the y axis <what we are going to calculate>)
del features_df[y_vector]

# create the x and y arrays from the dataset
x_axis = features_df.as_matrix()

y_axis = df[y_vector].as_matrix()

# split the data into test / train set (70 / 30 split) and shuffle
x_train, x_test, y_train, y_test=train_test_split(
  x_axis,
  y_axis,
  test_size=0.3,
  random_state=0
)

print 'shape of x_train is', df.shape
print 'features_df shape', features_df.shape

# setup algorithm - the regressor
model = ensemble.GradientBoostingRegressor(
  n_estimators=15,
  learning_rate=1,
  max_depth=30,
  min_samples_split=4,
  min_samples_leaf=6,
  max_features=0.6,
  loss='huber'
)

print model.get_params()

# run model on training data
train_data(x_axis, y_axis, model)

# test the data to check the accurracy
test_data(x_train, y_train, x_test, y_test)

