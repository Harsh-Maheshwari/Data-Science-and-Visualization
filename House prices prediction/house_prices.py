# Import Libraries
import numpy as np
import pandas as pd
# Data Vis
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
# Stat
from scipy import stats
from scipy.stats import norm
# Outlier Detection
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, accuracy_score
# memory management
import gc 
# utilities
from itertools import chain 
# ML
import lightgbm as lgb
import xgboost as xgb
# We can also use whitegrid style for our seaborn plots.( most basic one )
sns.set(style='white', context='notebook', palette='deep') 

class FeatureSelector():
	def __init__(self, data, labels=None):
		
		# Dataset and optional training labels
		self.data = data
		self.labels = labels

		if labels is None:
			print('No labels provided. Feature importance based methods are not available.')
		
		self.base_features = list(data.columns)
		self.one_hot_features = None
		
		# Dataframes recording information about features to remove
		self.record_missing = None
		self.record_single_unique = None
		self.record_collinear = None
		self.record_zero_importance = None
		self.record_low_importance = None
		
		self.missing_stats = None
		self.unique_stats = None
		self.corr_matrix = None
		self.feature_importances = None
		
		# Dictionary to hold removal operations
		self.ops = {}
		
		self.one_hot_correlated = False
		
	def identify_missing(self, missing_threshold):
		"""Find the features with a fraction of missing values above `missing_threshold`"""
		
		self.missing_threshold = missing_threshold

		# Calculate the fraction of missing in each column 
		missing_series = self.data.isnull().sum() / self.data.shape[0]
		self.missing_stats = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})

		# Sort with highest number of missing values on top
		self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending = False)

		# Find the columns with a missing percentage above the threshold
		record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = 
																											   {'index': 'feature', 
																												0: 'missing_fraction'})

		to_drop = list(record_missing['feature'])

		self.record_missing = record_missing
		self.ops['missing'] = to_drop
		
		print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.missing_threshold))
		
	def identify_single_unique(self):
		"""Finds features with only a single unique value. NaNs do not count as a unique value. """

		# Calculate the unique counts in each column
		unique_counts = self.data.nunique()
		self.unique_stats = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'nunique'})
		self.unique_stats = self.unique_stats.sort_values('nunique', ascending = True)
		
		# Find the columns with only one unique count
		record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns = {'index': 'feature', 
																												0: 'nunique'})

		to_drop = list(record_single_unique['feature'])
	
		self.record_single_unique = record_single_unique
		self.ops['single_unique'] = to_drop
		
		print('%d features with a single unique value.\n' % len(self.ops['single_unique']))
	
	def identify_collinear(self, correlation_threshold, one_hot=False):
		
		self.correlation_threshold = correlation_threshold
		self.one_hot_correlated = one_hot
		
		 # Calculate the correlations between every column
		if one_hot:
			
			# One hot encoding
			features = pd.get_dummies(self.data)
			self.one_hot_features = [column for column in features.columns if column not in self.base_features]

			# Add one hot encoded data to original data
			self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)
			
			corr_matrix = pd.get_dummies(features).corr()

		else:
			corr_matrix = self.data.corr()
		
		self.corr_matrix = corr_matrix
	
		# Extract the upper triangle of the correlation matrix
		upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
		
		# Select the features with correlations above the threshold
		# Need to use the absolute value
		to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

		# Dataframe to hold correlated pairs
		record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

		# Iterate through the columns to drop to record pairs of correlated features
		for column in to_drop:

			# Find the correlated features
			corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

			# Find the correlated values
			corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
			drop_features = [column for _ in range(len(corr_features))]    

			# Record the information (need a temp df for now)
			temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
											 'corr_feature': corr_features,
											 'corr_value': corr_values})

			# Add to dataframe
			record_collinear = record_collinear.append(temp_df, ignore_index = True)

		self.record_collinear = record_collinear
		self.ops['collinear'] = to_drop
		
		print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.correlation_threshold))

	def identify_zero_importance(self, task, eval_metric=None, 
								 n_iterations=10, early_stopping = True):

		if early_stopping and eval_metric is None:
			raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
							 "l2" for regression.""")
			
		if self.labels is None:
			raise ValueError("No training labels provided.")
		
		# One hot encoding
		features = pd.get_dummies(self.data)
		self.one_hot_features = [column for column in features.columns if column not in self.base_features]

		# Add one hot encoded data to original data
		self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)

		# Extract feature names
		feature_names = list(features.columns)

		# Convert to np array
		features = np.array(features)
		labels = np.array(self.labels).reshape((-1, ))

		# Empty array for feature importances
		feature_importance_values = np.zeros(len(feature_names))
		
		print('Training Gradient Boosting Model\n')
		
		# Iterate through each fold
		for _ in range(n_iterations):

			if task == 'classification':
				model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)

			elif task == 'regression':
				model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)

			else:
				raise ValueError('Task must be either "classification" or "regression"')
				
			# If training using early stopping need a validation set
			if early_stopping:
				
				train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15)

				# Train the model with early stopping
				model.fit(train_features, train_labels, eval_metric = eval_metric,
						  eval_set = [(valid_features, valid_labels)],
						  early_stopping_rounds = 100, verbose = -1)
				
				# Clean up memory
				gc.enable()
				del train_features, train_labels, valid_features, valid_labels
				gc.collect()
				
			else:
				model.fit(features, labels)

			# Record the feature importances
			feature_importance_values += model.feature_importances_ / n_iterations

		feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

		# Sort features according to importance
		feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

		# Normalize the feature importances to add up to one
		feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
		feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

		# Extract the features with zero importance
		record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
		
		to_drop = list(record_zero_importance['feature'])

		self.feature_importances = feature_importances
		self.record_zero_importance = record_zero_importance
		self.ops['zero_importance'] = to_drop
		
		print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))
	
	def identify_low_importance(self, cumulative_importance):

		self.cumulative_importance = cumulative_importance
		
		# The feature importances need to be calculated before running
		if self.feature_importances is None:
			raise NotImplementedError("""Feature importances have not yet been determined. 
										 Call the `identify_zero_importance` method first.""")
			
		# Make sure most important features are on top
		self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

		# Identify the features not needed to reach the cumulative_importance
		record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]

		to_drop = list(record_low_importance['feature'])

		self.record_low_importance = record_low_importance
		self.ops['low_importance'] = to_drop
	
		print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (len(self.feature_importances) -
																			len(self.record_low_importance), self.cumulative_importance))
		print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
																							   self.cumulative_importance))
		
	def identify_all(self, selection_params):

		# Check for all required parameters
		for param in ['missing_threshold', 'correlation_threshold', 'eval_metric', 'task', 'cumulative_importance']:
			if param not in selection_params.keys():
				raise ValueError('%s is a required parameter for this method.' % param)
		
		# Implement each of the five methods
		self.identify_missing(selection_params['missing_threshold'])
		self.identify_single_unique()
		self.identify_collinear(selection_params['correlation_threshold'])
		self.identify_zero_importance(task = selection_params['task'], eval_metric = selection_params['eval_metric'])
		self.identify_low_importance(selection_params['cumulative_importance'])
		
		# Find the number of features identified to drop
		self.all_identified = set(list(chain(*list(self.ops.values()))))
		self.n_identified = len(self.all_identified)
		
		print('%d total features out of %d identified for removal after one-hot encoding.\n' % (self.n_identified, 
																								  self.data_all.shape[1]))
		
	def check_removal(self, keep_one_hot=True):
		
		"""Check the identified features before removal. Returns a list of the unique features identified."""
		
		self.all_identified = set(list(chain(*list(self.ops.values()))))
		print('Total of %d features identified for removal' % len(self.all_identified))
		
		if not keep_one_hot:
			if self.one_hot_features is None:
				print('Data has not been one-hot encoded')
			else:
				one_hot_to_remove = [x for x in self.one_hot_features if x not in self.all_identified]
				print('%d additional one-hot features can be removed' % len(one_hot_to_remove))
		
		return list(self.all_identified)
		
	
	def remove(self, methods, keep_one_hot = True):     
		
		features_to_drop = []
	  
		if methods == 'all':
			
			# Need to use one-hot encoded data as well
			data = self.data_all
										  
			print('{} methods have been run\n'.format(list(self.ops.keys())))
			
			# Find the unique features to drop
			features_to_drop = set(list(chain(*list(self.ops.values()))))
			
		else:
			# Need to use one-hot encoded data as well
			if 'zero_importance' in methods or 'low_importance' in methods or self.one_hot_correlated:
				data = self.data_all
				
			else:
				data = self.data
				
			# Iterate through the specified methods
			for method in methods:
				
				# Check to make sure the method has been run
				if method not in self.ops.keys():
					raise NotImplementedError('%s method has not been run' % method)
					
				# Append the features identified for removal
				else:
					features_to_drop.append(self.ops[method])
		
			# Find the unique features to drop
			features_to_drop = set(list(chain(*features_to_drop)))
			
		features_to_drop = list(features_to_drop)
			
		if not keep_one_hot:
			
			if self.one_hot_features is None:
				print('Data has not been one-hot encoded')
			else:
							 
				features_to_drop = list(set(features_to_drop) | set(self.one_hot_features))
	   
		# Remove the features and return the data
		data = data.drop(columns = features_to_drop)
		self.removed_features = features_to_drop
		
		if not keep_one_hot:
			print('Removed %d features including one-hot features.' % len(features_to_drop))
		else:
			print('Removed %d features.' % len(features_to_drop))
		
		return data
	
	def plot_missing(self):
		"""Histogram of missing fraction in each feature"""
		if self.record_missing is None:
			raise NotImplementedError("Missing values have not been calculated. Run `identify_missing`")
		
		self.reset_plot()
		
		# Histogram of missing values
		plt.style.use('seaborn-white')
		plt.figure(figsize = (7, 5))
		plt.hist(self.missing_stats['missing_fraction'], bins = np.linspace(0, 1, 11), edgecolor = 'k', color = 'red', linewidth = 1.5)
		plt.xticks(np.linspace(0, 1, 11));
		plt.xlabel('Missing Fraction', size = 14); plt.ylabel('Count of Features', size = 14); 
		plt.title("Fraction of Missing Values Histogram", size = 16);
		
	
	def plot_unique(self):
		"""Histogram of number of unique values in each feature"""
		if self.record_single_unique is None:
			raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')
		
		self.reset_plot()

		# Histogram of number of unique values
		self.unique_stats.plot.hist(edgecolor = 'k', figsize = (7, 5))
		plt.ylabel('Frequency', size = 14); plt.xlabel('Unique Values', size = 14); 
		plt.title('Number of Unique Values Histogram', size = 16);
		
	
	def plot_collinear(self, plot_all = False):
   
		if self.record_collinear is None:
			raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')
		
		if plot_all:
			corr_matrix_plot = self.corr_matrix
			title = 'All Correlations'
		
		else:
			# Identify the correlations that were above the threshold
			# columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
			corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])), 
													list(set(self.record_collinear['drop_feature']))]

			title = "Correlations Above Threshold"

	   
		f, ax = plt.subplots(figsize=(10, 8))
		
		# Diverging colormap
		cmap = sns.diverging_palette(220, 10, as_cmap=True)

		# Draw the heatmap with a color bar
		sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,
					linewidths=.25, cbar_kws={"shrink": 0.6})

		# Set the ylabels 
		ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
		ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));

		# Set the xlabels 
		ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
		ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
		plt.title(title, size = 14)
		
	def plot_feature_importances(self, plot_n = 15, threshold = None):

		if self.record_zero_importance is None:
			raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')
			
		# Need to adjust number of features if greater than the features in the data
		if plot_n > self.feature_importances.shape[0]:
			plot_n = self.feature_importances.shape[0] - 1

		self.reset_plot()
		
		# Make a horizontal bar chart of feature importances
		plt.figure(figsize = (10, 6))
		ax = plt.subplot()

		# Need to reverse the index to plot most important on top
		# There might be a more efficient method to accomplish this
		ax.barh(list(reversed(list(self.feature_importances.index[:plot_n]))), 
				self.feature_importances['normalized_importance'][:plot_n], 
				align = 'center', edgecolor = 'k')

		# Set the yticks and labels
		ax.set_yticks(list(reversed(list(self.feature_importances.index[:plot_n]))))
		ax.set_yticklabels(self.feature_importances['feature'][:plot_n], size = 12)

		# Plot labeling
		plt.xlabel('Normalized Importance', size = 16); plt.title('Feature Importances', size = 18)
		plt.show()

		# Cumulative importance plot
		plt.figure(figsize = (6, 4))
		plt.plot(list(range(1, len(self.feature_importances) + 1)), self.feature_importances['cumulative_importance'], 'r-')
		plt.xlabel('Number of Features', size = 14); plt.ylabel('Cumulative Importance', size = 14); 
		plt.title('Cumulative Feature Importance', size = 16);

		if threshold:

			# Index of minimum number of features needed for cumulative importance threshold
			# np.where returns the index so need to add 1 to have correct number
			importance_index = np.min(np.where(self.feature_importances['cumulative_importance'] > threshold))
			plt.vlines(x = importance_index + 1, ymin = 0, ymax = 1, linestyles='--', colors = 'blue')
			plt.show();

			print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

	def reset_plot(self):
		plt.rcParams = plt.rcParamsDefault


def rmse(y_true, y_pred):
	
	return np.sqrt(mean_squared_error(y_true, y_pred))

def glimpse(df, maxvals=10, maxlen=110):
	print('Shape: ', df.shape)
	print('')
	print(df.dtypes.value_counts())
	print('')
	def pad(y):
		max_len = max([len(x) for x in y])
		return [x.ljust(max_len) for x in y]
	
	# Column Name
	toprnt = pad(df.columns.tolist())
	
	# Column Type
	toprnt = pad([toprnt[i] + ' ' + str(df.iloc[:,i].dtype) for i in range(df.shape[1])])
	
	# Num NAs
	num_nas = [df.iloc[:,i].isnull().sum() for i in range(df.shape[1])]
	num_nas_ratio = [int(round(x*100/df.shape[0])) for x in num_nas]
	num_nas_str = [str(x) + ' (' + str(y) + '%)' for x,y in zip(num_nas, num_nas_ratio)]
	max_len = max([len(x) for x in num_nas_str])
	num_nas_str = [x.rjust(max_len) for x in num_nas_str]
	toprnt = [x + ' ' + y + ' NAs' for x,y in zip(toprnt, num_nas_str)]
	
	# Separator
	toprnt = [x + ' ; Unique ' for x in toprnt]
	
	# Values
	#toprnt = [toprnt[i] + ', '.join([str(y) for y in df.iloc[:min([maxvals,df.shape[0]]), i]]) for i in range(df.shape[1])]
	toprnt = [toprnt[i] + '('+str(df.iloc[:,i].nunique())+') : ' +str(df.iloc[:,i].unique()[:5]) for i in range(df.shape[1])]
	
	# Trim to maxlen
	toprnt = [x[:min(maxlen, len(x))] for x in toprnt]
	
	for x in toprnt:
		print(x)
	print(df.describe())
	print('Head')
	print(df.head())
	print('Tail')
	print(df.tail())

def missing(df):
	# Capture the necessary data
	variables = df.columns

	count = []

	for variable in variables:
		length = df[variable].count()
		count.append(length)

	count_pct = np.round(100 * pd.Series(count) / len(df), 2)
	count = pd.Series(count)

	missing = pd.DataFrame()
	missing['variables'] = variables
	missing['count'] = len(df) - count
	missing['count_pct'] = 100 - count_pct
	missing = missing[missing['count_pct'] > 0]
	missing.sort_values(by=['count_pct'], inplace=True)

	# #Plot number of available data per variable
	# plt.subplots(figsize=(15,6))

	# # Plots missing data in percentage
	# plt.subplot(1,2,1)
	# plt.barh(missing['variables'], missing['count_pct'])
	# plt.title('Count of missing  data in percent', fontsize=15)

	# # Plots total row number of missing data
	# plt.subplot(1,2,2)
	# plt.barh(missing['variables'], missing['count'])
	# plt.title('Count of missing data as total records', fontsize=15)

	# plt.show(block=False)
	# plt.pause(3)
	# plt.close()
	miss = pd.DataFrame(list(missing['variables']),columns=['Features'])
	miss.insert(1, "Percentage", list(missing['count_pct']), True)
	miss.insert(2, "Count", list(missing['count']), True)
	print(miss)

	return miss    

def cleanup(test_file,target,char_var,df_train,df_test = None):

	# Missing Values
	print('Missing Data In the Train File')
	miss_train = missing(df_train)
	nan_var = list(miss_train[miss_train.Percentage>=40].Features)

	if test_file==True:
		print('Missing Data In the Test File')
		miss_test = missing(df_test)
		nan_var_test = list(miss_test[miss_test.Percentage>=40].Features)
		nan_var.extend(nan_var_test)
		del nan_var_test

	nan_var = set(nan_var)
	
	# Missing Value Treatement
	df_train = df_train.dropna(how = 'all') # Dropping rows if all values in that row are missing
	df_train = df_train.drop(nan_var,axis = 1) # removing more than 40 % nan Columns
	df_train = df_train.interpolate(method ='linear', limit_direction ='forward') # Rest nan values are interpolated

	# from sklearn.preprocessing import Imputer
	# imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

	df_train = df_train.fillna(method='pad')
	print('After Clenup Missing data in train')
	_ = missing(df_train)

	if test_file==True:
		df_test = df_test.dropna(how = 'all') # Dropping rows if all values in that row are missing
		df_test = df_test.drop(nan_var,axis = 1) # removing more than 50 % nan Columns
		df_test = df_test.interpolate(method ='linear', limit_direction ='forward') # Rest nan values are interpolated
		df_test = df_test.fillna(method='pad')
		print('After Clenup Missing data in test')
		_ = missing(df_test)


	# Correct Data Types in Categorical and Numerical Features
	variables = df_train.columns
	
	num_var = df_train.select_dtypes(include=['float64','float32','int32','int64']).columns
	try:
		num_var = num_var.drop(char_var)
		print("char_var removed from num_var") 
	except:
		pass    
	cat_var = df_train.select_dtypes(include=['object','category']).columns
	try:
		cat_var = cat_var.drop(char_var)
		print("char_var removed from cat_var")
	except:
		 pass
	try:
		char_var = char_var.drop(target)
		print("target removed from char_var") 
	except:
		pass
	try:
		cat_var = cat_var.drop(target)
		print("target removed  from cat_var")
	except:
		pass
	try:
		num_var = num_var.drop(target)
		print("target removed  from num_var")
	except:
		pass
	char_var = pd.Index(char_var)
	print('Variables : '); print(variables.values)
	print('Numerical Features : ' ); print(num_var.values)
	print('Categorical Features : ' ); print(cat_var.values)
	print('Name Features : ' ); print(char_var.values)

	# Remove Name/Char Features and other unnecessary Features From DataSet Before EDA
	df_train=df_train.drop(char_var,axis=1)
	if test_file == True:
		df_test=df_test.drop(char_var,axis=1)

	return df_train,df_test,num_var,cat_var,char_var

def EDA(df_train,num_var,cat_var,char_var,target ,target_type='continuos'):


	if target_type == 'continuos':
		# Histogram of Target
		plt.figure(figsize=(10,6))
		sns.distplot(df_train[target], color='g', hist_kws={'alpha': 0.4}, fit=norm)
		plt.title('Histogram of %s' % target)
		plt.show(block=False)
		plt.pause(3)
		plt.close()

	if target_type == 'discrete' :
		pass

	# Numerical Features
	
	if len(num_var)>0:
		## Histograms of Numerical Features with a Normal fit plot added
		f = pd.melt(df_train, value_vars=num_var)
		g = sns.FacetGrid(f, col="variable",  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(sns.distplot, "value" , fit=norm ,color='b',  kde_kws={'bw':0.1},hist_kws={'alpha': 0.4})
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		## Scatterplot of Numerical Features against Target
		f = pd.melt(df_train, id_vars=[target], value_vars=num_var) 
		g = sns.FacetGrid(f, col="variable",  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(sns.regplot, "value", target,color='g')
		plt.show(block=False)
		plt.pause(7)
		plt.close()

	# Categorical Features

	if len(cat_var)>0:
		## Countplots of Categorical Features (Use hue = target , if target is not continuous)
		def countplot(x, **kwargs):
			sns.countplot(x=x)
			x=plt.xticks(rotation=90)
		f = pd.melt(df_train, value_vars=cat_var)
		g = sns.FacetGrid(f, col='variable',col_wrap=6, sharex=False, sharey=False, height=5) # hue = target
		g = g.map(countplot, 'value' )
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		## Box-whisker Plots of Categorical Features against Target  (Possible only if target is continuous)
		def boxplot(x, y, **kwargs):
			sns.boxplot(x=x, y=y)
			x=plt.xticks(rotation=90)
		f = pd.melt(df_train, id_vars=[target], value_vars=cat_var)
		g = sns.FacetGrid(f, col='variable',  col_wrap=6, sharex=False, sharey=False, height=5)
		g = g.map(boxplot, 'value', target)
		plt.show(block=False)
		plt.pause(7)
		plt.close()

		# # Combine Violin Plot & Swarm Plot
		# def swarmviolin(x, y, **kwargs):
		# 	sns.violinplot(x=x, y=y)
		# 	sns.swarmplot(x=x, y=y, color = 'k', alpha = 0.6)
		# 	x=plt.xticks(rotation=90)
		# f = pd.melt(df_train, id_vars=[target], value_vars=cat_var)
		# g = sns.FacetGrid(f, col='variable',  col_wrap=6, sharex=False, sharey=False, height=5)
		# g = g.map(swarmviolin, 'value', target)
		# plt.show(block=False)
		# plt.pause(7)
		# plt.close()

	# Correlation Matrix
	corr = df_train.corr()
	mask = np.zeros_like(corr, dtype=np.bool) # Generate a mask for the upper triangle
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(11, 9)) # Set up the matplotlib figure
	cmap = sns.diverging_palette(220, 10, as_cmap=True) # Generate a custom diverging colormap
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=False, cmap=cmap) 
	plt.yticks(rotation=0)
	plt.title('Correlation Matrix of all Numerical Variables')
	plt.show(block=False)
	plt.pause(7)
	plt.close()
	
	print('Features with absolute correlation values greater than 0.5 are :')
	print(list(corr[(corr >= 0.5) | (corr <= -0.5)].index))
	ax = plt.subplots(figsize=(11, 9))
	sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4) ], cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
				annot=True, annot_kws={"size": 6}, square=False,cbar=True );

	# Correlation with respect to target
	df_corr = pd.DataFrame(corr.nlargest(corr.shape[1],target)[target])
	df_corr = df_corr[(df_corr>=0.5)|(df_corr<=-0.5)].dropna(how = 'all')

	Golden_Features = list(df_corr.index)
	print('Features with absolute correlation values greater than 0.5 wrt Target are : ')
	print(Golden_Features)

	# # PairPlots  
	# sns.set()
	# sns.pairplot(df_train)
	# plt.show(block=False)
	# plt.pause(7)
	# plt.close()

	return Golden_Features, corr

def plot_out_liers(df,cur_var,target):

	plt.scatter(df[cur_var],df[target])
	plt.show(block=False)
	plt.pause(5)
	plt.close()

	scaler = MinMaxScaler(feature_range=(0, 1))
	df[[cur_var,target]] = scaler.fit_transform(df[[cur_var,target]])

	X1 = df[cur_var].values.reshape(-1,1)
	X2 = df[target].values.reshape(-1,1)

	X = np.concatenate((X1,X2),axis=1)
	random_state = np.random.RandomState(42)
	outliers_fraction = 0.05
	# Define seven outlier  tools detectionto be compared
	classifiers = {
			'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
			'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
			'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
			'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
			'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
			'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
			'Average KNN': KNN(method='mean',contamination=outliers_fraction)
	}

	xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))

	for i, (clf_name, clf) in enumerate(classifiers.items()):
		clf.fit(X)
		# predict raw anomaly score
		scores_pred = clf.decision_function(X) * -1
			
		# prediction of a datapoint category outlier or inlier
		y_pred = clf.predict(X)
		n_inliers = len(y_pred) - np.count_nonzero(y_pred)
		n_outliers = np.count_nonzero(y_pred == 1)
		plt.figure(figsize=(10, 10))
		
		# copy of dataframe
		dfx = df
		dfx['outlier'] = y_pred.tolist()
		
		# IX1 - inlier feature 1,  IX2 - inlier feature 2
		IX1 =  np.array(dfx[cur_var][dfx['outlier'] == 0]).reshape(-1,1)
		IX2 =  np.array(dfx[target][dfx['outlier'] == 0]).reshape(-1,1)
		
		# OX1 - outlier feature 1, OX2 - outlier feature 2
		OX1 =  dfx[cur_var][dfx['outlier'] == 1].values.reshape(-1,1)
		OX2 =  dfx[target][dfx['outlier'] == 1].values.reshape(-1,1)
			 
		print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
			
		# threshold value to consider a datapoint inlier or outlier
		threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
			
		# decision function calculates the raw anomaly score for every point
		Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
		Z = Z.reshape(xx.shape)
			  
		# fill blue map colormap from minimum anomaly score to threshold value
		plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
			
		# draw red contour line where anomaly score is equal to thresold
		a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
			
		# fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
		plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
			
		b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
		
		c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
		   
		plt.axis('tight')  
		
		# loc=2 is used for the top left corner 
		plt.legend(
			[a.collections[0], b,c],
			['learned decision function', 'inliers','outliers'],
			prop=matplotlib.font_manager.FontProperties(size=20),
			loc=2)
		  
		plt.xlim((0, 1))
		plt.ylim((0, 1))
		plt.title(clf_name)
		plt.show(block=False)
		plt.pause(5)
		plt.close()

def out_lier_score(df,target,num_var):

	scaler = MinMaxScaler(feature_range=(0, 1))
	df = scaler.fit_transform(df.loc[:,num_var],df[target])#.to_numpy()
	random_state = np.random.RandomState(42)
	outliers_fraction = 0.05

	X = df
	df_out_score = []
	# Define seven outlier  tools detectionto be compared
	classifiers = {
			'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
			'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
			'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
			'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
			'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
			'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
			'Average KNN': KNN(method='mean',contamination=outliers_fraction)
	}
	for i, (clf_name, clf) in enumerate(classifiers.items()):
		clf.fit(X)
		# predict raw anomaly score
		scores_pred = clf.decision_function(X) * -1	
		# prediction of a datapoint category outlier or inlier
		y_pred = clf.predict(X)
		df_out_score.append(y_pred.tolist())
		
	df_out_score = pd.DataFrame(df_out_score).T
	df_out_score.columns = list(classifiers.keys())
	return df_out_score

def run(test_file,val_size,n_jobs,target,cur_on,cur_var,char_var,path_train,Definite_vars_to_remove,selection_params,methods,path_test=None):

	# Import Data
	print('################## Importing Data ##################')
	df_train = pd.read_csv(path_train)
	df_train = df_train.replace(r'^\s+$', np.nan, regex=True) # Replacing empty spaces with Null values
	if len(Definite_vars_to_remove)>0:
		df_train=df_train.drop(Definite_vars_to_remove,axis =1)     
	if test_file == True:		
		df_test = pd.read_csv(path_test)
		# df_test[target] = int(0)
		df_test = df_test.replace(r'^\s+$', np.nan, regex=True) # Replacing empty spaces with Null values
		df_test=df_test.drop(Definite_vars_to_remove,axis =1)     
		# df = pd.concat([df_train, df_test])

	# Glimpse of Data
	print('################## Glimpse Data ##################')
	glimpse(df_train) 
	if test_file == True:
		glimpse(df_test)

	# Cleaning Data 
	print('################## Cleaning Data ##################')
	if test_file == True:
		df_train,df_test,num_var,cat_var,char_var = cleanup(test_file,target,char_var,df_train,df_test)
	else:
		df_train,df_test,num_var,cat_var,char_var = cleanup(test_file,target,char_var,df_train)

	# EDA
	print('################## Exploratory Data Analysis ##################')
	# Golden_Features, corr_matrix = EDA(df_train,num_var,cat_var,char_var,target ,target_type='continuos')

	# Outlier Detection
	print('################## Outlier Detection ##################')
	if cur_on:
		df_send = df_train.copy()
		plot_out_liers(df_send,cur_var,target)
	# df_send = df_train.copy()
	# df_out_score = out_lier_score(df_send,target,num_var)
	# print('No of Outliers : ' + str(np.sum(df_out_score.sum(axis=1)>=3)))
	# df_train = df_train.loc[df_out_score.sum(axis=1)<3,:]

	# Feature Engineering
	print('################## Feature Engineering ##################')
	y = df_train.iloc[:,df_train.columns==target]

	df_send = df_train.copy(deep=True)

	fs = FeatureSelector(data = df_send.iloc[:,df_send.columns!=target], labels = df_send.iloc[:,df_send.columns==target])
	fs.identify_all(selection_params = selection_params)

	methods_to_use = []
	for k,v in methods.items():
		if v:
			methods_to_use.append(k)
	x = fs.remove(methods = methods_to_use, keep_one_hot = False)

	df_train = df_train[x.columns]
	df_train['train'] = 1
		
	if test_file == True:
		df_test = df_test[x.columns]
		df_test['train'] = 0
		df = pd.concat([df_train, df_test])
		df = pd.get_dummies(df,drop_first=True)
		df_train = df[df['train']==1]
		df_test = df[df['train']==0]
		df_test = df_test.drop('train',axis=1)
	if test_file ==False:
		df_train = pd.get_dummies(df_train)

	df_train = df_train.drop('train',axis=1)

	print('Features Used are : ' + str(df_train.columns))

	X = df_train.iloc[:,df_train.columns!=target]
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = val_size, random_state = 10)

	scX = StandardScaler()
	X_train = scX.fit_transform(X_train)
	X_valid = scX.transform(X_valid)
	if test_file == True:
		X_test = scX.transform(df_test)

	if selection_params['task'] == 'classification':
		pass
	elif selection_params['task'] == 'regression':
		scy = StandardScaler()
		y_train = scy.fit_transform(y_train)
		y_valid = scy.transform(y_valid)

	# Building Model
	print('################## Building Model ##################')
	if selection_params['task'] == 'classification':
		clf = xgb.XGBClassifier(
						 colsample_bytree=0.2,
						 gamma=0.0,
						 learning_rate=0.01,
						 max_depth=4,
						 min_child_weight=1.5,
						 n_estimators=7200,                                                                  
						 reg_alpha=0.9,
						 reg_lambda=0.6,
						 subsample=0.2,
						 seed=42,
						 silent=1
						 )
		# Run KFold prediction on training set to get a rough idea of how well it does.
		kfold = KFold(n_splits=5)
		results = cross_val_score(clf, X, y.values.reshape(y.shape[0],), cv=kfold)
		print("XGBoost Accuracy score on Training set: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

		clf.fit(X_train, y_train.to_numpy().ravel())
		y_pred = clf.predict(X_train)
		print("XGBoost rmse_score on Training set: ", rmse(y_train, y_pred))
		y_pred = clf.predict(X_valid)
		print("XGBoost rmse_score on Validation set: ", rmse(y_valid, y_pred))
		if test_file == True:
			y_pred = clf.predict(X_test)
			print("Test set predictions saved in output")

	elif selection_params['task'] == 'regression':
		reg = xgb.XGBRegressor(
						 colsample_bytree=0.2,
						 gamma=0.0,
						 learning_rate=0.01,
						 max_depth=4,
						 min_child_weight=1.5,
						 n_estimators=7200,                                                                  
						 reg_alpha=0.9,
						 reg_lambda=0.6,
						 subsample=0.2,
						 seed=42,
						 silent=1
						 )
		# Run KFold prediction on training set to get a rough idea of how well it does.
		kfold = KFold(n_splits=5)
		results = cross_val_score(reg, X, y, cv=kfold)
		print("XGBoost Accuracy score on Training set: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
		reg.fit(X_train, y_train)
		y_pred = reg.predict(X_train)
		print("XGBoost rmse_score on Training set: ", rmse(y_train, y_pred))
		y_pred = reg.predict(X_valid)
		print("XGBoost rmse_score on Validation set: ", rmse(y_valid, y_pred))
		if test_file == True:
			y_pred = reg.predict(X_test)
			print("Test set predictions saved in output")


def main():
	val_size = .25
	n_jobs = 6

	test_file = True
	cur_on = False
	target = "SalePrice"
	cur_var = 'LotFrontage'
	char_var = ['Id']
	Definite_vars_to_remove = []

	path_train = 'data/train.csv'
	selection_params = {'missing_threshold': 0.5, 'correlation_threshold': 0.8, 'task': 'regression', 'eval_metric': 'auc', 'cumulative_importance': 0.999}
	methods = {'missing':True,'single_unique':True, 'collinear':True, 'zero_importance':True, 'low_importance':True}

	if test_file == True:
		path_test = 'data/test.csv'
		run(test_file,val_size,n_jobs,target,cur_on,cur_var,char_var,path_train,Definite_vars_to_remove,selection_params,methods,path_test)
	else:
		run(test_file,val_size,n_jobs,target,cur_on,cur_var,char_var,path_train,Definite_vars_to_remove,selection_params,methods)

main()
