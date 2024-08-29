# For data manipulation
import pandas as pd


# For hypothesis testing
# from scipy.stats import ttest_1samp

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("winequality-red.csv")

# Mapping from 3-8 to 0-5
mapping = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}
# Apply the mapping to the 'target' column
df['quality'] = df['quality'].map(mapping)

# Drop duplicates and save resulting dataframe in a new variable as needed
df = df.drop_duplicates(keep='first')

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['pH'].quantile(0.25)
Q3 = df['pH'].quantile(0.75)

# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1

# Define the acceptable range (fence)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_filtered = df[(df['pH'] >= lower_bound) & (df['pH'] <= upper_bound)]

df = df_filtered

# features (X) and target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('quality', axis=1))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Create the SMOTE object
smote = SMOTE(
    sampling_strategy='auto',  # Strategy to sample the minority class. 'auto' means resample all minority classes.
    random_state=42,           # Seed for random number generation.
    k_neighbors=5,             # Number of nearest neighbors to use for generating synthetic samples.
    n_jobs=None                # Number of CPU cores to use (-1 means use all available cores).
)

# Apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,  stratify=y_resampled)



# Let's use the model that performs best for us
def prediction(X_test):
    model = XGBClassifier(num_class=6, use_label_encoder=False, eval_metric='mlogloss')
    xgb_cv = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    right_key = None
    for key, val in mapping.items():
        print("Val", val)
        if val == y_pred[0]:
            right_key = key
            print("Prediction", y_pred[0], right_key)
            break
        
        
    return right_key
    