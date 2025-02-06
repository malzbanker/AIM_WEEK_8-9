# Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# 1. Handle Missing Values
def handle_missing(df):
    # Drop columns with >50% missing
    df = df.dropna(thresh=len(df)//2, axis=1)
    # Impute numerical with median
    num_imputer = SimpleImputer(strategy='median')
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    # Impute categorical with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    return df

# 2. Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    # Convert date columns
    date_cols = ['transaction_date', 'user_dob']  # Example columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    return df

# 3. EDA
def perform_eda(df):
    # Univariate analysis
    df.hist(figsize=(12,10))
    df.describe(include='all')
    
    # Bivariate analysis
    df.corr(numeric_only=True).style.background_gradient()
    pd.crosstab(df['device_type'], df['class']).plot(kind='bar')

# 4. Geolocation Merge
def convert_ip(ip):
    return int(sum(int(octet) << (24 - 8*i) for i, octet in enumerate(ip.split('.'))))

def merge_geo_data(fraud_df, ip_df):
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(convert_ip)
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_int'),
        ip_df.sort_values('ip_start'),
        left_on='ip_int',
        right_on='ip_start',
        direction='forward'
    )
    return merged_df

# 5. Feature Engineering
def create_features(df):
    # Transaction frequency
    df['txn_freq_1h'] = df.groupby('user_id')['transaction_id'].transform(
        lambda x: x.rolling('1H', on='transaction_date').count()
    )
    
    # Time-based features
    df['hour_of_day'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    return df

# 6. Normalization
def scale_features(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# 7. Encode Categorical
def encode_features(df):
    df = pd.get_dummies(df, columns=['device_type', 'browser'])
    return df