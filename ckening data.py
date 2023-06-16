#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_json("C:\\Users\\nor29\Desktop\\puppeeter proj\\laptops_data.json")

# Function to extract the desired value
def extract_value(input_string):
    match = re.search(r"\((\d+\.\d+)GHz\)", input_string)
    if match:
        return match.group(1)
    return None

# Apply the extraction function to the desired column
df['ProcessorSpeed'] = df['ProcessorSpeed'].apply(extract_value)
df['ProcessorSpeed'] = df['ProcessorSpeed'].astype(float)

df.replace("NaN", np.nan, inplace=True)
# Count non-null values in each column
value_counts = df.count()

# Print the DataFrame and value countsdf
print(value_counts)
df


# In[3]:


df = df.drop(['Type', 'Weight', 'BluetoothVersion', 'NumberOfCores'], axis=1)
df


# In[4]:


df = df.drop_duplicates()
df


# In[5]:


df = df.dropna(subset=['Price'])
nan_counts = df[['Price']].isna().sum()
print(nan_counts)


# In[6]:


df = df.dropna(thresh=df.shape[1] - 3)
df


# In[7]:


brand_mapping = {
    'Lenovo': 1,
    'MSI': 2,
    'HP': 3,
    'ASUS': 4,
    'DELL': 5,
    'LG Electronics': 6,
    'SAMSUNG': 7,
    'Acer America': 8,
    'IPASON': 9,
    'Microsoft': 10,
    'LincPlus': 11,
    'BMAX': 12,
    'iVIEW': 13,
    'CTL': 14,
    'DURABOOK/Twinhead': 15,
    'Gateway': 16,
    'geektech': 17,
    'Panasonic': 18,
    
    # Add more brand names and their corresponding integer values as needed
}
# Replace the values in 'Brand' column with integers using the mapping
df['Brand'] = df['Brand'].replace(brand_mapping)

df


# In[8]:


df = df[df['Price'] != '']
df


# In[9]:


def convert_price(price):
    if ',' in price:
        return float(price.replace(',', '').replace('$', ''))
    else:
        return float(price.replace('$', ''))

df.loc[:, 'Price'] = df['Price'].apply(convert_price)
df


# In[10]:


def convert_ram_size(ram):
    if pd.isnull(ram):
        return None
    numeric_part = re.findall(r'\d+', str(ram))
    if numeric_part:
        return int(numeric_part[0])
    else:
        return None

# Applying conversion function to 'RAMSize' column
df['RAMSize'] = df['RAMSize'].apply(convert_ram_size).astype('Int64')

df['RAMSize'] = df['RAMSize'].astype(str)

# Remove any non-digit characters from the values
df['RAMSize'] = df['RAMSize'].str.replace('\D', '', regex=True)

# Convert values in 'RAMSize' column to integers
df['RAMSize'] = df['RAMSize'].apply(lambda x: int(x) if x else 0)
df


# In[11]:


def convert_screen_size(screen):
    if isinstance(screen, float) or screen is None:
        return screen
    else:
        numeric_part = re.findall(r'\d+\.\d+', str(screen))
        if numeric_part:
            return float(numeric_part[0])
        else:
            return None

# Applying conversion function to 'ScreenSize' column
df['ScreenSize'] = df['ScreenSize'].apply(convert_screen_size)
df


# In[12]:


# Extracting the Length and Width from the Resolution column
df[['Length', 'Width']] = df['Resolution'].str.extract(r'(\d+) x (\d+)', expand=True)

# Converting the Length and Width columns to integers
df['Length'] = df['Length'].fillna(0).astype(int)
df['Width'] = df['Width'].fillna(0).astype(int)

df


# In[13]:


df = df.drop('Resolution', axis=1)


# In[14]:


def convert_ssd(ssd):
    if pd.isna(ssd) or ssd == 'No':
        return 0
    elif 'TB' in ssd:
        numeric_part = ssd.split()[0]
        if numeric_part.isdigit():
            return int(numeric_part) * 1000
        else:
            return 0
    elif 'GB' in ssd:
        numeric_part = ssd.split()[0]
        if numeric_part.isdigit():
            return int(numeric_part)
        else:
            return 0
    else:
        return 0

# Convert SSD values in the DataFrame to the desired form
df['SSD'] = df['SSD'].apply(convert_ssd)


# In[15]:


nan_counts = df.isna().sum()

# Print the NaN counts for each column
for column, count in nan_counts.items():
    print(f"Column '{column}': {count} NaN values")
    


# In[16]:


def is_number(value):
    return isinstance(value, int) or isinstance(value, float)

# Check if all values in each column are integers or floats
is_number_columns = df.applymap(is_number)

# Check if all values in each column are integers or floats
for column in is_number_columns.columns:
    if is_number_columns[column].all():
        print(f"All values in column '{column}' are integers or floats")
    else:
        print(f"Not all values in column '{column}' are integers or floats")


# In[17]:


unique_values = {}
for column in df.columns:
    unique_values[column] = df[column].unique()

# Print the unique values for each column
for column, values in unique_values.items():
    print(f"Column '{column}':")
    for value in values:
        print(value)


# In[18]:


# Fill NaN values in 'ProcessorSpeed' column with the median
processor_speed_median = df['ProcessorSpeed'].median()
df['ProcessorSpeed'] = df['ProcessorSpeed'].fillna(processor_speed_median)

# Fill NaN values in 'ScreenSize' column with the median
screen_size_median = df['ScreenSize'].median()
df['ScreenSize'] = df['ScreenSize'].fillna(screen_size_median)



# Converting the Length and Width columns to float
df['Length'] = df['Length'].astype(float)
df['Width'] = df['Width'].astype(float)

# Fill NaN values in 'Length' column with the median
length_median = df['Length'].median()
df['Length'] = df['Length'].fillna(length_median)

# Fill NaN values in 'Width' column with the median
width_median = df['Width'].median()
df['Width'] = df['Width'].fillna(width_median)


df


# In[19]:



# Print the maximum values in the RAMSize and SSD columns
print("Maximum RAMSize:", df['RAMSize'].max())
print("Maximum SSD:", df['SSD'].max())


# In[20]:


# Define the outliers criteria
ramsize_outlier_threshold = 300
ssd_outlier_threshold = 5000

# Calculate the replacement values
ramsize_replacement = df['RAMSize'].median()
ssd_replacement = df['SSD'].median()

# Replace values greater than the RAMSize threshold with the replacement value
df.loc[df['RAMSize'] > ramsize_outlier_threshold, 'RAMSize'] = ramsize_replacement

# Replace values greater than the SSD threshold with the replacement value
df.loc[df['SSD'] > ssd_outlier_threshold, 'SSD'] = ssd_replacement

# Verify the changes
ramsize_outliers = df[df['RAMSize'] > ramsize_outlier_threshold]
num_ramsize_outliers = len(ramsize_outliers)
print(f"Number of RAMSize outliers after treatment: {num_ramsize_outliers}")
print(ramsize_outliers)

ssd_outliers = df[df['SSD'] > ssd_outlier_threshold]
num_ssd_outliers = len(ssd_outliers)
print(f"Number of SSD outliers after treatment: {num_ssd_outliers}")
print(ssd_outliers)


# In[21]:


# Print the maximum values in the RAMSize and SSD columns
print("Maximum RAMSize:", df['RAMSize'].max())
print("Maximum SSD:", df['SSD'].max())


# In[22]:


# Print the number of zero values before replacement
num_ramsize_zeros_before = df[df['RAMSize'] == 0]['RAMSize'].count()
num_ssd_zeros_before = df[df['SSD'] == 0]['SSD'].count()
print("Number of zero values in RAMSize (before):", num_ramsize_zeros_before)
print("Number of zero values in SSD (before):", num_ssd_zeros_before)


# In[23]:


# Compute the descriptive statistics of the data
data_stats = df.describe()

# Check the skewness of the data
skewness = df.skew()

# Check for outliers using the interquartile range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

# Print the descriptive statistics, skewness, and outlier count
print("Descriptive Statistics:")
print(data_stats)
print("\nSkewness:")
print(skewness)
print("\nOutlier Count:")
print(outliers)


# In[24]:


# Replace zero values in RAMSize column with the median value
ramsize_median = df[df['RAMSize'] != 0]['RAMSize'].median()
df.loc[df['RAMSize'] == 0, 'RAMSize'] = ramsize_median

# Replace zero values in SSD column with the median value
ssd_median = df[df['SSD'] != 0]['SSD'].median()
df.loc[df['SSD'] == 0, 'SSD'] = ssd_median


# In[25]:


num_zero_ramsize = len(df[df['RAMSize'] == 0])
num_zero_ssd = len(df[df['SSD'] == 0])

print(f"Number of zero values in RAMSize: {num_zero_ramsize}")
print(f"Number of zero values in SSD: {num_zero_ssd}")


# In[26]:


# Replace zero values in 'Width' column with the median value
Width_median = df[df['Width'] != 0]['Width'].median()
df.loc[df['Width'] == 0, 'Width'] = Width_median

# Replace zero values in 'Length' column with the median value
Length_median = df[df['Length'] != 0]['Length'].median()
df.loc[df['Length'] == 0, 'Length'] = Length_median


# In[27]:


num_zero_Width = len(df[df['Width'] == 0])
num_zero_Length = len(df[df['Length'] == 0])

print(f"Number of zero values in Width: {num_zero_Width}")
print(f"Number of zero values in Length: {num_zero_Length}")


# In[28]:


# Count the number of zero values in each column
zero_counts = df.eq(0).sum()

# Print the zero counts for each column
print("Zero Counts:")
print(zero_counts)


# In[29]:


df.to_csv('ניקוי מעודכן סופי', index=False)


# In[30]:


df.info()


# In[31]:





# In[32]:


df


# In[ ]:




