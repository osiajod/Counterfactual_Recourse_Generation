#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def generate_data(generator, y_name, decider, n = 100):
  '''This function provides a general framework to generate artificial datasets.
  The user defines a 'generator' and 'decider' which generate an individual
  data point and then decide if the response for this variable is positive or
  negative (0/1). Generator should return a dictionary of individual attributes,
  and decider should accept the individual dictionary as a parameter and return
  a 0/1 based on the individual's attributes.
  '''
  data = []

  for i1 in range(n):
    datum = generator()

    datum[y_name] = decider(datum)

    data.append(datum)

  return pd.DataFrame(data)


# In[31]:


AGE_MIN = 20
AGE_MAX = 80

SEX = ['male', 'female']

# https://grants.nih.gov/grants/guide/notice-files/not-od-15-089.html
RACE = ['American Indian or Alaska Native',
         'Asian',
         'Black or African American',
         'Hispanic or Latino',
         'Native Hawaiian or Other Pacific Islander',
         'White']

# Generalize credit score to the five ranges
# 300-579: Poor.
# 580-669: Fair.
# 670-739: Good.
# 740-799: Very good.
# 800-850: Excellent.
CREDIT_SCORE = ['poor', 'fair', 'good', 'very good', 'excellent']

# Generalize education to three levels, much more complex in real world
EDUCATION = ['high school', 'college', 'graduate school']

# Generalize salary to three buckets
SALARY = ['low', 'medium', 'high']

# Generalize loan amnt to three buckets
LOAN_AMOUNT = ['low', 'medium', 'high']

def insurance_generator():
  '''Generate an individual included in the insurance dataset.'''
  return {
      'age': np.random.randint(AGE_MIN, AGE_MAX)/100.0,
      'sex': np.random.choice(SEX),
      'race': np.random.choice(RACE),
      'credit_score': np.random.choice(CREDIT_SCORE),
      'education': np.random.choice(EDUCATION),
      'salary': np.random.choice(range(100, 20000))/10000.0,
      'debt': np.random.choice(range(0, 15000))/10000.0,
      'married': round(np.random.random()), # 0 or 1
      'loan_amount': np.random.choice(LOAN_AMOUNT)
  }


# In[32]:


def random_decider(individual):
    # Totally random, 50% chance
    d2i = (individual['debt']/individual['salary'])
    if d2i >= 1.0 or individual['credit_score'] == 'poor' or (individual['education'] == 'high school' and individual['salary'] <= 0.1):
            return 0
    return 1
#return int(np.random.random() < 0.5)

generate_data(insurance_generator, 'loan_approved', random_decider, n = 10)


# In[33]:


np.random.choice(range(1000, 250000))


# In[30]:


0.161/1.10


# In[ ]:




