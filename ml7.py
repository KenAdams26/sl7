#7. Write a python program to make CATEGORICALvalues in numeric format for a given dataset.
# Defining sample Employee Data
import pandas as pd
EmployeeData=pd.DataFrame({'id': [101,102,103,104,105],
                        'Gender': ['M','M','M','F','F'],
                           'Age': [21,25,24,28,25],
                           'Department':['QA','QA','Dev','Dev','UI'],
                           'Rating':['A','B','B','C','B'] })
# Priting data
print(EmployeeData)
# Converting Ordinal Variable Rating to numeric
EmployeeData['Rating'].replace({'A':3, 'B':2, 'C':1}, inplace=True)
print(EmployeeData)
