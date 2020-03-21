import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

here = os.path.dirname(os.path.abspath(__file__))

path_to_data = os.path.join(here, '..', 'data', 'SMSSpamCollection')

df = pd.read_csv(path_to_data, sep='\t', names=['Status', 'Message'])

df.loc[df['Status'] == 'ham', 'Status'] = 1
df.loc[df['Status'] == 'spam', 'Status'] = 0

df_x = df['Message']
df_y = df['Status']

cv = CountVectorizer()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)

x_traincv = cv.fit_transform(x_train)

x_testcv = cv.transform(x_test)

mnb = MultinomialNB()

y_train = y_train.astype('int')

mnb.fit(x_traincv, y_train)

predictions = mnb.predict(x_testcv)

expected_values = np.array(y_test)

count = 0

for i in range(len(predictions)):
    if predictions[i] == expected_values[i]:
        count +=1

percentage_accuracy = count/len(predictions) * 100

print(f"""Our prediction accuracy is {round(percentage_accuracy,2)}%""")

