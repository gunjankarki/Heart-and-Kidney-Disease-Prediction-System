import pandas as pd

df=pd.read_csv("X.csv")

print(df.columns)
df.drop(columns=['Unnamed: 0', 'class'], inplace=True)

print(df.iloc[264,:])
def prediction(l):
    import pickle

    # Load the model from the pickle file
    filename = 'my_classifier.pkl'
    with open(filename, 'rb') as file:
        clf = pickle.load(file)

    # Define your single data point as a list
    single_data_point = l # 

    # Now you can use the loaded model to make predictions on this single data point
    prediction = clf.predict([single_data_point])
    probabilities = clf.predict_proba([single_data_point])

    # Print or use the prediction and probabilities as needed
    return max(prediction),max(probabilities)

l=[]
p,q=prediction(list(df.iloc[264,:]))
print(p)

#[66.0,70.0,1.020	0.0	0.0	1	1	0	0	94.0	19.0	0.7	135.0	3.9	16.0	41.0	5300.0	5.9	0	0	0	0	0	0	1]
