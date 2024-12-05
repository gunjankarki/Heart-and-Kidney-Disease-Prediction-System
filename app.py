from flask import Flask, render_template, jsonify, request,flash,redirect
#from model import predict_image
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/services',methods=['GET', 'POST'])
def service():
    return render_template('page.html')
import pandas as pd
import numpy as np

def predict2(values, dic):
    model = pd.read_pickle('models/kidney.pkl')
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

@app.route("/ckd")
def ckd():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/working")
def workingPage():
    return render_template('working.html')

@app.route("/graph")
def graph():
    return render_template('graph.html')


def prediction(l):
    import pickle

    # Load the model from the pickle file
    filename = 'models/my_classifier.pkl'
    with open(filename, 'rb') as file:
        clf = pickle.load(file)

    # Define your single data point as a list
    single_data_point = l # 

    # Now you can use the loaded model to make predictions on this single data point
    prediction = clf.predict([single_data_point])
    probabilities = clf.predict_proba([single_data_point])

    # Print or use the prediction and probabilities as needed
    return max(prediction),max(probabilities)



@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():

    if request.method == 'POST':
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = float(request.form['rbc'])
        pc = float(request.form['pc'])
        pcc = float(request.form['pcc'])
        ba = float(request.form['ba'])
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        pot = float(request.form['pot'])
        wc = float(request.form['wc'])
        htn = float(request.form['htn'])
        dm = float(request.form['dm'])
        cad = float(request.form['cad'])
        pe = float(request.form['pe'])
        ane = float(request.form['ane'])
        sg = float(request.form['sg'])
        sodium = float(request.form['sodium'])
        hg = float(request.form['hg'])
        pcv = float(request.form['pcv'])
        rbc_count = float(request.form['rbcc'])
        appetite = float(request.form['apetite'])
        data = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,sodium, pot,hg,pcv, wc,rbc_count, htn, dm, cad,appetite, pe, ane]
        print(data)
        pred,prob=prediction(data)
    

    return render_template('predict.html', pred = pred,prob=prob,data=data)

import pandas as pd
import joblib

# Load your trained model
model_rf = joblib.load('heart_disease.pkl')

@app.route('/heartt')
def heartt():
    return render_template('heart.html')

@app.route('/predicth', methods=['POST'])
def predicth():
    # Get user inputs from the forml
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(request.form[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    # Make a prediction using the trained model
    pred_user = model_rf.predict(user_DF)

    # Display the result on the result.html page
    return render_template('result.html', result=pred_user[0])


@app.route('/accuracy')
def accuracy():
    accuracy_data = {
        "labels": ["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "XGB"],
        "testing_accuracy": [83.07, 91.53, 96.83, 98.41, 98.41],
        "training_accuracy": [86.19, 96.68, 100.0, 100.0, 100.0]
    }

    return render_template('accuracy.html', accuracyData=accuracy_data)




if __name__ == "__main__":
    app.run(debug=True)
