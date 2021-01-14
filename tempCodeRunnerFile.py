import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

def load_model():
   return pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    item = [x for x in request.form.values()]
    data = []

    ##ApplicantIncome
    data.append(int(item[0]))
    ##CoApplicantIncome
    data.append(int(item[1]))
    ##LoanAmount
    data.append(int(item[2]))
    ##LoanAmountTerm
    data.append(int(item[3]))
    ##Credit_History
    data.append(int(item[4]))

    ##Gender
    if item[5] == 'Male':
        data.append(0)
        data.append(1)
    else:
        data.append(1)
        data.append(0)
    ##Married
    if item[6] == 'No':
        data.append(1)
        data.append(0)
    else:
        data.append(0)
        data.append(1)
    ##Dependents
    if item[7] == '0':
        data.append(1)
        data.append(0)
        data.append(0)
        data.append(0)
    elif item[7] == '1':
        data.append(0)
        data.append(1)
        data.append(0)
        data.append(0)
    elif item[7] == '2':
        data.append(0)
        data.append(0)
        data.append(1)
        data.append(0)
    else:
        data.append(0)
        data.append(0)
        data.append(0)
        data.append(1)
    ##Graduation
    if item[8] == 'Graduate':
        data.append(1)
        data.append(0)
    else:
        data.append(0)
        data.append(1)

    ##Self_Employed
    if item[9] == 'No':
        data.append(1)
        data.append(0)
    else:
        data.append(0) 
        data.append(1)

    ##Property
    if item[10] == 'Rural':
        data.append(1)
        data.append(0)
        data.append(0)
    elif item[10] == 'Semiurban':
        data.append(0)
        data.append(1)
        data.append(0)
    else :
        data.append(0)
        data.append(0)
        data.append(1)

       
    model = load_model()
    prediction = model.predict([data])

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The Loan Status will be {}'.format(output))


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)