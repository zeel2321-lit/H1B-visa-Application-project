from flask import Flask, request, render_template
import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np

model = pickle.load(open('dtmodel.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    AGENT_REPRESENTING_EMPLOYER = request.form['AGENT_REPRESENTING_EMPLOYER']
    CONTINUED_EMPLOYMENT = request.form['CONTINUED_EMPLOYMENT']
    CHANGE_PREVIOUS_EMPLOYMENT = request.form['CHANGE_PREVIOUS_EMPLOYMENT']
    NEW_CONCURRENT_EMPLOYMENT = request.form['NEW_CONCURRENT_EMPLOYMENT']
    CHANGE_EMPLOYER = request.form['CHANGE_EMPLOYER']
    AMENDED_PETITION = request.form['AMENDED_PETITION']
    H1B_DEPENDENT = request.form['H1B_DEPENDENT']
    SUPPORT_H1B = request.form['SUPPORT_H1B']
    WILLFUL_VIOLATOR = request.form['WILLFUL_VIOLATOR']
    arr = np.array([[AGENT_REPRESENTING_EMPLOYER, CONTINUED_EMPLOYMENT, CHANGE_PREVIOUS_EMPLOYMENT, NEW_CONCURRENT_EMPLOYMENT, CHANGE_EMPLOYER, AMENDED_PETITION, H1B_DEPENDENT, SUPPORT_H1B, WILLFUL_VIOLATOR]]).astype(int)
    
    # pred = model.predict(arr)
    pred_prob = model.predict_proba(arr)
    return render_template('after.html',arr1=arr, pp=pred_prob)

if __name__ == "__main__":
    app.run(debug=True)

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     features=[X for X in request.form.values()]
#     final=[np.array(features)]
#     # print(features)
#     # print(final)
#     prediction=model.predict(features)
#     # output=accuracy_score(final, prediction)

#     if accuracy_score(final, prediction)>str(0.5):
#         return render_template('index.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
#     else:
#         return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
    

# if __name__ == '__main__':
#     app.run(debug=True)