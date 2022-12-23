import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle,joblib


app = Flask(__name__)
#model = pickle.load(open('random_forest_regression_model.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')
    

@app.route('/predict/',methods = ['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        T = request.form.get('T')
        TM = request.form.get('TM')
        Tm = request.form.get('Tm')
        SLP = request.form.get('SLP')
        H = request.form.get('H')
        VV = request.form.get('VV')
        V = request.form.get('V')
        VM = request.form.get('VM')
        try:
            prediction=preprocess(T,TM,Tm,SLP,H,VV,V,VM)
            return render_template('predict.html',prediction=prediction)
        
        except ValueError:
            
            return "Please enter valid values"
        
        pass
        
    pass
    
def preprocess(T,TM,Tm,SLP,H,VV,V,VM):
    test_data=[T,TM,Tm,SLP,H,VV,V,VM]
    print(test_data)
    
    test_data=np.array(test_data).astype(np.float)
    
    test_data=test_data.reshape(1,-1)
    print(test_data)
    
    file=open('random_forest_regression_model.pkl','rb')
    trained_model=joblib.load(file)
    
    prediction=trained_model.predict(test_data)
    return prediction
    pass

#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
    

#     output = prediction[0]
#     return render_template('home.html', prediction_text="AQI for Jaipur {}".format(output))
   
#     pass


if __name__ == '__main__':
    app.run(debug=True)
    
   