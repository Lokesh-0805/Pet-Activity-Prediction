import pickle
import pandas as pd
from flask import Flask , request , jsonify , render_template

with open("label.pkl","rb") as file:
    label = pickle.load(file)

with open("model.pkl","rb") as file:
    model = pickle.load(file)

with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == "POST":
        steps_cleaned = str(request.form.get('steps_cleaned'))
        heart_rate_cleaned = int(request.form.get('heart_rate_cleaned'))
        hour = int(request.form.get('hour'))
        day_of_week = int(request.form.get('day_of_week'))

        input_data = pd.DataFrame([[steps_cleaned,heart_rate_cleaned,hour,day_of_week]],columns=['steps_cleaned', 'heart_rate_cleaned', 'hour', 'day_of_week'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data)
       
        if prediction==0:
            result = "Active"
        if prediction == 1:
            result = "Lightly Active"
        else :
            result = "Resting"
        return render_template('predict.html',results = result)
    
    else:
        return render_template('predict.html')
    
if __name__ == "__main__":
    app.run(debug = True)