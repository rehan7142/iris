from flask import  Flask
import numpy as np
from flask import render_template,request
import joblib
app = Flask(__name__)
model = joblib.load('model.pkl')
@app.route('/')
def home():
	return render_template("index.html")
@app.route('/predict' , methods = ['POST','GET'])
def predict():
	data1 = request.form['a']
	data2 = request.form['b']
	data3 = request.form['c']
	data4 = request.form['d']
	arr = np.array([[data1,data2,data3,data4]])
	pred = model.predict(arr)
	return render_template("after.html", data=pred)
if __name__ == "__main__":
	app.run(debug = True)

