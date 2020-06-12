from flask import Flask,render_template,redirect,request
import pickle 

#load the file
filename = "spam_sms_LR.pkl"
classifier = pickle.load(open(filename,'rb'))
cv = pickle.load(open("cv-transform.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        c_vect = cv.transform(data).toarray()
        prediction = classifier.predict(c_vect)
        return render_template("result.html",prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)