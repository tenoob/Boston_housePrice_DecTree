
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForestRegressor.pkl','rb'))

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def prediction():
    if (request.method=='POST'):
        crim = float(request.form['crim'])
        indus = float(request.form['indus'])
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b = float(request.form['b'])
        lstat = float(request.form['lstat'])

        ans = model.predict([[crim,indus,nox,rm,age,dis,tax,ptratio,b,lstat]])

        return render_template('index.html',result=round(ans[0],2))

if __name__=='__main__':
    app.run(debug=True)