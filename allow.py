import numpy as np
import json as json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

modal = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

modelh = pickle.load(open('pipeh.pkl', 'rb')) 
# here h is for house
dfh = pickle.load(open('dfh.pkl', 'rb'))

modalc = pickle.load(open('pipec.pkl','rb'))
# here c is for car
dfc = pickle.load(open('dfc.pkl','rb'))


@app.route('/predict/price', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # print(data['Ram'])
    to_predict = [data['company'], data['TypeName'], data['Ram'], data['Weight'], data['Touchscreen'],
                  data['IPS'], data['Ppi'], data['Cpubrand'], data['HDD'], data['SSD'], data['Gpubrand']]

    res = np.exp(modal.predict([to_predict]))

    print("prediction made")
    return jsonify((int)(res))


@app.route('/formData', methods=['GET'])
def data():

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    Type = df['TypeName'].unique()
    company = df['Company'].unique()
    Ram = df['Ram'].unique()
    touchscreen = df['Touchscreen'].unique()
    ips = df['IPS Panel'].unique()
    cpubrand = df['Cpu brand'].unique()
    hdd = df['HDD'].unique()
    ssd = df['SSD'].unique()
    gpubrand = df['Gpu brand'].unique()

    form_data = json.dumps({'company': company, 'Type': Type, 'Ram': Ram, 'touchscreen': touchscreen, 'ips': ips, 'cpubrand': cpubrand,
                            'hdd': hdd, 'ssd': ssd, 'gpubrand': gpubrand},
                           cls=NumpyEncoder)

    print("data sended")
    return form_data


@app.route('/housePricePredictor',methods=['POST'])
def hpredict():
    data = request.get_json(force=True)

    print(data['location'])

    to_predict = [data['location'],data['total_sqft'],data['bath'],data['bhk']]

    res = np.exp(modelh.predict([to_predict]))*1e5

    print("data recieved")
    return jsonify((int)(res))

@app.route('/formDatah', methods=['GET'])
def datah():

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    location = dfh['location'].unique()
    bath = dfh['bath'].unique()

    form_data = json.dumps({'location': location,'bath':bath},
                           cls=NumpyEncoder)

    print("data sended")
    return form_data


@app.route('/carPricePredictor',methods=['POST'])
def cpredict():
    data = request.get_json(force=True)

    print(data['company'])

    to_predict = [data['name'],data['company'],data['year'],data['kms_driven'],data['fuel_type']]

    res = np.exp(modalc.predict([to_predict]))

    print("data recieved")
    return jsonify((int)(res))


@app.route('/formDatac', methods=['GET'])
def datac():

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    name = dfc['name'].unique()
    company = dfc['company'].unique()
    fuel_type = dfc['fuel_type'].unique()
    year = dfc['year'].unique()

    form_data = json.dumps({'name': name,'company':company,'fuel_type':fuel_type,'year':year},
                           cls=NumpyEncoder)

    print("data sended")
    return form_data


if __name__ == '__main__':
    app.run(debug=True)
