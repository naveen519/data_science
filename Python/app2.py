
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)


with open('C:/Users/phsivale/Documents/Trainings/flaskPOC/model24Mar.pkl','rb') as f:
	clf = pickle.load(f)

with open('C:/Users/phsivale/Documents/Trainings/flaskPOC/bin24Mar.pkl','rb') as f:
	binz = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
	json_ = request.get_json(force =True)
	print(json_)
	# requestDF = pd.read_csv(json_['path'])
	try:
		requestDF = pd.DataFrame(json_, index=[0])
	except:
		requestDF = pd.DataFrame(json_)
	
	data_bin = pd.DataFrame(binz.transform(requestDF['Home']))
	requestDF.drop(['Home'],axis=1,inplace=True) 
	requestDF =pd.concat([requestDF,data_bin],axis=1)
	print(requestDF)

	preds = clf.predict(requestDF)
	return jsonify({'prediction': str(preds)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


