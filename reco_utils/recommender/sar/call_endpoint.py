import requests


scoring_uri = 'http://d115de67-b4eb-4d1d-895a-83319479e6ca.westus2.azurecontainer.io/score'

# send a random row from the test set to score
test  = [1,2]
input_data = "{\"data\": [" + str(list(test)) + "]}"

headers = {'Content-Type':'application/json'}

# for AKS deployment you'd need to the service key in the header as well
# api_key = service.get_key()
# headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)}

resp = requests.post(scoring_uri, input_data, headers=headers)

print("POST to url", scoring_uri)
#print("input data:", input_data)
print("prediction:", resp.text)