import boto3
import json

from Model_Ids import Llama_Id

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-2'  # Replace with your AWS region
)

model_id = Llama_Id # Replace with the desired model
prompt = "Act as a Shakespeare and write a poem on Generative AI"
body = json.dumps({
    "prompt": prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
    # "parameters": {
    #   "temperature": 0.7,
    #   "top_p": 0.9,
    #   "max_gen_len": 256
    # }
})

print(body)
response = bedrock_runtime.invoke_model(
    body=body,
    modelId=model_id,
    contentType='application/json',
    accept='application/json'
)

# response_body = json.loads(response.get('body').read())
# print(response_body)
response_body=json.loads(response.get("body").read())
repsonse_text = response_body['generation']
print(repsonse_text)
