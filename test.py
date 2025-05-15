import boto3
import json

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-2'  # Replace with your AWS region
)

model_id = 'meta.llama3-3-70b-instruct-v1:0' # Replace with the desired model
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

response_body = json.loads(response.get('body').read())
print(response_body)
