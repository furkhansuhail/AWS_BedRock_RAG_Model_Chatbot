import boto3
import json

# Claude 3.5 Haiku model ID (use this exact string)
model_id = "Enter Model Id"

# Define your prompt
prompt_text = "Can you write a python code to make a dataframe"

# Claude message format (JSON body)
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 200,
    "top_k": 250,
    "stop_sequences": [],
    "temperature": 1,
    "top_p": 0.999,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]
}

# Initialize the Bedrock Runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-2")  # Adjust region if needed

# Invoke the model
response = bedrock.invoke_model(
    modelId=model_id,
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json"
)

# Parse and print the response
response_body = json.loads(response['body'].read())
response_text = response_body['content'][0]['text']
print(response_text)
