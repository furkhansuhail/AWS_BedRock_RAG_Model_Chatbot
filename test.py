# API Request for LLama 3

#     {
#         "modelId": "meta.llama3-3-70b-instruct-v1:0",
#         "contentType": "application/json",
#         "accept": "application/json",
#         "body": "{\"prompt\":\"this is where you place your input text\",\"max_gen_len\":512,\"temperature\":0.5,\"top_p\":0.9}"
#     }



# APi Request for  Claude 3.5 Haiku

# {
#   "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
#   "contentType": "application/json",
#   "accept": "application/json",
#   "body": {
#     "anthropic_version": "bedrock-2023-05-31",
#     "max_tokens": 200,
#     "top_k": 250,
#     "stopSequences": [],
#     "temperature": 1,
#     "top_p": 0.999,
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {
#             "type": "text",
#             "text": "hello world"
#           }
#         ]
#       }
#     ]
#   }
# }


