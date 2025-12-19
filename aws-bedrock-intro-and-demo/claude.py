import boto3
import json

prompt_data = """
Act as a Robert Frost and write a poem on Machine Learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_data
                }
            ]
        }
    ]
}

body = json.dumps(payload)

model_id = "us.anthropic.claude-3-sonnet-20240229-v1:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())

print("Response Body:", response_body)

content = response_body.get("content")
if content and len(content) > 0:
    response_text = content[0].get("text", "")
    print("Generated Poem:", response_text)
else:
    print("No content found or response structure is unexpected.")
