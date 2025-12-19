from openai import OpenAI
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

api_key=os.getenv("NVIDIA_API_KEY")

from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

completion = client.chat.completions.create(
  model="meta/llama-3.2-1b-instruct",
  messages=[{"role":"user","content":"tell me about bangladesh"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

