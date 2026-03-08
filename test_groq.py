import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write one short sentence about stock prediction."}
    ],
    temperature=0.3,
    max_completion_tokens=60,
)

print(response.choices[0].message.content)
