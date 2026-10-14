import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=200,
    messages=[
    {"role": "user", "content": "Here's some text: 'The Eiffel Tower was completed in 1889 and stands 330 meters tall.' Based only on this text, how tall is the Eiffel Tower?"}
]
)

print(response.choices[0].message.content)