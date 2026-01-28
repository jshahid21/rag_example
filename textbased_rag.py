text = """Dr. Ananya Verma, a legendary Indian physicist born in Kolkata in 1960, has made significant contributions to theoretical physics. Her achievements include the discovery of the Bose–Verma condensate, formulation of quantum entanglement theorems, and resolution of the Verma–Hawking black hole paradox. Dr. Verma’s brilliance has earned her praise from fellow scientists, and her legacy continues to inspire generations of physicists worldwide."""

from sentence_transformers import SentenceTransformer,util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

#chunking
sentences = sent_tokenize(text)
chunks =[' '.join(sentences[i:i+2]) for i in range(0,len(sentences),2)]
print("Auto generated chunks are:")
print(chunks)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #nvidia/Nemotron-Orchestrator-8B

chunk_embeddings = model.encode(chunks,convert_to_tensor=True)

#embed the question
question = "what are discoveries by Dr. Ananya Verma?"
question_embedding = model.encode(question,convert_to_tensor=True)

cosine_scores = util.cos_sim(question_embedding,chunk_embeddings)[0]
top_k = 2
top_indices = np.argsort(-cosine_scores.cpu().numpy())[:top_k]
relevangt_chunks = [chunks[i] for i in top_indices]
print(relevangt_chunks)

context = "\n".join(relevangt_chunks)
print(context)

final_prompt = f"""Context:{context}
          "question":{question}
          Answer :"""
print(final_prompt)

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from google.colab import userdata
userdata.get('OPENAI_API_KEY')

import openai
from google.colab import userdata

# Get the API key directly from Colab's userdata
api_key_from_colab = userdata.get('OPENAI_API_KEY')

# Initialize the OpenAI client by passing the API key directly
client = openai.OpenAI(api_key=api_key_from_colab)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
      {"role":'system',"content":"you are a helpful assistant"},
      {"role":"user","content":final_prompt}
  ]
)
answer = response.choices[0].message.content.strip()
print(answer)

from datasets import load_dataset

ds = load_dataset("Anthropic/AnthropicInterviewer")
