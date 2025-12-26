import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Configurar el modelo que confirmamos que funciona
# Configuramos el modelo con el parámetro de temperatura
# 0.0 = Muy preciso, casi no varía (ideal para datos técnicos)
# 0.7 - 0.9 = Más creativo y fluido (ideal para consultoría o resúmenes)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.7  # <--- Agrega esta línea
)
# 2. Leer tu archivo local
with open("datos.txt", "r", encoding="utf-8") as f:
    contenido = f.read()

# 3. Fragmentar el texto (Crucial para RAG)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
fragmentos = text_splitter.split_text(contenido)

print(f"✅ Archivo cargado y dividido en {len(fragmentos)} fragmentos.")

# 4. Pregunta al modelo usando el contexto de tu archivo
pregunta = "¿Qué hace exactamente la empresa de Jeremías y por qué es relevante?"
prompt = f"""
Actúa como un consultor experto en tecnología. 
Usa el siguiente contexto para explicar con tus propias palabras la actividad de la empresa:
---
{contenido}
---
Instrucción: No repitas el texto de forma literal. Explícalo de forma profesional y amena.
"""

response = llm.invoke(prompt)
print("\n🤖 Respuesta basada en tu archivo:")
print(f"\n🤖 Respuesta: {response.content[0]['text']}")