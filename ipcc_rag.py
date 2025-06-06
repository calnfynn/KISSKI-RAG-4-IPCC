from enum import Enum
import argparse
import faiss
import os
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
from dotenv import load_dotenv
from alive_progress import alive_bar


#####
# Enums and variables
#####


ID_prompt = "If you're taking information from the text to answer a question, pass back the ID of the paragraph(s) you're taking the information from."

class Prompt(Enum):
    BASIC = f'{ID_prompt} You are explaining to someone with basic knowledge of the topic.'
    ADVANCED = f'{ID_prompt} You are explaining to someone with advanced knowledge of the topic.'

class Model(Enum):
    LLAMA = 'meta-llama-3.1-8b-instruct'
    #MISTRAL = 'mistral-large-instruct'
    GEMMA = 'gemma-3-27b-it'
    
class Embedding(Enum):
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    GTR = "sentence-transformers/gtr-t5-base"
    MPNET = "sentence-transformers/paraphrase-mpnet-base-v2"

EMBED_DIM_MAP = {
    Embedding.MINILM: 384,
    Embedding.MPNET: 768,
    Embedding.GTR: 768
}

parser = argparse.ArgumentParser(description="Script to build and use a RAG.")

parser.add_argument("--model", type=str, default="LLAMA", choices=[m.name for m in Model], help="Which model to use (default: LLAMA)")
parser.add_argument("--prompt", type=str, default="BASIC", choices=[m.name for m in Prompt], help="Which prompt type to use (default: BASIC)")
parser.add_argument("--embed_model", type=str, default="MINILM", choices=[m.name for m in Embedding], help="Which HuggingFace embedding model to use (default: MINILM)")
parser.add_argument("--input_dir", type=str, default="html", help="Input directory for documents (default: ./html)")
parser.add_argument("--index_dir", type=str, default="./faiss_index", help="Directory for storing/loading the index (default: ./faiss_index)")
parser.add_argument("--tokens", type=int, default=1024, help="Tokens per chunk for splitting (default: 1024)")
parser.add_argument("--overlap", type=int, default=200, help="Token overlap between chunks (default: 200)")
parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the FAISS index even if one exists")


args = parser.parse_args()

llm_model = Model[args.model]
answer_level = Prompt[args.prompt]
embed_model = Embedding[args.embed_model]
vector_dimensions = EMBED_DIM_MAP[embed_model]

index_dir = args.index_dir
input_dir = args.input_dir
tokens_per_chunk = args.tokens
chunk_overlap = args.overlap
force_rebuild = args.rebuild

bar_style = "fishes"


#####
# Load, chunk, and embed HTML
#####


def make_index(index_dir, embed_model, force_rebuild):

    # Embed Chunks with HuggingFace
    embedder = HuggingFaceEmbedding(model_name=embed_model)

    vector_store = FaissVectorStore.from_persist_dir(index_dir)

    faiss_index = vector_store._faiss_index
    stored_dim = faiss_index.d

    #if:
    # - not instructed to rebuild index
    # - stored index fits the dimensions required by embedding model
    # - index directory exists
    # - index directory isn't empty
    if (not force_rebuild) and (stored_dim == vector_dimensions) and os.path.exists(index_dir) and os.listdir(index_dir):

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_dir
        )
        index = load_index_from_storage(storage_context=storage_context, embed_model=embedder)
        print("Using stored index.")

    else:

        # Load HTML file(s)
        documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        print(f"Loaded {len(documents)} document(s).")

        # Chunk with SentenceSplitter (progress bar per doc)
        splitter = SentenceSplitter(chunk_size=tokens_per_chunk, chunk_overlap=chunk_overlap)

        nodes = []
        with alive_bar(title="Generating chunks...", unknown=bar_style, spinner=None) as bar:
            for doc in documents:
                nodes.extend(splitter.get_nodes_from_documents([doc]))
                bar()

        print(f"Generated {len(nodes)} chunks.")

        # Create Index
        faiss_index = faiss.IndexFlatL2(vector_dimensions)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        with alive_bar(title="Creating index...", unknown=bar_style, spinner=None) as bar:
            index = VectorStoreIndex(
                nodes,
                embed_model=embedder,
                storage_context=storage_context,
            )
            bar()


        # Save index
        index.storage_context.persist(persist_dir=index_dir)
        print(f"Index stored in {index_dir}")

    return index

#####
# LLM
#####

def load_llm(llm_model, answer_level):
    
  load_dotenv()

  api_key = os.getenv("OPENAI_API_KEY")
  base_url = os.getenv("KISSKI_URL")

  if not api_key or not base_url:
      raise ValueError("Missing API key or URL.")

  client = OpenAI(
      api_key=api_key,
      base_url=base_url
  )

  def ask_openai_llm(prompt: str) -> str:
      response = client.chat.completions.create(
          model=llm_model,
          messages=[
              {"role": "system", "content": answer_level},
              {"role": "user", "content": prompt}
          ]
      )
      return response.choices[0].message.content
  return ask_openai_llm

#####
# Query
#####

def ask_question(index, ask_openai_llm):
  while True:
      #\033[1m + \033[0m for bold text
      query = input("\033[1m" + "Enter your question (or type 'q' to quit): " + "\033[0m").strip()
      if query.lower() == 'q':
          print("Session ended.")
          break

      nodes = index.as_retriever().retrieve(query)
      context = "\n---\n".join([n.get_content() for n in nodes])
      full_prompt = f"""
  Context:
  {context}

  Question:
  {query}"""

      answer = ask_openai_llm(full_prompt)
      print(f"\nQ:")
      print(query)
      print("\nA:")
      print(answer)
      print("___\n")

#####
# Starting point
#####

if __name__ == "__main__":

    index = make_index(index_dir, embed_model.value, force_rebuild)
    ask_openai_llm = load_llm(llm_model.value, answer_level.value)
    ask_question(index, ask_openai_llm)