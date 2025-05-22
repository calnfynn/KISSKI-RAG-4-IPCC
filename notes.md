# RAG for Climate Literature

## What's happening? 

1. Load HTML file with SimpleDirectoryReader
2. Semantically-ish chunk HTML with SentenceSplitter
3. Embed chunks into vector with HuggingFace
4. Create re-usable index
5. Set up LLM / KISSKI API
6. Send query, generate answer

## Tools

### KISSKI

- service of a local (to me) computing centre
- has a number of different models (I'm using Llama3)
- recommended to me by my supervisor

### Llama 3

- more open than other models
- good documentation 
- easy to run locally if someone wants to re-use the code without an API

### **OpenAI Client Setup**

- recommended client by KISSKI

### **Llama_index**

- ~~not sure I'll keep using this, I might switch to LangChain (I haven't tried that yet)~~
- thoughts: 
  - llama_index seems to be better suited for RAG
  - not as many general LLM features as LangChain but the RAG specific ones seem to be easier to set up 
  - has everything I want for RAG, no need to make this more complicated (so far?)

### **SentenceSplitter**

- splits semantically &rarr; has better chances of keeping paragraphs intact ***BUT*** I don't think this works with the HTML since it's looking for `\n\n` to indicate a paragraph break
- comes with Llama_index

### **HuggingFaceEmbedding, FAISS Vector**

- local
- FOSS
- comes with Llama_index

## To Do

### right now 

- [ ] check if there already is a vector
- [ ] find a splitter that works with HTML syntax or change settings for the one I have
- [ ] change prompt for LLM
- [ ] embedding works, but seems slow &rarr; try some other tools

### next

#### evaluation

  - [ ] look up if llama_index has any RAG evaluation tools
  - [ ] look for external tools
    - RAGAS?