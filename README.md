# RAG for Climate Literature

## Tools

### **OpenAI Client Setup**
- recommended by KISSKI

### **Llama_index**
- not sure I'll keep using this, I might switch to LangChain (I haven't tried that yet)

### **SentenceSplitter**
- splits semantically &rarr; has better chances of keeping paragraphs intact *BUT* I don't think this works with the HTML since it's looking for `\n\n` to indicate a paragraph break
- comes with Llama_index

### **FAISS Vector**
- FOSS
- comes with Llama_index

### **HuggingFaceEmbedding**
- local
- FOSS
- comes with Llama_index

## To Do

- [ ] check if there already is a vector
- [ ] find a splitter that works with HTML syntax
- [ ] change prompt for LLM
