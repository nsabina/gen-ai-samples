
import os
import wget

# get a token: https://replicate.com/account
from getpass import getpass

REPLICATE_API_TOKEN = getpass()
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

# set the LLM
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)
# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)
# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

#url = 'https://admin.owners.infinitiusa.com/content/manualsandguides/IPL_G_Coupe/2011/2011-Infiniti-IPL-G-Coupe.pdf'
output_path = '2011-Infiniti-IPL-G-Coupe.pdf'


#filename = wget.download(url, out=output_path)

# Read documents from the specified directory and load a specific document, "report.pdf".
documents = SimpleDirectoryReader("./").load_data(output_path)

# Create a VectorStoreIndex object from the documents. This will involve processing the documents
# and creating a vector representation for each of them, suitable for semantic searching.
index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("How do I clean car glass?")
print(f"{response.response}")