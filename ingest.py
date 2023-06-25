import argparse

import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, LatexTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document



# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="The folder to be ingested", type=str)
parser.add_argument("--chunk_size", help="Chunk size", type=int, default=1500)
parser.add_argument('--chunk_overlap', help='Chunk overlap', type=int, default=400)
parser.add_argument('--separator', help='Separator', type=str, default='\n')
parser.add_argument('--use_tex_splitter', help='Use tex splitter', type=bool, default=False)

args = parser.parse_args()

FOLDER = args.folder
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP = args.chunk_overlap
SEPARATOR = args.separator
USE_TEX_SPLITTER = args.use_tex_splitter


class Ingest():

    def __init__(
            self,
            folder,
            chunk_size,
            separator,
            chunk_overlap,
            use_tex_splitter,
    ):
        self.vectorstore = Chroma(persist_directory='./chroma', embedding_function=OpenAIEmbeddings())
        print(f"Count of {self.vectorstore._collection.count()} in vectostore")
        print(f"Deleting previous items from {folder}")
        self.vectorstore._collection.delete(where={'module' : folder})
        print(f"New count, {self.vectorstore._collection.count()}")

        self.folder = folder
        self.chunk_size = chunk_size

        self.data_path = os.path.join('./data', self.folder)

        self.splitter = CharacterTextSplitter(        
            separator = separator,
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
        )

        if use_tex_splitter:
            self.splitter = LatexTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap  = chunk_overlap,
            )

    def _load_tex(self, path):
        with open(path, "r") as f:
            return f.read()

    def ingest(self):
        # find all .pdf files in the data folder

        documents = []
        # pdfs
        pdffiles = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".pdf")]
        for f in pdffiles:
            loader = PyPDFLoader(f)
            docs = loader.load()
            for i in docs: i.metadata['source'] = os.path.basename(f).split(".")[0]
            documents.extend(docs)
        
        #txts
        txtfiles = [f for f in os.listdir(os.path.join('./data', self.folder)) if f.endswith(".txt")]
        for t in txtfiles:
            with open(os.path.join('./data', os.path.join(self.folder, t)), "r") as f:
                documents.append(Document(page_content=f.read(), metadata={"source": os.path.basename(t).split(".")[0] + ' transcript'}))

        # tex
        texfiles = [f for f in os.listdir(os.path.join('./data', self.folder)) if f.endswith(".tex")]
        for t in texfiles:
            documents.append(Document(page_content=self._load_tex(os.path.join('./data', os.path.join(self.folder, t))), metadata={"source": os.path.basename(t).split(".")[0] + ' transcript'}))


        for i in documents:
            i.metadata['module'] = self.folder

        # split texts into chunks
        print("Splitting texts into chunks...")
        chunks = self.splitter.split_documents(documents)
        #[chunks.extend(self.splitter.split_documents(i)) for i in documents]
        embeddings = OpenAIEmbeddings()
        # create store
        print("Embedding chunks...")
        self.vectorstore.add_texts(texts=[d.page_content for d in chunks], metadatas=[d.metadata for d in chunks])

if __name__ == "__main__":
    ingest = Ingest(
        folder = FOLDER,
        chunk_size = CHUNK_SIZE,
        separator = SEPARATOR,
        chunk_overlap = CHUNK_OVERLAP,
        use_tex_splitter = USE_TEX_SPLITTER,
    )
    ingest.ingest()