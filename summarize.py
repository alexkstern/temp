from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, LLMChain, PromptTemplate

from datasets import Dataset
import textwrap
from tqdm import tqdm
import os 






template = 'Write a verbose summary of the following:\n\n\n"{text}"\n\nDo not omit any information. VERBOSE SUMMARY:\n\n\n'
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=False)

def _summarize_func(chunk):
    chunk = chunk["chunk"]
    summary = chain.run(chunk)
    assert isinstance(summary, str)
    return dict(summary=summary)


class RecursiveSummarizer():

    def __init__(
            self,
    ):       
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3900, chunk_overlap=0, separator=" ")

    def _save_txt(self, string, lecture_number):

        #string = textwrap.wrap(string, width=60)
        #string = "\n".join(string)
        with open(f"./data/bab/lecture_{lecture_number}.txt", "w") as f:
            f.write(string)
            
    def summarize(self, text, n):

        text_length = len(text)
        print("Initial Text length: ", text_length)

        i = 0
        while text_length > 18000:
            i += 1
            print(f"Summarizing p{i}...")
            # split text into chunks
            chunks = self.splitter.split_text(text)
            print(f"Number of chunks: {len(chunks)}")

            # summarize each chunk in different threads
            ds = Dataset.from_list([{"chunk": chunk} for chunk in chunks])
            summaries = ds.map(_summarize_func, num_proc=len(chunks), remove_columns=["chunk"])['summary']

            # join summaries
            summary = " ".join(summaries)

            text = summary
            text_length = len(text)

        self._save_txt(text, lecture_number=n)

        return text
    

if __name__ == "__main__":
    summarizer = RecursiveSummarizer()
    
    # scan for .txt files
    txtfiles = [f for f in os.listdir(".") if f.endswith(".txt") if f.startswith("lecture")]

    for t in tqdm(txtfiles):
        # extract lecture number
        n = t.split("_")[1].split(".")[0]
        print(f"Summarizing {t}...")
        # get text from text.txt
        with open(t, "r") as f:
            text = f.read()

        summarizer.summarize(text, n)