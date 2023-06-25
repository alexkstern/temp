from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from prompts import VANILLA_PROMPT

import os

import gradio as gr


class Chatbot():

    def __init__(
            self,
    ):
        
        self.models = ['gpt3', 'gpt4']

        # vectorstore
        self.vectorstore = Chroma(persist_directory='./chroma', embedding_function=OpenAIEmbeddings())
        self.vectordbkwargs = {"search_distance": 0.9, 'k' : 4}
        self.modules = list(set([d['module'] for d in self.vectorstore._collection.get(include=['metadatas'])['metadatas']]))
        print(f"Found modules: {self.modules}")


    """ Initialise bots """
    def _get_llm(self, model):
        assert model in self.models

        if model == 'gpt3':
            return OpenAI()
        
        if model == 'gpt4':
            return ChatOpenAI(model='gpt-4')


    def _initialise_augmented_chatbot(self, model):

        #doc_chain = load_qa_with_sources_chain(self._get_llm(model), chain_type="map_reduce")
        #question_generator = LLMChain(llm=self._get_llm(model), prompt=CONDENSE_QUESTION_PROMPT)

        chain = ConversationalRetrievalChain.from_llm(
            self._get_llm(model),
            retriever=self.vectorstore.as_retriever(),
            #combine_docs_chain=doc_chain,
            #question_generator=question_generator,
            return_source_documents=True
            )
            
        return chain
    
    def _initialise_vanilla_chatbot(self, model):
        # vanilla gpt
        template = VANILLA_PROMPT
        prompt = PromptTemplate(template=template, input_variables=["human_input", "history"])
        chain = LLMChain(llm=self._get_llm(model), prompt=prompt)

        return chain


    """ Format """

    def _format_chat_history(self, history):
        res = []
        for human, ai in history:
            res.append(f"Human:{human}\nAI:{ai}")
        return "\n".join(res)

    def _format_search_source_documents(self, documents):
        # add page if none
        for d in documents:
            try:
                d.metadata['page']
            except:
                d.metadata['page'] = ''

        output = ' '.join([
            f'SOURCE {i}\n' + d.page_content.replace('$', '') + '\n\nSource: ' + d.metadata['source'] + '\nPage: ' + str(d.metadata['page']) + '\n\n\n' + '-'*100
            for i, d in enumerate(documents)
        ])

        return output
    
    def _format_chat_source_docments(self, documents):
        # add page if none
        for d in documents:
            try:
                d.metadata['page']
            except:
                d.metadata['page'] = 0


        # get unique sources
        unique_sources = list(set([d.metadata['source'] for d in documents]))
        # get unique pages for each source
        unique_dict = {s : list(set([d.metadata['page'] for d in documents if d.metadata['source'] == s])) for s in unique_sources}

        output = '\n'.join([
            f"{k}, pages: " + ', '.join([str(i) for i in v])
            for k, v in unique_dict.items()
        ])

        return '\n\n' + 'SOURCES:\n' + output


    """ Main Functions """
    def search(
            self,
            inp,
            history,
            module,
    ):
        history = history or []

        output_raw = self.vectorstore.similarity_search(inp, filter=dict(module=module), k=8)
        output = self._format_search_source_documents(output_raw)

        history.append((inp, output))

        return history, history

    def chat(
            self,
            inp: str,
            history,
            module,
            model,
        ):

        """Method for integration with gradio Chatbot"""
        if model == None:
            model = 'gpt4'

        history = history or []
        
        chain = self._initialise_augmented_chatbot(model=model)
        output_raw = chain(
            {
                "question": inp,
                "chat_history": history,
                "vectordbkwargs":
                self.vectordbkwargs | {"filter" : {"module" : module}}
            }
        )

        output = output_raw["answer"] + self._format_chat_source_docments(output_raw["source_documents"])

        history.append((inp, output))
        
        return history, history#, ""
    
    def chat_vanilla(
            self,
            inp: str,
            history,
            model,
    ):
        """ Vanilla GPT 4"""

        if model == None:
            model = 'gpt4'

        history = history or []

        chain = self._initialise_vanilla_chatbot(model=model)
        history_formatted = self._format_chat_history(history)
        output = chain({"human_input": inp, "history": history_formatted})['text']

        history.append((inp, output))

        return history, history


    """ Interface """
    def launch_app(self):

        block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

        with block:
            
            with gr.Row():
                gr.Markdown("<h3><center>y2clutch</center></h3>")


            with gr.Tab("Augmented GPT"):
                

                with gr.Row():
                    chatbot = gr.Chatbot()

                
                with gr.Row():
                    message = gr.Textbox(
                        lines=1,
                    )
                    submit = gr.Button(value="Send", variant="secondary").style(full_width=False)
                    
                state = gr.State()
                module = gr.Dropdown(self.modules, label="Select a module *Required*")
                model = gr.Dropdown(self.models, label="Select a model *Required*")

                submit.click(self.chat, inputs=[message, state, module, model], outputs=[chatbot, state])
                message.submit(self.chat, inputs=[message, state, module, model], outputs=[chatbot, state])
                
                gr.Examples(
                    examples=[
                    'Answer the following question, explain your reasoning:\n',
                    'Answer the following question, explain your reasoning, use latex format:\n',
                    'Answer the following multiple choice question, explain your reasoning:\n',
                    ],
                    inputs=message
                )
            
            with gr.Tab("Search"):

                with gr.Row():
                    search = gr.Chatbot()

                with gr.Row():
                    message = gr.Textbox(
                        lines=1,
                    )
                    submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

                search_state = gr.State()
                module = gr.Dropdown(self.modules, label="Select a module *Required*")

                submit.click(self.search, inputs=[message, search_state, module], outputs=[search, search_state])
                message.submit(self.search, inputs=[message, search_state, module], outputs=[search, search_state])


            with gr.Tab("Vanilla GPT"):

                with gr.Row():
                    vanilla_chatbot = gr.Chatbot()

                with gr.Row():
                    message = gr.Textbox(
                        lines=1,
                    )
                    submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

                vanilla_state = gr.State()
                model = gr.Dropdown(self.models, label="Select a model *Required*")

                submit.click(self.chat_vanilla, inputs=[message, vanilla_state, model], outputs=[vanilla_chatbot, vanilla_state])
                message.submit(self.chat_vanilla, inputs=[message, vanilla_state, model], outputs=[vanilla_chatbot, vanilla_state])


        block.launch(debug=True, share=False)
        

if __name__ == '__main__':
    Chatbot().launch_app()