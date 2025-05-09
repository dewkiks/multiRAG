from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.prompts import PromptTemplate
import time
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os

llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY")
        )

class Multirag:
    def __init__(self, path):
        self.pdf_path = path
        chunks = partition_pdf(
            filename = self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )

        # print(set([str(type(el)) for el in chunks]))
        # elements = chunks[0].metadata.orig_elements
        # chunk_images = [el for el in elements if 'Image' in str(type(el))]
        # print(chunk_images[0].to_dict())

        tables = []
        texts = []
        imagesb64 = []

        for chunk in chunks:
            if "Table" in str(type(chunk)): #extract table
                tables.append(chunk)
        
            if "CompositeElement" in str(type(chunk)):      #extract text and images combined
                texts.append(chunk)
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        imagesb64.append(el.metadata.image_base64)
        # print(texts[0])
        # print(imagesb64[0])
        #IMAGES AND TEXT EXTRACTION WORKS, TABLES NEED FIXING
        self.summarize(texts, tables, imagesb64)

    def summarize(self, texts, tables, imagesb64):
        
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {texts}
        """

        prompt = PromptTemplate(input_variables=["texts"], template=prompt_text)

        #TEXT SUMMARIZATION
        llm_chain = {"texts": lambda x: x} | prompt | llm | StrOutputParser() 
        text_summaries=llm_chain.batch(texts, {"max_concurrency":3})
        #print(text_summaries)

        #IMAGE SUMMARIZATION
        image_prompt="""Describe image in detail. For context the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": image_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{imagesb64}"},
                    },
                ],
            )
        ]
       
        img_prompt = ChatPromptTemplate.from_messages(messages)
        chain= img_prompt | llm | StrOutputParser()

        image_summary = []
        for img in imagesb64:
            time.sleep(1)  #delay between requests to address rate limit
            try:
                result = chain.invoke(img)
                image_summary.append(result)
            except Exception as e:
                print(f"Error processing image: {e}")
                image_summary.append("Failed to process image")
        #print(image_summary)
        self.vectorize(texts, text_summaries, imagesb64, image_summary)

    def vectorize(self, texts, text_summaries, imagesb64, image_summary):
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection("link_collection")

        vectorstore = Chroma(
            client=persistent_client,
            collection_name="link_collection",
            embedding_function=GoogleGenerativeAIEmbeddings(
                google_api_key=os.getenv("GEMINI_API_KEY"),
                model="models/embedding-001"),
        )

        # The storage layer for the parent documents
        store = InMemoryStore()
        id_key = "doc_id"

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add image summaries
        img_ids = [str(uuid.uuid4()) for _ in imagesb64]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summary)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, imagesb64)))

        docs = retriever.invoke(" What are the types of transformers")
        for doc in docs:
            print(str(doc) + "\n\n" + "-" * 80)

# tt = TextBookLoader(r"Data\sem1_debug.pdf")
tt = Multirag(r"Data\transformer.pdf")