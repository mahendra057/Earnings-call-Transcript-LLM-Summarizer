from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain, RefineDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import warnings
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.llm import LLMChain
warnings.filterwarnings("ignore")

def stuff_refine_summarizer(llm, document_chunks):
    # Implementing Stuff

    # Define the stuff prompt for single chunk
    stuff_template = """
    You are a financial analyst tasked with creating a **visually structured** summarizing an earnings call transcript. The document provided is comprehensive, and your goal is to create a final summary based on the following guidelines:
    Use **bold** for headings like "Financial Metrics," "Operational Updates," and "Guidance and Strategic Focus."
    1. **Title**: Begin with a **motivational title** reflecting the company's achievements or aspirations.
    2. **Introduction**: Provide a **concise introduction** summarizing the purpose and highlights of the earnings call.
    - Bullet points highlighting:
     **Financial Metrics**: Highlight key figures such as revenue, earnings per share (EPS), profit margins, and any guidance updates.
     **Operational Updates**: Mention significant developments such as product launches, customer growth, partnerships, and other notable changes.
     **Guidance and Strategic Focus**: Include forward-looking statements, market trends, or strategic priorities outlined during the call.
    
    Ensure the summary is cohesive, precise
    Ensure headings are in bold and the content should be visually structured and maintain space between among each metric

    The following Document to be Summarized:
    {input_documents}
    """
    stuff_prompt = ChatPromptTemplate([("human", stuff_template)])
    stuff_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=stuff_prompt), document_variable_name="input_documents"
    )

    ## Implementing Refine 

    # Document formatting prompt
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    document_variable_name = "context"

    # Initial summarization prompt (Map stage)
    summarize_prompt = ChatPromptTemplate(
        [
            ("human", """
            You are a financial analyst summarizing an earnings call transcript.  Please focus on the following aspects while summarizing the text below:
            1. Financial Metrics: Key figures (revenue, EPS, profit margins, and updates to guidance).
            2. Operational Updates: Product launches, customer growth, partnerships, and other significant changes.
            3. Guidance and Strategic Focus: Forward-looking statements, market trends, and strategic priorities.

            Summarize the following text:
            ------------
            {context}
            ------------

            Provide a concise summary.
            """),
        ]
    )

    initial_llm_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    initial_response_name = "existing_answer"

    # Refinement prompt (Reduce stage)
    refine_prompt = ChatPromptTemplate(
        [
            ("human", """
            Refine the following summary using new context provided:

            Existing Summary:
            {existing_answer}

            New Context:
            ------------

            {context}
            ------------

            You are tasked with creating a **visually structured final summary of an earnings call transcript
            Focus on:
            1. Financial Metrics (revenue, EPS, margins, guidance).
            2. Operational Updates (product launches, partnerships, growth).
            3. Guidance and Strategic Focus (forward-looking statements).
            Use **bold** for headings like "Financial Metrics," "Operational Updates," and "Guidance and Strategic Focus."

            Produce a cohesive final summary structure:
            - A **motivational title** reflecting achievements.
            - A concise **introduction** about the earnings call.
            - Bullet points highlighting:
            * **Key financial metrics.**
            * **Significant operational updates.**
            * **Forward-looking guidance and priorities.**
            Ensure headings are in bold and the content should be visually structured.
            """),
        ]
    )

    refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)

    # RefineDocumentsChain configuration
    refine_chain = RefineDocumentsChain(
        initial_llm_chain=initial_llm_chain,
        refine_llm_chain=refine_llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
    )

    # Generating table based on summary
    table_prompt = ChatPromptTemplate(
        [
            ("human", """
            Below is the summary of an earnings call transcript:

            Summary:
            {summary}

            Your task:
            - Extract numerical terms from the summary.
            - Create a table with the following columns:
              1. Metric Name (e.g., Revenue, EPS, Margins).
              2. Numerical Value.

            Ensure the table is structured and easy to interpret.
            """),
        ]
    )

    table_llm_chain = LLMChain(llm=llm, prompt=table_prompt)

    if len(document_chunks) == 1:
        # Single document processing
        print("Single document processing")
        result_summary = stuff_chain.invoke({"input_documents": document_chunks})
        if isinstance(result_summary, dict) and "output_text" in result_summary:
            result_summary = result_summary["output_text"]
    else:
        # Map-Reduce processing
        print("Refine Processing")
        result_summary = refine_chain.run(document_chunks)
        if isinstance(result_summary, dict) and "output_text" in result_summary:
            result_summary = result_summary["output_text"]

    # Generate the table based on the summary
    table_result = table_llm_chain.invoke({"summary": result_summary})
    table_output = table_result["text"]

    return result_summary, table_output
