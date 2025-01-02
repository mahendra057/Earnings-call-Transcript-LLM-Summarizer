from scraping import precise_scarping
from preprocessing import initialize_azure_openai_llm,split_text_by_tokens
from llm_summarizer import stuff_refine_summarizer
from langchain.schema import Document # Wrap each chunk in a Document object
import streamlit as st
from langchain.callbacks import get_openai_callback  # Callback for token usage tracking



#input_string_new=precise_scarping()

#llm=initialize_azure_openai_llm()
#token_chunks = split_text_by_tokens(input_string_new, chunk_size=16000, overlap=100)
#print(f"Total Number of Chunks: {len(token_chunks)}")
#document_chunks = [Document(page_content=chunk) for chunk in token_chunks]

#result=stuff_refine_summarizer(llm,document_chunks)
#print(result)



# Streamlit App
def main():
    # Streamlit Title and Description
    st.title("Earnings Call Transcript Summarizer 📊")
    st.write(
        "This app scrapes an earnings call transcript, processes the content, "
        "and generates a summarized version focusing on financial metrics, "
        "operational updates, and strategic guidance."
    )

    # User Input for URL
    url = st.text_input("Enter the URL of the earnings call transcript:", "")
    if not url:
        st.warning("Please enter a valid URL to proceed.")
        return

    # Button to trigger scraping and summarization
    if st.button("Generate Summary"):
        st.info("Scraping data from the earnings call webpage...")
        input_string_new = precise_scarping(url)

        # Initialize the Azure OpenAI LLM
        st.info("Initializing Azure OpenAI LLM...")
        llm = initialize_azure_openai_llm()

        # Split text into tokenized chunks
        st.info("Splitting the text into chunks...")
        token_chunks = split_text_by_tokens(input_string_new, chunk_size=16000, overlap=100)
        st.write(f"Total Number of Chunks: {len(token_chunks)}")

        # Wrap token chunks into Document objects
        document_chunks = [Document(page_content=chunk) for chunk in token_chunks]

        # Run the summarizer
        st.info("Summarizing the document...")
        # Track token usage using LangChain's callback
        with get_openai_callback() as cb:
            result_summary, result_table = stuff_refine_summarizer(llm, document_chunks)

            # Display the results
            st.success("Summary and Table Generated Successfully!")
            st.subheader("Summary:")
            st.write(result_summary)

            st.subheader("Table of Metrics:")
            st.write(result_table)

            # Display token usage and cost details
            st.subheader("Token Usage Details 📈")
            st.write(f"**Tokens Used:** {cb.total_tokens}")
            st.write(f"**Prompt Tokens:** {cb.prompt_tokens}")
            st.write(f"**Completion Tokens:** {cb.completion_tokens}")
            st.write(f"**Total Cost (USD):** ${cb.total_cost:.4f}")

            # Print to console for debugging
            print(result_summary)
            print(result_table)
            print(f"Tokens Used: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost:.4f}")


if __name__ == "__main__":
    # Set Streamlit page configuration with dark mode
    st.set_page_config(
        page_title="Earnings Call Summarizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()