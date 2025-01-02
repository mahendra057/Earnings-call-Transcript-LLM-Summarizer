from bs4 import BeautifulSoup
import requests


def precise_scarping(url):

    # URL of the webpage to scrape
    #url = "https://www.fool.com/earnings/call-transcripts/2024/12/10/designer-brands-dbi-q3-2024-earnings-call-transcri/"

    # Fetch the webpage
    page = requests.get(url)
    page.raise_for_status()  # Check if the request was successful
    soup = BeautifulSoup(page.text, 'html.parser')



    # Extract the <p> tag with the specific class
    p_tag = soup.find("p", class_="text-xl font-bold text-gray-1100 mb-32px")
    p_text = p_tag.text if p_tag else ""

    # Extract the <h1> tag with the specific class
    h1_tag = soup.find("h1", class_="text-3xl font-medium tracking-tight text-gray-1100 leading-relative-2 md:text-5xl")
    h1_text = h1_tag.text if h1_tag else ""

    # Find the <div> with class "article-body" and get all <p>, <h1>-<h6>, <ul>, and <li> tags within it
    article_body_div = soup.find("div", class_="article-body")
    all_tags = article_body_div.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])

    # Exclude the first paragraph and the last two paragraphs
    filtered_tags = all_tags[1:-2]

    # Concatenate all filtered tag text into a single string
    filtered_text = "\n".join([tag.text for tag in filtered_tags])

    # Combine all the outputs into a single string
    input_string_new = f"{p_text}\n\n{h1_text}\n\n{filtered_text}"

    #print(input_string_new)

    return input_string_new
