import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv

load_dotenv()


def chunk_text(text, max_length=3000):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]


class Chain:

    def __init__(self):

        self.llm = ChatGroq( api_key=os.getenv("GROQ_API_KEY"),model="llama3-70b-8192",temperature=0)

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE: 
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing keys:'Job Title'
            # Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract= prompt_extract| self.llm
        json_parser = JsonOutputParser()

        all_jobs = []

        # Chunking if input is too long
        chunks = chunk_text(cleaned_text, max_length=3000)

        for chunk in chunks:
            try:
                res = chain_extract.invoke(input={"page_data": chunk})
                parsed = json_parser.parse(res.content)
                if isinstance(parsed, list):
                    all_jobs.extend(parsed)
                else:
                    all_jobs.append(parsed)
            except OutputParserException as e:
                print(f"Failed to parse chunk: {e}")
                continue  # or log & skip

        return all_jobs

    def write_email(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Zumanji, a business development executive at BrotherHood. BrotherHood is an AI & Software Consulting company dedicated to facilitate the seamless integration
            of business processes through automated tools.
            Over our experience , we have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency
            Your job is to write a cold email to the client regarding the job ,mentioned above describing the capability of BrotherHood in fulfilling theit needs.
            Also add the most relevant ones from the following links to showcase BrotherHood portfolio: {link_list}
            
            Remember you are Zumanji, BDE at BrotherHood.
            Do not provide a preamble
            ### EMAIL (NO PREAMBLE)
            
            """
        )

        chain_email = prompt_email| self.llm
        res = chain_email.invoke({"job_description" : str(job), "link_list": links})


        return  res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))


