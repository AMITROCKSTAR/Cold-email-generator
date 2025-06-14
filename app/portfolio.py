import pandas as pd
import chromadb
import uuid

class Portfolio:
    def __init__(self, file_path="app/resource/techstack_links_rowwise.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')  # for pandas >= 1.3.0



        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")


    def load_portfolio(self):
        if not self.collection.count():
            for _,row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        # Filter invalid skill strings
        filtered_skills = [s for s in skills if isinstance(s, str) and s.strip()]
        if not filtered_skills:
            return []

        try:
            results = self.collection.query(query_texts=filtered_skills, n_results=2)
            if not results or 'metadatas' not in results:
                return []

            metadatas = results['metadatas']
            links = [m.get('links') for m in metadatas if isinstance(m, dict) and 'links' in m]
            return links

        except Exception as e:
            print(f"[ERROR] ChromaDB query failed: {e}")
            return []

