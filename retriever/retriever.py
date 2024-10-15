class Retriever:
    def __init__(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs

    def create_retriever(self, db):
        self.retiever = db.as_retriever(
            self.search_type, 
            self.search_kwargs
        )

    def query_vector_store(self, query):
        return self.retriever.invoke(query)