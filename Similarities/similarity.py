class Similarity():
    def __init__(
            self, name, categories, args
    ):
        self.name       = name 
        self.categories = categories
        self.args       = args 
        self.column     = args.typeColumn
    

    def similarity(self,u):
        return 0
    
    def set_documents(self, documents):
        self.documents = documents 
    
    def set_annotations(self, annotations):
        self.annotations = annotations

    def set_participant(self, participant):
        self.participant = participant 