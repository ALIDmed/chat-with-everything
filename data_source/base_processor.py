from abc import ABC

class BaseProcessor(ABC):
    def load(self):
        """
        load data
        """

    def chunk(self):
        """
        data chunking
        """