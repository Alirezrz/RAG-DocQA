import json
from django.db import models


class Document(models.Model):

    title  = models.CharField(max_length=255)
    file       = models.FileField(upload_to='documents/')
    extracted_text = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at= models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    @property
    def chunk_count(self):
        return self.chunks.count()


class DocumentChunk(models.Model):
    """
    bm25_tf_json:
    Stores the term-frequency map for this chunk as a JSON string.
    Example:  {"contract": 3, "duration": 2, "party": 5, ...}
    """
    document= models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content= models.TextField()
    chunk_index  = models.IntegerField(default=0)
    embedding_json = models.TextField(null=True, blank=True)

    bm25_tf_json  = models.TextField(null=True, blank=True)

    def set_embedding(self, vector: list):
        self.embedding_json = json.dumps(vector)

    def get_embedding(self) -> list:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return []

    def set_bm25_tf(self, tf_dict: dict):
        """Store term-frequency dict as JSON."""
        self.bm25_tf_json = json.dumps(tf_dict)

    def get_bm25_tf(self) -> dict:
        """Return term-frequency dict (empty dict if not set)."""
        if self.bm25_tf_json:
            return json.loads(self.bm25_tf_json)
        return {}

    def __str__(self):
        return f"{self.document.title} — Chunk {self.chunk_index}"