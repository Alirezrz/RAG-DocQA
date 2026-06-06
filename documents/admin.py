from django.contrib import admin
from documents.models import Document, DocumentChunk
@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'chunk_count', 'uploaded_at']
    search_fields = ['title']

@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'content_preview']
    
    def content_preview(self, obj):
        return obj.content[:100] + '...'
    
