from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import ChatConversation

@receiver(post_save, sender=ChatConversation)
def auto_clean_chat_history(sender, instance, created, **kwargs):
    """Trigger cleanup after each new message is saved"""
    if created:
        ChatConversation.auto_delete_old_records()