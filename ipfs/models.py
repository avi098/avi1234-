from django.db import models


class IPFSText(models.Model):

    ipfs_hash = models.CharField(max_length=600)
    text = models.TextField()

    def __str__(self):
        return self.text
