# Generated by Django 4.2.5 on 2023-09-08 13:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='IPFSText',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ipfs_hash', models.CharField(max_length=600)),
                ('text', models.TextField()),
            ],
        ),
    ]