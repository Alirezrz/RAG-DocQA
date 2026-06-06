from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentchunk',
            name='bm25_tf_json',
            field=models.TextField(blank=True, null=True),
        ),
    ]
