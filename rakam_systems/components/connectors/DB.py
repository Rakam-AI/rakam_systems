from django.db import models
from django.core.management import call_command

from rakam_systems.components.component import Component

# /home/ec2-user/coding/rs_service_template/application/rakam_systems/rakam_systems/components/connectors/DB.py


class DBComponent(Component):
    def __init__(self, app_label):
        self.app_label = app_label

    def create_model(self, name, fields):
        """
        Dynamically create a model class.
        :param name: Name of the model.
        :param fields: Dictionary of field names and their types.
        :return: Model class.
        """
        attrs = {'__module__': self.app_label}
        attrs.update(fields)
        model = type(name, (models.Model,), attrs)
        return model

    def migrate(self):
        """
        Run migrations for the app.
        """
        call_command('makemigrations', self.app_label)
        call_command('migrate', self.app_label)

# Example usage
if __name__ == "__main__":
    db_component = DBComponent('my_app')

    # Define fields for the model
    fields = {
        'name': models.CharField(max_length=255),
        'age': models.IntegerField(),
    }

    # Create the model
    MyModel = db_component.create_model('MyModel', fields)

    # Run migrations
    db_component.migrate()