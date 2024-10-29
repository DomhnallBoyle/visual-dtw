import numpy as np
import pytest
from main import configuration, create_app, models
from main.utils.io import read_json_file


@pytest.fixture(scope='session')
def create_object_fixtures():
    def wrapper():
        data = read_json_file(configuration.FIXTURES_PATH)

        for model_fixtures in data:
            model = model_fixtures['model']
            model_cls = getattr(models, model)
            records = model_fixtures['records']
            for record in records:
                obj = model_cls.create(**record)

                # assign random matrix blobs to the templates
                if isinstance(obj, models.PAVATemplate):
                    models.PAVATemplate.update(
                        id=obj.id, blob=np.random.rand(100, 100)
                    )

    return wrapper


@pytest.fixture(scope='session')
def client(create_object_fixtures):
    # this method is ran once within the test session
    app = create_app(drop_db=True)

    create_object_fixtures()

    with app.test_client() as client:
        yield client
