import main.utils.tasks as module
from main.models import PAVAList, PAVAModel, PAVAPhrase, PAVATemplate, \
    PAVAUser


def test_refresh_cache(mocker, random_signal):
    view_args = {'list_id': 'fake_list_id'}

    mock_cache = mocker.patch.object(module, 'cache')

    phrase_content = 'Fake phrase'
    mock_phrase = mocker.create_autospec(PAVAPhrase)
    mock_phrase.content = phrase_content

    template_signal = random_signal(3, 3)
    mock_template = mocker.create_autospec(PAVATemplate)
    mock_template.phrase = mock_phrase
    mock_template.blob = template_signal

    mock_model = mocker.create_autospec(PAVAModel)
    mock_model.get_templates.return_value = [mock_template]

    lst_cache_key = 'fake_cache_key'
    mock_lst = mocker.create_autospec(PAVAList)
    mock_lst.cache_key = lst_cache_key
    mock_lst.current_model = mock_model

    mock_user = mocker.create_autospec(PAVAUser)
    mock_user.lists = [mock_lst]

    mocker.patch.dict(module.get_user_functions,
                      {'list_id': lambda s, _id: mock_user})

    # synchronous blocking call to celery task
    result = module.refresh_cache.apply(args=(view_args,)).get()

    mock_cache.set.assert_called_once_with(lst_cache_key, [
        (phrase_content, template_signal)
    ])

    assert result


def test_test_models(mocker):
    pass
