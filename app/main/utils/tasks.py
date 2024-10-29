from main import cache, celery
from main.models import Config, PAVAList, PAVAModelAccuracy, \
    PAVAModelSession, PAVAPhrase, PAVASession, PAVATemplate, PAVAUser
from main.services.transcribe import transcribe_signal
from main.utils.db import db_session
from main.utils.exceptions import ListNotFoundException, UserNotFoundException
from sqlalchemy.orm import joinedload

get_user_functions = {
    'user_id': lambda s, user_id: PAVAUser.get(
        s, filter=(PAVAUser.id == user_id), first=True
    ),
    'list_id': lambda s, list_id: PAVAUser.get(
        s, filter=((PAVAList.id == list_id)
                   & (PAVAList.user_id == PAVAUser.id)), first=True
    ),
    'phrase_id': lambda s, phrase_id: PAVAUser.get(
        s, filter=((PAVAPhrase.id == phrase_id)
                   & (PAVAPhrase.list_id == PAVAList.id)
                   & (PAVAList.user_id == PAVAUser.id)), first=True
    ),
    'session_id': lambda s, session_id: PAVAUser.get(
        s, filter=((PAVASession.id == session_id)
                   & (PAVASession.list_id == PAVAList.id)
                   & (PAVAList.user_id == PAVAUser.id)), first=True
    ),
}


@celery.task
def refresh_cache(view_args):
    """Update cache with user lists and their last activity timestamp"""
    if not view_args:
        return False

    with db_session() as s:
        user = None
        for arg_name in get_user_functions.keys():
            if arg_name in view_args:
                try:
                    user = get_user_functions[arg_name](s, view_args[arg_name])
                except UserNotFoundException:
                    continue

                assert isinstance(user, PAVAUser)
                break

        if not user:
            return False

        for lst in user.lists:
            # only cache lists with models
            current_model = lst.current_model
            if current_model:

                ref_signals = [
                    (template.phrase.content, template.blob)
                    for template in current_model.get_templates()
                ]

                # this resets the timeout on the key when set again
                cache.set(lst.cache_key, ref_signals)

        return True


@celery.task
def test_models(list_id):
    """Test models of a particular list id

    Calculates the Rank 1 accuracy

    The following events trigger this task:
    - Every 25 transcribed & labelled templates
    - A new session is completed (not in any model and recorded in full) (DEPRECATED)
    - A model is created
    """
    dtw_params = Config().__dict__

    with db_session() as s:
        try:
            lst = PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
        except ListNotFoundException:
            return False

        list_models = lst.models
        list_sessions = lst.sessions

    if not list_models or not list_sessions:
        return False

    # grab all list templates (transcription and non model session ones)
    # to ensure fair testing, use same templates and sessions that aren't
    # in any of the models
    all_session_ids = [s.str_id for s in list_sessions]
    model_session_ids = []
    for model in list_models:
        model_session_ids.extend(
            [
                str(_id[0]) for _id in PAVAModelSession.get(
                    s, query=(PAVAModelSession.session_id,),
                    filter=(PAVAModelSession.model_id == model.id)
                )
            ]
        )
    non_model_session_ids = list(
        set(all_session_ids) - set(model_session_ids)
    )
    test_templates = PAVATemplate.get(
        s, loading_options=(joinedload('phrase'),), filter=(
            (PAVATemplate.phrase_id == PAVAPhrase.id)
            & (PAVAPhrase.list_id == list_id)
            & ((PAVATemplate.session_id == None) |
               (PAVATemplate.session_id.in_(non_model_session_ids)))  # either don't belong to a session or belong to non model sessions
            & (PAVAPhrase.content != 'None of the above')
            & (PAVAPhrase.archived == False)
        )
    )
    test_signals = [(template.phrase.content, template.blob)
                    for template in test_templates]
    if not test_signals:
        return False

    num_test_signals = len(test_signals)

    for model in list_models:
        ref_signals = [(template.phrase.content, template.blob)
                       for template in model.get_templates()]

        if not ref_signals:
            continue

        num_correct = 0
        for actual_label, test_signal in test_signals:
            try:
                predictions = transcribe_signal(ref_signals, test_signal, None,
                                                **dtw_params)
            except Exception:
                continue

            if actual_label == predictions[0]['label']:
                num_correct += 1

        accuracy = round(num_correct / num_test_signals, 2)

        PAVAModelAccuracy.create(model_id=model.id, accuracy=accuracy,
                                 num_test_templates=num_test_signals)

    return True
