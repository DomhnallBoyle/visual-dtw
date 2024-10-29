# import pytest
# from main import configuration
# from main.models import PAVASession
# from main.utils.db import db_session
# from scripts import session_selection as module
# from tests.integration.utils import get_accuracy, sessions_to_templates
#
#
# class TestSessionSelection:
#
#     @pytest.mark.parametrize(
#         'fast, expected_session_labels, expected_accuracies', [
#             (True, ['default_3', 'default_4', 'default_5'],
#              [58.69565217391305, 78.26086956521739, 82.6086956521739]),
#             (False, ['default_6', 'default_5', 'default_3', 'default_4'],
#              [69.56521739130434, 84.78260869565217, 91.30434782608695])
#     ])
#     def test_session_selection(self, client, mocker, fast,
#                                expected_session_labels, expected_accuracies):
#         num_training_sessions = 2
#         num_sessions_to_add = 5
#         num_test_sessions = 2
#
#         # get default sessions
#         with db_session() as s:
#             default_sessions = PAVASession.get(s, filter=(
#                 PAVASession.list_id == configuration.DEFAULT_PAVA_LIST_ID
#             ))
#             assert len(default_sessions) == configuration.NUM_DEFAULT_SESSIONS
#             all_default_sessions = [
#                 (f'default_{i+1}', [(template.phrase.content, template.blob)
#                                     for template in session.templates])
#                 for i, session in enumerate(default_sessions)
#             ]
#
#         training_sessions = all_default_sessions[:num_training_sessions]
#         sessions_to_add = \
#             all_default_sessions[num_training_sessions:][:num_sessions_to_add]
#         test_sessions = \
#             all_default_sessions[num_training_sessions:][num_sessions_to_add:][:num_test_sessions]
#
#         # get training and test templates
#         training_templates = sessions_to_templates(training_sessions)
#         test_templates = sessions_to_templates(test_sessions)
#
#         # prevent algorithm from shuffling so we get the same order
#         # of labels everytime
#         mock_random = mocker.patch.object(module, 'random')
#         mock_random.shuffle.return_value = training_templates
#
#         if fast:
#             actual_session_labels = \
#                 module.session_selection_with_cross_validation_fast(
#                     _sessions=sessions_to_add,
#                     _training_templates=training_templates
#                 )
#         else:
#             actual_session_labels = \
#                 module.session_selection_with_cross_validation(
#                     _sessions=sessions_to_add,
#                     _training_templates=training_templates,
#                     initial_max=2
#                 )
#
#         assert actual_session_labels == expected_session_labels
#
#         # get accuracies of models from external test templates
#         model_sessions = [s for s in sessions_to_add
#                           if s[0] in actual_session_labels]
#         ref_templates = sessions_to_templates(model_sessions)
#         actual_accuracies = get_accuracy(ref_templates, test_templates)
#
#         assert actual_accuracies == expected_accuracies
