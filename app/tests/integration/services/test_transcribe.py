import os

import pytest
from main import configuration as config
from main.models import SRAVITemplate
from main.services.transcribe import transcribe_signal
from main.utils.db import db_session
from main.utils.io import read_ark_file, read_json_file
from main.utils.pre_process import pre_process_signals


class TestTranscribe:

    @pytest.mark.parametrize(
        'file_name, ref_users, ref_sessions, params, phrase_set, '
        'feature_type, expected_response',
        [
            ('001_S2R0010_S005.ark', [1, 2, 3, 4, 5, 6, 7, 18], [1, 2, 3],
             'params.json', 'S2R', 'AE_norm',
             [{'label': 'S2R10', 'accuracy': 0.97},
              {'label': 'S2R17', 'accuracy': 0.02},
              {'label': 'S2R9', 'accuracy': 0.00}]),

            ('001_S2R0004_S005.ark', [1, 2, 3, 4, 5, 6, 7, 18], [1, 2, 3],
             'params.json', 'S2R', 'AE_norm',
             [{'label': 'S2R2', 'accuracy': 0.55},
              {'label': 'S2R4', 'accuracy': 0.42},
              {'label': 'S2R5', 'accuracy': 0.02}]),

            ('018_S2R0002_S005.ark', [1, 2, 3, 4, 5, 6, 7, 18], [1, 2, 3],
             'params.json', 'S2R', 'AE_norm',
             [{'label': 'S2R2', 'accuracy': 0.89},
              {'label': 'S2R4', 'accuracy': 0.11},
              {'label': 'S2R13', 'accuracy': 0.00}])
        ])
    def test_transcribe(self, file_name, ref_users, ref_sessions,
                        params, phrase_set, feature_type, expected_response):
        ark_path = os.path.join(config.DATA_PATH, feature_type, file_name)
        params_path = os.path.join(config.DATA_PATH, params)

        ark_matrix = read_ark_file(ark_path)
        params = read_json_file(params_path)

        with db_session() as s:
            ref_templates = SRAVITemplate.get(
                s, filter=(
                    (SRAVITemplate.feature_type == feature_type)
                    & (SRAVITemplate.user_id.in_(ref_users))
                    & (SRAVITemplate.phrase.has(phrase_set=phrase_set))
                    & (SRAVITemplate.session_id.in_(ref_sessions))
                )
            )

        ref_signals = pre_process_signals(
            signals=[template.blob for template in ref_templates],
            **params
        )
        ref_signals = [(template.phrase_id, ref_signal)
                       for template, ref_signal in
                       zip(ref_templates, ref_signals)]

        test_signal = pre_process_signals(
            signals=[ark_matrix], **params
        )[0]

        actual_response = transcribe_signal(ref_signals=ref_signals,
                                            test_signal=test_signal,
                                            **params)

        assert actual_response == expected_response
