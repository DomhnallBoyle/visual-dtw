"""Try wav2lip in the session selection algorithm to improve on users who
can't give recordings

Settings:
    - speed - fast or slow
    - accent - co.uk, com, com.au, ie, ca, co.in, co.za

    python app/main/research/wav2lip_session_selection.py /media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/a98b8478-3f11-4968-b287-ee0f40d2c5b4
"""
import argparse
import random
from http import HTTPStatus

import requests
from main.research.research_utils import *
from scripts.session_selection import \
    session_selection_with_cross_validation_fast

ACCENTS = [
    'co.uk',
    'com',
    'com.au',  # australia
    'ie',
    'ca',  # canada
    'co.in',  # india
    'co.za'  # south africa
]


def main(args):
    templates = create_templates(args.videos_directory,
                                 include_video_paths=True,
                                 save=True)
    random.shuffle(templates)

    default_templates = sessions_to_templates(get_default_sessions())

    num_sessions = 20
    cloning_templates = templates[:num_sessions]
    other_templates = templates[num_sessions:]

    fake_video_path = '/tmp/fake.mp4'

    phrases = list(
        read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT'].values()
    )

    fake_sessions = []
    bad_phrases = []
    for i in range(num_sessions):
        session = []
        cloning_template = cloning_templates[i]

        for phrase in phrases:

            # randomise wav2lip tts settings
            slow = True if random.random() < 0.5 else False
            accent = random.choice(ACCENTS)

            response = requests.post(
                f'http://127.0.0.1:8000/generate?phrase={phrase}&slow={slow}&accent={accent}',
                files={'video_file': open(cloning_template[2], 'rb')},
            )
            if response.status_code != HTTPStatus.OK:
                print(response.content)
                continue
            with open(fake_video_path, 'wb') as f:
                f.write(response.content)

            template = create_template(fake_video_path, debug=True)
            if template:
                session.append((phrase, template))
            else:
                bad_phrases.append(phrase)

        fake_sessions.append((f'fake_{i+1}', session))

    bad_phrases = set(bad_phrases)
    print('Bad Phrases:', bad_phrases)

    # removing bad phrases
    fake_sessions = [
        (s[0], [(phrase, template.blob)
                for phrase, template in s[1]
                if phrase not in bad_phrases])
        for s in fake_sessions
    ]

    for session in fake_sessions:
        assert len(session[1]) == len(set(phrases) - bad_phrases)

    training_test_split = int(len(other_templates) * 0.7)
    training_templates = [(t[0], t[1])
                          for t in other_templates[:training_test_split]
                          if t[0] not in bad_phrases]
    test_templates = [(t[0], t[1])
                      for t in other_templates[training_test_split:]
                      if t[0] not in bad_phrases]
    default_templates = [(t[0], t[1].blob)
                         for t in default_templates
                         if t[0] not in bad_phrases]

    selected_session_labels = \
        session_selection_with_cross_validation_fast(fake_sessions,
                                                     training_templates)
    print(f'Selected {len(selected_session_labels)} sessions')
    print(selected_session_labels)
    selected_sessions = [s for s in fake_sessions
                         if s[0] in selected_session_labels]
    ref_templates = sessions_to_templates(selected_sessions)

    session_accuracies = get_accuracy(ref_templates, test_templates)[0]
    print('Selected:', session_accuracies)

    default_accuracies = get_accuracy(default_templates, test_templates)[0]
    print('Default:', default_accuracies)

    # TODO: Do session selection with augmentation and compare



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')

    main(parser.parse_args())
