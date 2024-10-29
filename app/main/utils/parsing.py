"""Parsing utilities.

Contains methods for parsing template file names
"""
import re
import string

TEMPLATE_REGEX = r'0*(\d+)_(S\w?[A-Z])0*(\d+)_S0*(\d+)'
TEMPLATE_REGEX_FROM_DEFAULT_LIST = r'AE_norm_2_(\d+)_(S\w?[A-Z])(\d+)_(\d+)'


def extract_template_info(template_id, from_default_list=False):
    regex = TEMPLATE_REGEX
    if from_default_list:
        regex = TEMPLATE_REGEX_FROM_DEFAULT_LIST

    user_id, phrase_set, phrase_id, session_id = \
        re.match(regex, template_id).groups()

    return int(user_id), phrase_set, phrase_id, int(session_id)


def clean_string(s):
    s = s.lower()
    for c in string.punctuation:
        s = s.replace(c, '')

    return s
