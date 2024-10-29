import numpy as np
from main import configuration
from main.models import PAVAList
from main.research.utils import generate_dtw_params
from main.utils.cluster import Clustering
from main.utils.exceptions import IncorrectPhraseException, \
    WeakTemplateException
from main.utils.io import read_pickle_file


def construct_clusters(lst):
    return {
        p.content: np.asarray([t for t in p.templates])
        for p in lst.phrases if p.content != 'None of the above'
    }


def label_template(selected_phrase, template_to_add):
    """

    We need to compare with all phrase clusters because if we didn't, we
    could create another phrase and add a template to it that is similar to
    other templates of a different phrase

    Args:
        phrase:
        template_to_add:

    Returns:

    """
    # get list phrase belongs to
    lst = PAVAList.get(filter=(PAVAList.id == selected_phrase.list_id),
                       first=True)

    if lst.default:
        # get default list from file
        default_list = read_pickle_file(configuration.DEFAULT_LIST_PATH)

        # add default templates from default list
        base_templates = construct_clusters(lst=default_list)

        # add any templates user has added themselves
        for phrase in lst.phrases:
            for template in phrase.templates:
                base_templates[phrase.content] = \
                    np.hstack(base_templates[phrase], template)
    else:
        base_templates = construct_clusters(lst=lst)

    # print(base_templates)

    dtw_params = generate_dtw_params()

    clustering = Clustering(clusters=base_templates, **dtw_params)
    if clustering.is_phrase_correct(uttered_phrase=selected_phrase.content,
                                    template=template_to_add):
        if clustering.is_suitable(closest_phrase=selected_phrase.content,
                                  template=template_to_add):
            return
        else:
            raise WeakTemplateException
    else:
        raise IncorrectPhraseException
