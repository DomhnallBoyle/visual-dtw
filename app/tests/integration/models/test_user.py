import pytest
from main.models import Config, PAVAList, PAVAPhrase, PAVATemplate, PAVAUser
from main.utils.db import db_session
from main.utils.exceptions import ListNotFoundException, UserNotFoundException


class TestUser:

    def test_delete_user(self):
        user = PAVAUser.create(default_list=True)

        user_id = user.id
        config_id = user.config_id

        with db_session() as s:
            list_id = PAVAList.get(s, filter=(PAVAList.user_id == user_id),
                                   first=True).id

        PAVAUser.delete(id=user_id)

        with pytest.raises(UserNotFoundException), db_session() as s:
            PAVAUser.get(s, filter=(PAVAUser.id == user_id), first=True)

        with pytest.raises(ListNotFoundException), db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

        with db_session() as s:
            templates = PAVATemplate.get(s, filter=(
                (PAVATemplate.phrase_id == PAVAPhrase.id)
                & (PAVAPhrase.list_id == PAVAList.id)
                & (PAVAList.user_id == user_id)
            ))
            assert len(templates) == 0

            assert len(Config.get(s, filter=(Config.id == config_id))) == 0
