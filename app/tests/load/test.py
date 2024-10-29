"""
Check out scripts/load_tests.sh
"""
import io
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from locust import HttpLocust, TaskSet, constant, task
from main import configuration
from main.models import PAVAList, PAVAUser
from main.utils.db import db_session

VIDEO_PATH = os.path.join(configuration.VIDEOS_PATH, 'move_me.mp4')


class UserBehaviour(TaskSet):

    user_id = None
    list_id = None
    f = None

    def on_start(self):
        user = PAVAUser.create(default_list=True)
        with db_session() as s:
            lst = PAVAList.get(s, filter=(PAVAList.user_id == user.id),
                               first=True)

        self.user_id = user.id
        self.list_id = lst.id
        self.f = open(VIDEO_PATH, 'rb')

    def on_stop(self):
        PAVAUser.delete(id=self.user_id)
        self.f.close()

    @task
    def transcribe(self):
        self.client.post(
            f'pava/api/v1/lists/{self.list_id}/transcribe/video',
            files={'file': ('video.mp4', io.BytesIO(self.f.read()))},
            verify=configuration.CFE_VERIFY
        )

        # put cursor to start of file
        self.f.seek(0)


class User(HttpLocust):
    task_set = UserBehaviour
    wait_time = constant(1)


def show_results(files):
    dfs = []

    for filename in files:
        df = pd.read_csv(filename)
        aggregated_row = df.loc[df['Name'] == 'Aggregated']
        aggregated_row['Num Users'] = filename.split('_')[0]
        dfs.append(aggregated_row)

    df = pd.concat(dfs)
    df.plot(kind='bar', x='Num Users', y=['Average response time', '90%'],
            rot=0)
    plt.ylabel('Response Time (ms)')
    plt.show()


if __name__ == '__main__':
    results_files = sys.argv[1:]
    show_results(results_files)
