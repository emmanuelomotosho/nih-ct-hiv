#!/usr/bin/env python3
# only prompts annotation for studies where RE analyzer returns True

import sqlite3
import sys
from time import sleep

from re_analyze import score_text as re_score_text


def _save_annotation(c, id, value):
    c.execute("INSERT OR REPLACE INTO hiv_status VALUES(?, ?)", [id, value])


def annotate_interactive(conn, id, allow_skip=False):
    c = conn.cursor()
    c.execute("SELECT EligibilityCriteria FROM studies WHERE NCTId=?", [id])
    print(id)
    print(c.fetchone()[0])
    v = None
    if allow_skip:
        prompt = 'Enter y/n/s --> '
    else:
        prompt = 'Enter y/n --> '
    while v is None:
        rv = input(prompt).lower()
        if rv == 'y':
            v = True
        elif rv == 'n':
            v = False
        elif allow_skip and rv == 's':
            return None
    _save_annotation(c, id, v)
    value = v
    conn.commit()
    return value


def print_status(conn, counter):
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM hiv_status')
    total = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM hiv_status WHERE hiv_eligible=1')
    eligible = c.fetchone()[0]
    print("Annotated %s this session, %s total, %s eligible" % (counter, total, eligible))

if __name__ == '__main__':
    conn = sqlite3.connect(sys.argv[1])
    c = conn.cursor()

    c.execute('SELECT NCTId, EligibilityCriteria FROM studies WHERE NOT EXISTS(SELECT * FROM hiv_status WHERE studies.NCTId=hiv_status.NCTId) ORDER BY random()')
    i = 0
    for row in c.fetchall():
        if re_score_text(*row):
            annotate_interactive(conn, row[0], allow_skip=True)
            i += 1
            print_status(conn, i)
            sleep(0.5)
