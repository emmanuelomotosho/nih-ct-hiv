#!/usr/bin/env python3

import os
import pickle
import sqlite3
import sys
import xml.etree.ElementTree as ET

DATABASE = 'studies.sqlite'
METAMAP_XML_DIR = 'metamap_out'

def extract_cuis(study_id):
    """
    Returns a a list of the CUIs extracted from the study's MetaMap XML file
    """
    with open(os.path.join(METAMAP_XML_DIR, study_id + '.xml')) as f:
        f.readline()  # workaround for non-XML first line
        root = ET.fromstring(f.read())
        cuis = []

        # for x in root.find('.//Negations'):
        #     neg_type = x.find('NegType').text
        #     neg_trigger = x.find('.//NegTriggerPI')
        #     neg_pos = int(neg_trigger.find('StartPos').text)
        #     neg_length = int(neg_trigger.find('Length').text)
        #     features[neg_pos] = (neg_type, neg_length)
        #     names[neg_type] = neg_type
        #     ncui = x.find('.//NegConcCUI').text

        for phrase in root.findall('.//Phrase'):
            mappings = phrase.findall('.//Mapping[1]')
            if len(mappings):
                for mapping in mappings:
                    for candidate in mapping.findall('.//Candidate'):
                        cui = candidate.find('CandidateCUI').text
                        concept = candidate.find('CandidatePreferred').text
                        # sem_types = set([st.text for st in candidate.findall('.//SemType')])
                        if int(candidate.find('Negated').text) == 1:
                            cui = 'N' + cui
                        cuis.append(cui)
        return cuis

if __name__ == '__main__':
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        'SELECT t1.NCTId FROM studies AS t1, hiv_status AS t2 WHERE t1.NCTId=t2.NCTId ORDER BY t1.NCTId')
    data = {}
    i = 0
    for row in c.fetchall():
        study_id = row[0]
        data[study_id] = extract_cuis(study_id)
        i += 1
        sys.stderr.write(str(i) + '\n')
    pickle.dump(data, sys.stdout.buffer)

