if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from pitch_detection import get_notes

    hypo = np.zeros(200, dtype=int)
    notes = np.empty(0, dtype=int)
    persons = []
    with open('./results/worst30.txt') as fd:
        persons = [line.split('\'')[1] for line in fd.readlines()]
    fd.close()

    for person in persons:
        print(person)
        for root, dirs, files in os.walk('../DataSet/MIR-QBSH-corpus/waveFile/{}'.format(person)):
            for file in files:
                try:
                    if '.wav' not in file:
                        continue
                    ffile = os.path.join(root, file)
                    query_notes, query_onsets = get_notes(ffile)
                    for note in query_notes:
                        hypo[note] += 1
                    notes = np.append(query_notes, notes)
                except:
                    continue
            print(hypo)
            hypo = np.zeros(200, dtype=int)
            # print(notes)
    '''
    x = list(range(200))
    plt.bar(x, hypo)
    plt.xlabel('Note')
    plt.ylabel('Counts')
    plt.title('Distribution')
    plt.show()
    '''
