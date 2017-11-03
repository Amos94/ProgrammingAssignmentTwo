import collections


def tdd():
    return collections.defaultdict(tdd)

#For debug purposes
# d = tdd()
# d['this']['care'] = 1
# d['this']['win'] = 2
# d['is']['care'] = 3
# d['is']['win'] = 4
#
# print(d)
# print('\n---\n')
# print(d['this'])
# print('\n---\n')
# print(d['is']['win'])
# print('\n---\n')
# for word in d['this']:
#     print(word)
#     print(d['this'][word])