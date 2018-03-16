import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

X = np.load('feature_125_1000.npy')
y = np.load('name_125_1000.npy')
y = [int(a.split('_')[-1].split('.')[0]) for a in y]
uni_id = np.unique(y)
# y = [np.where(uni_id==a)[0].tolist()[0] for a in y]
y = np.array(y)


t_cnt = 0
num_test = 10
for test in range(num_test):
    ex = []
    gallery = []
    for idx in uni_id:
        exist_list = np.where(y==idx)[0]
        l = np.random.permutation(len(exist_list))
        l = l[0]
        gallery.append(np.array(X[exist_list[l], :]).reshape(1, -1))
        ex.append(exist_list[l])

    cnt = 0
    for i, f in enumerate(X):
        print(test, i, len(y))
        if i in ex:
            continue

        f = np.array(f).reshape(1, -1)
        idx = y[i]
        
        score = [cosine_similarity(g, f)[0][0] for g in gallery]
        print(score)
        import g

        predict_idx = uni_id[np.argmax(score)]
        if predict_idx == idx:
            cnt += 1
    t_cnt += (cnt)/(len(y)-len(uni_id))
    print((cnt)/(len(y)-len(uni_id)))
print(t_cnt/num_test)

'''
import h

mean1 = []
mean2 = []

for i, x in enumerate(X):

    b = x.reshape(1, -1)

    if y[i] == 0:
        mean1.append(cosine_similarity(a, b)[0][0])
    else:
        mean2.append(cosine_similarity(a, b)[0][0])

AA = (np.sum(mean1)-1)/(len(mean1)-1)
BB = np.mean(mean2)
print('idx=%d'%idx)
print(AA)
print(BB)
print()
if AA>BB:
    cnt += 1
'''

# print(cnt/70)

