from sklearn.model_selection import KFold
import torch
from functools import reduce

def setConfigure(dict_configure):
    tuned_hp = dict_configure.keys()
    values_list = dict_configure.values()
    configure_sizes = list(map(lambda l: len(l),values_list))
    configure_num = reduce(lambda x,y: x*y , configure_sizes)
    div = [1]
    for size in configure_sizes:
        div.append(div[-1]*size)

    div = div[:-1]

    extract_idxs = lambda i: [(i//d)%s for s,d in zip(configure_sizes,div)]
    extract_configure = lambda ll, idxs: [l[i] for l, i in zip(ll,idxs)]

    configures = []
    
    for i in range(configure_num):
        configure_idxs = extract_idxs(i)
        configure = extract_configure(values_list, configure_idxs)
        curr_configure = dict(zip(tuned_hp,configure))
        configures.append(curr_configure)
    return configures

def crossValidate(model, X, y, kf):
    first = True
    #list_apply = lambda l1,l2,l3,l4,func: (func(l1,l3),func(l2,l4)) 
    #list_sum = lambda l1, l2: [x1+x2 for x1,x2 in zip(l1,l2)]
    folds = kf.get_n_splits()
    for (train_index, test_index) in kf.split(X):
        train_fold_res, test_fold_res = model.fit(X[train_index,:],y[train_index], X[test_index,:], y[test_index])
        if first:
            res_train = torch.tensor(train_fold_res)
            res_test = torch.tensor(test_fold_res)
            first = False
        else:
            res_train += torch.tensor(train_fold_res)
            res_test += torch.tensor(test_fold_res)
    #        res_train = list_apply(*res_train, *train_fold_res, list_sum)
    #        res_test = list_apply(*res_test, *test_fold_res, list_sum)

    #div = lambda l,d: [x/d for x in l]
    #return list_apply(*res_train, folds,folds, div), list_apply(*res_test, folds,folds, div)
    return (res_train/folds).tolist(), (res_test/folds).tolist()