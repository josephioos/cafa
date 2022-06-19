import torch

def match_string(stra, strb):
    ''' 
        stra: labels.
        strb: unlabeled data predicts.
    '''
    l_b, prob = torch.argmax(strb, dim=0), torch.max(strb, dim=0).values
    print('l_b', l_b)
    print('prob', prob)
    permidx = torch.tensor(range(len(l_b)))
    # for i in range(len(l_b)):
    #     if stra[i] != l_b[i]:
    #         for j in range(i, len(l_b)):
    #             if stra[i] == l_b[j]:
    #                 tmp = permidx[i]
    #                 permidx[i] = permidx[j]
    #                 permidx[j] = tmp

    for i in range(len(l_b)):
        if stra[i] != l_b[i]:
            mask = (l_b[i:] == stra[i]).float()
            print('mask', mask)
            if mask.sum() > 0:
                idx_tmp = int(i + torch.argmax(prob[i:] * mask, dim=0))
                print('before: ', permidx[i], permidx[idx_tmp])
                tmp = permidx[i].data.clone()
                permidx[i] = permidx[idx_tmp]
                permidx[idx_tmp] = tmp
                print('i:', i, 'idx:', idx_tmp)
                print('after:', permidx[i], permidx[idx_tmp])
    return permidx

a = torch.randint(10, (10, ))
print('a', a)
b = torch.randn((10, 10))
print('b', b)
permidx = match_string(a, b)
print('permidx', permidx)
l_b = torch.argmax(b, dim=0)
perm_b = l_b[permidx]
print('perm b', perm_b)
