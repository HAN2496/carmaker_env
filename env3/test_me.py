def save_data_for_lowlevel(indexs, values):
    datas = {}
    for idx, index in enumerate(indexs):
        datas[index] = values[idx]
    return datas

indexs = ['a', 'b']
values = [1, 2]
d = save_data_for_lowlevel(indexs, values)
print(d['a'])
