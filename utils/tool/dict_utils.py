def merge_dicts(dict_list):
    # 合并字典，如果有重复的key，后面的会覆盖前面的
    result = {}
    for d in dict_list:
        result.update(d)
    return result
