def clean_result(result):
    result=dict(result)
    result.pop('model',None)
    return result
