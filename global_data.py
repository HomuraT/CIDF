### hmt
import json

from config import args

detailed_relation_description = None
detail2simple = None

def get_detailed_relation_description():
    global detailed_relation_description
    if detailed_relation_description:
        return detailed_relation_description

    if args.detailed_relation_description:
        with open(args.detailed_relation_description, 'r', encoding='utf-8') as f:
            detailed_relation_description = json.load(f)
            detailed_relation_description[''] = ''
        print(detailed_relation_description)
        return detailed_relation_description
    else:
        return None


def get_detail2simple():
    global detail2simple
    if detail2simple:
        return detail2simple

    _detailed_relation_description = get_detailed_relation_description()
    if _detailed_relation_description is None:
        return None
    detail2simple = {}
    for k, v in _detailed_relation_description.items():
        detail2simple[v] = k
    return detail2simple


### end hmt