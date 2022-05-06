import re


class NutritionLabelMap(dict):
    """
    A class that generates a map for aliases of a nutritional row label
    to their corresponding key for use in the JSON output
    """

    # the label map is generated from the dict below
    # labels and aliases must be in lowercase form with spaces replaced with '-'
    labels = {
        'energy': ['energy'],
        'protein': ['protein'],
        'fat-total': ['fat', 'fat-total', 'total-fat'],
        'fat-saturated': ['saturated', 'fat-saturated', 'saturated-fat'],
        'fat-trans': ['fat-trans', 'trans', 'trans-fat'],
        'fat-poly': ['polysaturated', 'poly', 'poly-fat', 'fat-poly', 'polysaturated-fat', 'fat-polysaturated'],
        'fat-mono': ['mono', 'monosaturated', 'monosaturated-fat', 'fat-monosaturated'],
        'carbohydrate': ['carbohydrate'],
        'sugars': ['sugars'],
        'sodium': ['sodium'],
        'fibre': ['fibre', 'dietary fibre'],
        'potassium': ['potassium'],
        'cholesterol': ['cholesterol']
    }

    def __init__(self):
        super(NutritionLabelMap, self).__init__()
        # invert the label dict to generate alias->label map
        for label, aliases in NutritionLabelMap.labels.items():
            for alias in aliases:
                self[alias] = label

    def __getitem__(self, label):
        key = _clean(label)
        return super(NutritionLabelMap, self).__getitem__(key)

    def __contains__(self, item):
        try:
            return bool(self[item])
        except KeyError:
            return False

    def get(self, k):
        if k in self:
            return self[k]
        else:
            return None


def _clean(string):
    string = string.lower().strip()
    string = re.sub(r'[^a-z]', " ", string)
    string = string.strip()
    string = re.sub(r'\s\s+', " ", string)
    string = re.sub(r' ', "-", string)
    return string


if __name__ == '__main__':
    a = NutritionLabelMap()
    print(a.get('total fat'))
