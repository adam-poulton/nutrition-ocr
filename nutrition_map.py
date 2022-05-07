import re


class NutritionLabelMap(dict):
    """
    A class that generates a map for aliases of a nutritional row label
    to their corresponding key for use in the JSON output
    """

    # the label map is generated from the dict below
    # labels and aliases must be in lowercase form with spaces replaced with '-'
    labels = {
        'energy': {'default_unit': 'kJ',
                   'aliases': []},
        'protein': {'default_unit': 'g',
                    'aliases': []},
        'fat-total': {'default_unit': 'g',
                      'aliases': ['fat', 'total-fat']},
        'fat-saturated': {'default_unit': 'g',
                          'aliases': ['saturated', 'saturated-fat']},
        'fat-trans': {'default_unit': 'g',
                      'aliases': ['trans', 'trans-fat']},
        'fat-poly': {'default_unit': 'g',
                     'aliases': ['polyunsaturated', 'poly', 'poly-fat', 'polyunsaturated-fat', 'fat-polyunsaturated']},
        'fat-mono': {'default_unit': 'g',
                     'aliases': ['mono', 'monounsaturated', 'monounsaturated-fat', 'fat-monounsaturated']},
        'carbohydrate': {'default_unit': 'g',
                         'aliases': []},
        'sugars': {'default_unit': 'g',
                   'aliases': []},
        'sodium': {'default_unit': 'mg',
                   'aliases': ['sooium']},
        'fibre': {'default_unit': 'g',
                  'aliases': ['dietary-fibre']},
        'potassium': {'default_unit': 'mg',
                      'aliases': ['potassium']},
        'cholesterol': {'default_unit': 'g',
                        'aliases': []},
        'gluten': {'default_unit': 'g',
                   'aliases': ['qluten']}
        }

    def __init__(self):
        super(NutritionLabelMap, self).__init__()
        self.defaults = {}
        # invert the label dict to generate alias->label map
        for label, data in NutritionLabelMap.labels.items():
            self[label] = label
            self.defaults[self[label]] = data['default_unit']  # store the default unit for each label
            for alias in data['aliases']:
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

    def default_unit(self, label):
        return self.defaults[self[label]]


def _clean(string):
    string = string.lower().strip()
    string = re.sub(r'[\d.,%]+[a-zA-Z]{0,3}', " ", string)
    string = re.sub(r'[^a-z]', " ", string)
    string = string.strip()
    string = re.sub(r'\s\s+', " ", string)
    string = re.sub(r' ', "-", string)
    return string


if __name__ == '__main__':
    a = NutritionLabelMap()
    print(_clean('Saturated  0.1g'))
