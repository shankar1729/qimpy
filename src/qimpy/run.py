import yaml

class  Qimpy(object):
    def __init__(self, yamlDict): 
        ''' 
        Input: dictionary yamlDict generated from yaml read-in
        Instantiates Qimpy object whose attributes are the keys of the yaml dict
        see: https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object
        '''
        for inputKeyword, inputValue in yamlDict.items(): # for key, value
            if isinstance(inputValue, (list, tuple)): # is value a list or tuple?
                setattr(self, inputKeyword, [Qimpy(x) if isinstance(x, dict) else x for x in inputValue])
            else:
                setattr(self, inputKeyword, Qimpy(inputValue) if isinstance(inputValue, dict) else inputValue)
           


if __name__ == "__main__":
    print('''
        This will serve as the main QimPy executable.
        After installation, we should be able to run as
        
            python -m qimpy.run <arguments>
            
        from anywhere in the python path. Without install,
        this will only work from the src/ path. During
        development, set up for testing by running:

            python setup.py develop --user

        in the root directory of the repository.
    ''')

    fileQimpy = 'qimpy_input_test.yaml'
    with open(fileQimpy) as f:
        allInputs = yaml.safe_load(f) # dictionary of inputs from yaml file

    qimpy = Qimpy(allInputs)
    print(qimpy.__dict__)
    print(qimpy.Lattice)
    print(qimpy.IonCoordinates)

