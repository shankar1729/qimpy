import yaml


class Qimpy(object):
    def __init__(self, yamlDict):
        '''
        Input: dictionary yamlDict generated from yaml read-in
        Instantiates Qimpy object whose attributes are keys of yamlDict
        see: https://stackoverflow.com/questions/1305532/
            convert-nested-python-dict-to-object
        '''
        for inputKeyword, inputValue in yamlDict.items():  # for key, value
            if isinstance(inputValue, (list, tuple)):
                setattr(
                    self, inputKeyword,
                    [(Qimpy(x) if isinstance(x, dict) else x)
                        for x in inputValue])
            else:
                setattr(
                    self, inputKeyword,
                    (Qimpy(inputValue) if isinstance(inputValue, dict)
                        else inputValue))


if __name__ == "__main__":

    # Parse the commandline arguments:
    import argparse
    parser = argparse.ArgumentParser(
        description='Run a QimPy calculation from an input file')
    parser.add_argument(
        'input_file',
        help='input file in YAML format')
    parser.add_argument(
        '-o', '--output-file',
        help='output file (stdout if unspecified)')
    parser.add_argument(
        '-c', '--cores', type=int,
        help='number of cores per process (overridden by SLURM)')
    parser.add_argument(
        '-n', '--dry-run', action='store_true',
        help='quit after initialization (to check input file)')
    parser.add_argument(
        '-d', '--no-append', action='store_true',
        help='overwrite output file instead of appending')
    parser.add_argument(
        '-m', '--mpi-log',
        help='file prefix for debug logs from other MPI processes')
    parser.add_argument(
        '-v', '--version', action='store_true',
        help='print version information and quit')
    parser.add_argument(
        '-V', '--verbose', action='store_true',
        help='print extra information in log for debugging')
    args = parser.parse_args()
    print(args)
    
    with open(args.input_file) as f:
        allInputs = yaml.safe_load(f)  # dictionary of inputs from yaml file

    qimpy = Qimpy(allInputs)
    print(qimpy.__dict__)
    print(qimpy.lattice)
    print(qimpy.ions)
