
##############################################################################################################
#                                                                                                            #
#    File: convert_ember_to_parquet.py                                                                       #
#    Date: January 27, 2020                                                                                  #
#    Purpose:                                                                                                #
#      This file contains functionality to convert parts of the EMBER dataset to a Parquet table.            #
#                                                                                                            #
#    Copyright (c) 2020 Zachary Wilkins                                                                      #
#    COUGAR is licensed under the MIT Licence.                                                               #
#                                                                                                            #
##############################################################################################################

from sys import argv, exit, stderr
from json import loads
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from os.path import basename

# TABLE 1: GENERAL INFO (PK: md5)
# md5*,sha256,label,avclass,num_imports
INFO_COLUMNS = ['md5', 'sha256', 'label', 'avclass', 'num_imports']

# TABLE 2: IMPORTS (PK: md5+library+function)
# md5*,library*,function*,(entrypoint)
IMPORTS_COLUMNS = ['md5', 'library', 'function']


def main():
    if len(argv) <= 2:
        # E.g.: 2018 ../EMBER/ember2018/train_features_2.jsonl
        print('Usage: python3 convert_ember_to_parquet.py <year> <path_to_jsonl_file>', file=stderr)
        exit(-1)

    year = argv[1]
    files = [f for f in argv[2:]]

    for file in files:
        file_wo_path = basename(file).split('.')[0]
        general_info_file = '../EMBER/Parquet/{}_{}_info.parquet'.format(file_wo_path, year)
        imports_file = '../EMBER/Parquet/{}_{}_imports.parquet'.format(file_wo_path, year)
        info_lines = list()
        import_lines = list()

        with open(file, 'r') as f:
            for line in tqdm(f):
                json_doc = loads(line)
                md5 = json_doc['md5']
                imports = json_doc['imports']
                label = json_doc['label']
                if label == 1:
                    count = 0
                    for library in imports:
                        for function in imports[library]:
                            count += 1
                            import_lines.append([md5, library, function])
                    info_lines.append([
                        md5,
                        json_doc['sha256'],
                        str(label),
                        json_doc['avclass'],
                        str(count)
                    ])

        # Transform to in-memory Apache Arrow tables, then export as Apache Parquet files
        info_df = pa.Table.from_pandas(pd.DataFrame(info_lines, columns=INFO_COLUMNS))
        pq.write_table(info_df, general_info_file)
        imports_df = pa.Table.from_pandas(pd.DataFrame(import_lines, columns=IMPORTS_COLUMNS))
        pq.write_table(imports_df, imports_file)


if __name__ == '__main__':
    main()
