import os
import os.path as opath
import pandas as pd
import PointSet


def file_system_scrawl(root_dir, ext=None):
    if opath.isfile(root_dir):
        yield root_dir
    elif opath.isdir(root_dir):
        for sub_path in sorted(os.listdir(root_dir)):
            p = opath.join(root_dir, sub_path)
            if opath.isdir(p):
                yield from file_system_scrawl(p)
            elif opath.isfile(p):
                if ext is None or (len(opath.splitext(sub_path)) == 2 and opath.splitext(sub_path)[1] == ext):
                    yield p


_cache = dict()


def read_file(file_path):
    if file_path not in _cache:
        ps = PointSet.MultivariatePointSet()
        raw_df = pd.read_csv(file_path, sep='\t')
        for _, row in raw_df.iterrows():
            for col in raw_df.columns:
                if col == 'Cell.X.Position' or col == 'Cell.Y.Position' or col == 'Other':
                    continue
                if row[col] == 'pos':
                    ps.add_point(float(row['Cell.X.Position']), float(row['Cell.Y.Position']), col)
        ps.build_index()
        _cache[file_path] = ps
    return _cache[file_path]
