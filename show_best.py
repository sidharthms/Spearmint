import os
import sys

__author__ = 'Sidharth Mudgal'

import argparse
import importlib
import pdb
import numpy as np
from collections import OrderedDict
from spearmint.utils.database.mongodb import MongoDB
from spearmint.main import get_options, parse_resources_from_config, load_jobs, remove_broken_jobs, \
    load_task_group, load_hypers


# Converts a vector in input space to the corresponding dict of params
def paramify(task_group, data_vector):
    if data_vector.ndim != 1:
        raise Exception('Input to paramify must be a 1-D array.')

    params = OrderedDict()
    for name, vdict in task_group.dummy_task.variables_meta.iteritems():
        indices = vdict['indices']
        params[name] = {}
        params[name]['type'] = vdict['type']

        if vdict['type'] == 'int' or vdict['type'] == 'float':
            params[name]['values'] = data_vector[indices]
        elif vdict['type'] == 'enum':
            params[name]['values'] = []
            for ind in indices:
                params[name]['values'].append(vdict['options'][data_vector[ind].argmax(0)])
        else:
            raise Exception('Unknown parameter type.')

    return params

def main(filter=None):
    """
    Usage: python make_plots.py PATH_TO_DIRECTORY
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='remove broken jobs')
    parser.add_argument('--table', action='store_true', help='print table')
    parser.add_argument('--csv', action='store_true', help='save table as csv')
    parser.add_argument('--d', type=int, help='sort by distance from dth smallest result')
    parser.add_argument('--name', help='experiment name', default=None)
    args, unknown = parser.parse_known_args()

    options, expt_dir = get_options(unknown)
    # print "options:"
    # print_dict(options)

    # reduce the grid size
    options["grid_size"] = 400

    resources = parse_resources_from_config(options)

    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(options)
    # print "chooser", chooser
    if args.name:
        experiment_name = args.name
    else:
        experiment_name     = options.get("experiment-name", 'unnamed-experiment')

    # Connect to the database
    db_address = options['database']['address']
    # sys.stderr.write('Using database at %s.\n' % db_address)
    db         = MongoDB(database_address=db_address)

    # testing below here
    jobs = load_jobs(db, experiment_name)
    print len(jobs), 'jobs found'
    # print jobs

    # remove_broken_jobs
    if args.clean:
        for job in jobs:
            if job['status'] == 'pending':
                sys.stderr.write('Broken job %s detected.\n' % job['id'])
                job['status'] = 'broken'
                db.save(job, experiment_name, 'jobs', {'id' : job['id']})

    # print "resources:", resources
    # print_dict(resources)
    resource = resources.itervalues().next()

    task_options = {task: options["tasks"][task] for task in resource.tasks}
    # print "task_options:"
    # print_dict(task_options) # {'main': {'likelihood': u'NOISELESS', 'type': 'OBJECTIVE'}}

    task_group = load_task_group(db, options, experiment_name, resource.tasks)
    hypers = load_hypers(db, experiment_name)
    chooser.fit(task_group, hypers, task_options)
    lp, x = chooser.best()

    if args.table:
        os.chdir(unknown[0])
        out_file = open('results.csv', 'w') if args.csv else sys.stdout

        # get the observed points
        task = task_group.tasks.itervalues().next()
        idata = task.valid_normalized_data_dict
        inputs = idata["inputs"]
        inputs = map(lambda i: [paramify(task_group, task_group.from_unit(i)).values(), i], inputs)
        vals = idata["values"]
        vals = [task.unstandardize_mean(task.unstandardize_variance(v)) for v in vals]

        out_file.write('\n%10s' % 'result')
        lengths = [10]
        for name, vdict in task.variables_meta.iteritems():
            name = '%10s' % name
            out_file.write(',' + name)
            lengths.append(len(name))
        out_file.write('\n')

        line_template = '%' + str(lengths[0]) + '.4f,' + ','.join(['%' + str(l) +
            ('.4f' if 'enum' not in inputs[0][0][i]['type'] else 's') for i, l in enumerate(lengths[1:])])

        points = sorted(zip(vals, inputs), key=lambda r: r[0])
        if args.d is not None:
            target = x
            if args.d >= 0:
                target = points[args.d][1][1]
            points = sorted(points, key=lambda r: np.linalg.norm(r[1][1] - target))
        for i, point in enumerate(points):
            subs = [point[0]] + [d['values'][0] for d in point[1][0]]
            out_file.write(line_template % tuple(subs) + '\n')
        out_file.close()


if __name__ == "__main__":
    main()
