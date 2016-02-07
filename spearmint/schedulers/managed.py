import multiprocessing
import os
import sys
import pdb
from abstract_scheduler import AbstractScheduler
from spearmint import launcher

__author__ = 'sidharth'


class Global:
    process_cache = {}
    my_gpu = None

def process_initializer(first_available_pid, gpus, simulate):
    import theano.sandbox.cuda
    with first_available_pid.get_lock():
        Global.my_gpu = int(first_available_pid.value) % gpus
        if not simulate:
            theano.sandbox.cuda.use('gpu' + str(Global.my_gpu))
        first_available_pid.value += 1
        print 'Process ', os.getpid(), 'took gpu', Global.my_gpu


def launch(**kwargs):
    print 'Launching task on process with gpu', Global.my_gpu
    if kwargs['output_filename'] is not None:
        output_file = open(kwargs['output_filename'], 'w')
        old_stdout = sys.stdout
        sys.stdout = output_file

    launcher.launch(kwargs['database_address'], kwargs['experiment_name'], kwargs['job_id'], Global.process_cache)

    if kwargs['output_filename'] is not None:
        # cleanup
        sys.stdout = old_stdout
        output_file.close()


class ExperimentRunner:
    def __init__(self, config):
        self.concurrent = config.get('max-concurrent', 1)
        self.gpus = config.get('gpus', 2)
        self.simulate = config.get('simulate_gpus', False)
        self.test = config.get('test', False)
        self._first_available_pid = multiprocessing.Value('i', config.get('first_gpu', 0))
        self._pool = None

    def schedule_job(self, **kwargs):
        if not self.test:
            if self._pool is None:
                print 'GPU count =', self.gpus
                print 'Process/GPU =', self.concurrent / self.gpus
                self._pool = multiprocessing.Pool(self.concurrent, process_initializer,
                    (self._first_available_pid, self.gpus, self.simulate))
            self._pool.apply_async(launch, (), kwargs)
        else:
            if not self.simulate:
                import theano.sandbox.cuda
                theano.sandbox.cuda.use('gpu0')
            kwargs['output_filename'] = None
            launch(**kwargs)
        return 1


def init(*args, **kwargs):
    return ManagedScheduler(*args, **kwargs)


class ManagedScheduler(AbstractScheduler):
    def __init__(self, config):
        super(ManagedScheduler, self).__init__(config)
        self.driver = ExperimentRunner(config)

    def submit(self, job_id, experiment_name, experiment_dir, database_address):
        output_directory = os.path.join(experiment_dir, 'output')
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        # allow the user to specify a subdirectory for the output
        if "output-subdir" in self.options:
            output_directory = os.path.join(output_directory, self.options['output-subdir'])
            if not os.path.isdir(output_directory):
                os.mkdir(output_directory)

        output_filename = os.path.join(output_directory, '%08d.out' % job_id)

        return self.driver.schedule_job(output_filename=output_filename, database_address=database_address,
                                        experiment_name=experiment_name, job_id=job_id)

    def alive(self, process_id):
        return True
