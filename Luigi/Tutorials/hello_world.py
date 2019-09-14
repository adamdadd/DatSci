#!/usr/bin/env python

import sciluigi

class MyFooWriter(sciluigi.Task):
    # We have no inputs here
    # Define outputs:
    def out_foo(self):
        return sciluigi.TargetInfo(self, 'foo.txt')


    def run(self):
        with self.out_foo().open('w') as foofile:
            foofile.write('foo\n')


class MyFooReplacer(sciluigi.Task):
    replacement = sciluigi.Parameter() # Here, we take as a parameter
                                  # what to replace foo with.
    # Here we have one input, a "foo file":
    in_foo = None
    # ... and an output, a "bar file":
    def out_replaced(self):
        # As the path to the returned target(info), we
        # use the path of the foo file:
        return sciluigi.TargetInfo(self, self.in_foo().path + '.bar.txt')
    def run(self):
        with self.in_foo().open() as in_f:
            with self.out_replaced().open('w') as out_f:
                # Here we see that we use the parameter self.replacement:
                out_f.write(in_f.read().replace('foo', self.replacement))


class MyWorkflow(sciluigi.WorkflowTask):
    def workflow(self):
        foowriter = self.new_task('foowriter', MyFooWriter)
        fooreplacer = self.new_task('fooreplacer', MyFooReplacer,
            replacement='bar')

        # Here we do the *magic*: Connecting outputs to inputs:
        fooreplacer.in_foo = foowriter.out_foo

        # Return the last task(s) in the workflow chain.
        return fooreplacer

# End of script ....
if __name__ == '__main__':
    sciluigi.run_local(main_task_cls=MyWorkflow)