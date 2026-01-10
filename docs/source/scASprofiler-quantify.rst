==============
scASprofiler-quantify CLI
==============

The scASprofiler-impute CLI is used to calcluate AS probability

you can generate a AS probability matrix by the command line like this:

.. code-block:: bash
    
    scASprofiler-quantify --input-file scASP-scasp_drop_0.1.csv --oudir ./



Options
=======

There are more parameters for setting (``scASprofiler-quantify --help`` always give the version 
you are using):

.. code-block:: html

    parameter settings

    Options:
        input-file: Input CSV file containing splice junction data

        outdir: Output directory to save results
