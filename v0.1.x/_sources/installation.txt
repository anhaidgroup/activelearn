============
Installation
============
 
Requirements
------------
    * Python 2.7 or Python 3.4+

Platforms
------------
activelearn has been tested on Linux (Ubuntu with  Kernel Version 3.13.0-40-generic), OS X (Darwin with Kernel Version 13.4.0), and Windows 8.1.

Dependencies
------------
    * pandas (provides data structures to store and manage tables)
    * scikit-learn (provides implementations for common machine learning algorithms)
    * six (to ensure our code run on both Python 2.x and Python 3.x)

.. note::

     The activelearn installer will automatically install the above required packages. 

There are two ways to install activelearn package: using pip or source distribution.

Installing Using pip
--------------------
The easiest way to install the package is to use pip, which will retrieve activelearn from PyPI then install it::

    pip install activelearn
    
Installing from Source Distribution
-------------------------------------
Step 1: Download the source code of the activelearn package from `here
<https://github.com/anhaidgroup/activelearn/releases>`_. (Download code in tar.gz format for Linux and OS X, and code in zip format for Windows.)

Step 2: Untar or unzip the package and execute the following command from the package root::

    python setup.py install
    
.. note::

    The above command will try to install activelearn into the defaul Python directory on your machine. If you do not have installation permission for that directory then you can install the package in your home directory as follows::

        python setup.py install --user

    For more information see the following StackOverflow `link
    <http://stackoverflow.com/questions/14179941/how-to-install-python-packages-without-root-privileges>`_.
