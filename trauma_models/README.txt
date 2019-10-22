The adjustable_values page is the only page you should need to change and access values. .

This model contains 4 iterations of the model design, and handles 4 layers of complexity in approaching the
subject matter.

vertical_only = only observing heritable transmission
vertical_age_dpt = heritable transmission integrated with age dependent values for ptsd
vertical_age_dpt_horizontal = vertical, age dependent trauma, and horizontal transmission
vertical_age_dpt_leaders = all of the above, with the added complexity of leaders

Each iteration can be used by inputting the string into the required area on the adjustable_values page

The server function is only necessary if you need to visualise a run, if so then unhash the server.

The data analysis is integrated into the application, and it is possible, and encouraged to run the model and then
produce statistical output immediately after within the same action. The data analysis component have sections at the
bottom of each data analysis page, there is where you structure the kinds of input you would like to produce from
the run. There are default values you can use, to see how they work. Data analysis is handled differently for each type
of analysis, and has it's own script: sensitivity analysis, single analysis, and comparative analysis.

Note the folder structure, everything has its place.

The server handles the visualisation. To run the model interactively, activate this line of code by unhashing.

The model tests and demonstrates several Mesa concepts and features:

    MultiGrid
    Multiple file programming
    Inheriting values

Python concepts and libraries:
    Data visualisation
    pandas
    Seasborn
    statistics


Installation

To install the dependencies use pip and the requirements.txt in this directory. e.g.

    $ pip install -r requirements.txt

How to Run
