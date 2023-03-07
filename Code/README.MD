# Codebase for CA Clustering
---
### Running and getting results

Currently, the script runs 20 trials and outputs the results into a textfile ```../Results/20trial_output.txt```

## The code now supports multithreading and command line arguments 
** The code also outputs into CSV files for easier inspection and analysis
Run the following command : 

    ``` python3 ./CA_script.py {num of threads} {num of trials}````

###  example:
 
    ``` python3 ./CA_script.py 2 3 ```
where the first argument is the number of threads to run i.e. 2, in our example above, and second is the number of trials from the list of all combinations of rule list i.e. 3 in our example above. 

** Note **
To run an exhaustive search, pass ```-1``` as the argument for {num of trials}


*************** DEPRECATED ************************
Run the following script to get the output: 

```python3 ./CA_script.py > ../Results/20trial_output.txt```




